# src/trading/virtual_trader.py
"""
Виртуальный трейдер — симуляция сделок для расчёта PR и статистики сценариев.

Ключевые особенности:
- Работает независимо от реальной торговли (даже для монет вне whitelist)
- Поддерживает partial TP с учётом minNotional биржи (динамический вызов через ccxt)
- Если позиция слишком мала для частичной фиксации — закрывает весь остаток одним ордером
- Учитывает комиссии (round-trip), slippage (моделируем 0.05–0.2%)
- Обновляет ScenarioTracker после каждой закрытой сделки
- Хранит историю виртуальных сделок для PR, винрейта и экспорта
- Продолжает считать PR фоном даже для исключённых монет
"""

import time
from typing import Dict, List, Optional
import numpy as np
import polars as pl
import ccxt

from src.trading.tp_sl_manager import TPSLManager
from src.trading.risk_manager import RiskManager
from src.model.scenario_tracker import ScenarioTracker
from src.data.storage import Storage
from src.utils.logger import get_logger
from src.core.config import load_config

logger = get_logger(__name__)

class VirtualTrader:
    def __init__(self, config: dict):
        self.config = config
        self.tp_sl_manager = TPSLManager(config)
        self.risk_manager = RiskManager(config)
        self.scenario_tracker = ScenarioTracker(config)
        self.storage = Storage(config)

        # Виртуальный баланс для симуляции P&L
        self.virtual_balance = config["finance"].get("deposit", 10000.0)

        # Открытые виртуальные позиции
        self.positions: Dict[str, Dict] = {}  # symbol → position dict

        # История закрытых сделок
        self.closed_trades: List[Dict] = []

        # Комиссия и slippage
        self.commission_rate = config["trading_mode"].get("commission_rate", 0.0004)  # 0.04% round-trip
        self.slippage_rate = config["trading_mode"].get("slippage_rate", 0.001)      # 0.1%

        # Кэш minNotional для символов (обновляется раз в 24 часа)
        self.min_notional_cache: Dict[str, float] = {}
        self.last_cache_update = 0

        # ccxt для получения minNotional
        self.exchange = ccxt.binance(config["exchange"])

    async def _get_min_notional(self, symbol: str) -> float:
        """Динамический вызов minNotional с биржи (кэшируется на 24 часа)"""
        now = time.time()
        if symbol in self.min_notional_cache and now - self.last_cache_update < 86400:
            return self.min_notional_cache[symbol]

        try:
            markets = await self.exchange.load_markets()
            market = markets.get(symbol)
            if not market:
                logger.warning(f"Символ {symbol} не найден → minNotional = 5.0 USDT")
                min_notional = 5.0
            else:
                min_notional = market["limits"]["cost"]["min"] or 5.0
                logger.debug(f"minNotional для {symbol}: {min_notional} USDT")

            self.min_notional_cache[symbol] = min_notional
            self.last_cache_update = now
            return min_notional
        except Exception as e:
            logger.error(f"Ошибка получения minNotional для {symbol}: {e}")
            return 5.0  # дефолт

    def open_position(self, symbol: str, pred: Dict, current_price: float):
        """
        Открытие виртуальной позиции по сигналу inference.
        """
        if symbol in self.positions:
            logger.debug(f"Виртуальная позиция {symbol} уже открыта — игнорируем")
            return

        direction = "L" if pred["prob"] > 0.5 else "S"
        size = self.risk_manager.calculate_size(self.virtual_balance, current_price, direction)

        tp_sl_levels = self.tp_sl_manager.calculate_levels(
            entry_price=current_price,
            direction=direction,
            avg_candle_pct=pred.get("avg_candle_pct", 0.8),
            position_size=size
        )

        position = {
            "symbol": symbol,
            "entry_time": time.time(),
            "entry_price": current_price,
            "direction": direction,
            "size": size,
            "tp_sl_levels": tp_sl_levels,
            "current_sl": tp_sl_levels.get("initial_sl", current_price * (0.99 if direction == "L" else 1.01)),
            "scenario_key": pred.get("scenario_key"),
            "unrealized_pnl": 0.0,
            "partial_closed": False,
            "remaining_size": size
        }

        self.positions[symbol] = position
        logger.info(f"[VIRTUAL OPEN] {direction} {symbol} @ {current_price:.2f} | size={size:.4f} | prob={pred['prob']:.4f}")

    def update_positions(self, symbol: str, current_candle: Dict):
        """
        Обновление виртуальных позиций на новой свече.
        Поддерживает partial TP с учётом minNotional биржи.
        """
        position = self.positions.get(symbol)
        if not position:
            return

        current_price = current_candle["close"]
        high = current_candle["high"]
        low = current_candle["low"]
        remaining_size = position["remaining_size"]
        tp_sl = position["tp_sl_levels"]
        direction = position["direction"]

        # Динамический minNotional (в USDT)
        min_notional = await self._get_min_notional(symbol)
        min_close_size = min_notional / current_price if current_price > 0 else float('inf')

        logger.debug(f"{symbol}: remaining={remaining_size:.6f}, min_close_size={min_close_size:.6f}")

        # Partial TP 1 (обычно 50%)
        if not position["partial_closed"] and tp_sl.get("tp1"):
            tp1_price = tp_sl["tp1"]["price"]
            tp1_portion = tp_sl["tp1"]["portion"]

            if (direction == "L" and high >= tp1_price) or (direction == "S" and low <= tp1_price):
                desired_partial = remaining_size * tp1_portion
                actual_partial = 0.0

                if desired_partial >= min_close_size:
                    actual_partial = desired_partial
                elif remaining_size >= min_close_size:
                    actual_partial = remaining_size
                    logger.info(f"{symbol}: позиция мала — закрываем весь остаток одним ордером (TP1)")
                else:
                    logger.debug(f"{symbol}: позиция слишком мала для TP1 ({remaining_size:.6f} < {min_close_size:.6f}) — пропускаем")

                if actual_partial > 0:
                    pnl = self._calculate_pnl(position, tp1_price, actual_partial)
                    self._apply_partial_close(symbol, pnl, "TP1", actual_partial)

                    position["partial_closed"] = True
                    position["remaining_size"] -= actual_partial

        # Partial TP 2 (обычно 30%)
        if position["partial_closed"] and tp_sl.get("tp2"):
            tp2_price = tp_sl["tp2"]["price"]
            tp2_portion = tp_sl["tp2"]["portion"]

            if (direction == "L" and high >= tp2_price) or (direction == "S" and low <= tp2_price):
                desired_partial = remaining_size * tp2_portion
                actual_partial = 0.0

                if desired_partial >= min_close_size:
                    actual_partial = desired_partial
                elif remaining_size >= min_close_size:
                    actual_partial = remaining_size
                    logger.info(f"{symbol}: остаток мал — закрываем весь остаток на TP2")
                else:
                    logger.debug(f"{symbol}: остаток слишком мал для TP2 ({remaining_size:.6f} < {min_close_size:.6f}) — пропускаем")

                if actual_partial > 0:
                    pnl = self._calculate_pnl(position, tp2_price, actual_partial)
                    self._apply_partial_close(symbol, pnl, "TP2", actual_partial)
                    position["remaining_size"] -= actual_partial

        # Trailing на остаток (если включён и была хотя бы одна фиксация)
        if position["remaining_size"] > 0 and position["partial_closed"]:
            new_sl = self.tp_sl_manager.update_trailing(current_price, position)
            if new_sl:
                position["current_sl"] = new_sl

            if (direction == "L" and low <= position["current_sl"]) or \
               (direction == "S" and high >= position["current_sl"]):
                pnl = self._calculate_pnl(position, position["current_sl"], position["remaining_size"])
                self._close_position(symbol, pnl, "SL")
                return

            # Обновляем нереализованный P&L на остаток
            position["unrealized_pnl"] = (current_price - position["entry_price"]) * position["remaining_size"] if direction == "L" else \
                                         (position["entry_price"] - current_price) * position["remaining_size"]

    def _apply_partial_close(self, symbol: str, pnl: float, reason: str, partial_size: float):
        """Фиксация части позиции"""
        self.virtual_balance += pnl
        self.closed_trades.append({
            "symbol": symbol,
            "entry_price": self.positions[symbol]["entry_price"],
            "exit_price": self.positions[symbol]["tp_sl_levels"].get("tp1", {}).get("price") or self.positions[symbol]["current_sl"],
            "pnl": pnl,
            "reason": reason,
            "direction": self.positions[symbol]["direction"],
            "partial_size": partial_size,
            "timestamp": time.time()
        })
        logger.info(f"[VIRTUAL PARTIAL] {symbol} | {reason} | pnl={pnl:.2f} | size={partial_size:.4f}")

    def _close_position(self, symbol: str, pnl: float, reason: str):
        """Полное закрытие позиции"""
        position = self.positions.pop(symbol, None)
        if not position:
            return

        net_pnl = pnl
        self.virtual_balance += net_pnl

        trade = {
            "symbol": symbol,
            "entry_price": position["entry_price"],
            "exit_price": position["tp_sl_levels"].get("tp_price") or position["current_sl"],
            "pnl": net_pnl,
            "reason": reason,
            "direction": position["direction"],
            "scenario_key": position["scenario_key"],
            "timestamp": time.time()
        }

        self.closed_trades.append(trade)

        if position["scenario_key"]:
            is_win = net_pnl > 0
            self.scenario_tracker.update_scenario(position["scenario_key"], is_win)

        logger.info(f"[VIRTUAL CLOSE] {symbol} | PNL={net_pnl:.2f} | reason={reason} | balance={self.virtual_balance:.2f}")

        self.storage.save_trade(trade)

    def _calculate_pnl(self, position: Dict, exit_price: float, size: float) -> float:
        """Расчёт P&L с комиссией и slippage"""
        if position["direction"] == "L":
            gross_pnl = (exit_price - position["entry_price"]) * size
        else:
            gross_pnl = (position["entry_price"] - exit_price) * size

        commission = abs(gross_pnl) * self.commission_rate * 2
        slippage = abs(position["entry_price"] - exit_price) * size * self.slippage_rate

        return gross_pnl - commission - slippage

    def get_stats(self) -> Dict[str, Any]:
        """Метрики виртуальной торговли"""
        trades_count = len(self.closed_trades)
        if trades_count == 0:
            return {"trades_count": 0, "winrate": 0.0, "profit_factor": 0.0, "max_dd": 0.0}

        wins = sum(1 for t in self.closed_trades if t["pnl"] > 0)
        winrate = wins / trades_count

        total_profit = sum(t["pnl"] for t in self.closed_trades if t["pnl"] > 0)
        total_loss = sum(abs(t["pnl"]) for t in self.closed_trades if t["pnl"] < 0)
        profit_factor = total_profit / total_loss if total_loss > 0 else float("inf")

        max_dd = self._calculate_max_drawdown()

        return {
            "trades_count": trades_count,
            "winrate": round(winrate, 4),
            "profit_factor": round(profit_factor, 4),
            "max_dd": round(max_dd, 2),
            "balance": round(self.virtual_balance, 2)
        }

    def _calculate_max_drawdown(self) -> float:
        """Максимальная просадка по equity curve"""
        equity = [self.config["finance"].get("deposit", 10000.0)]
        for trade in self.closed_trades:
            equity.append(equity[-1] + trade["pnl"])

        peak = equity[0]
        max_dd = 0.0
        for eq in equity:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak
            max_dd = max(max_dd, dd)

        return max_dd * 100

    def reset(self):
        """Сброс состояния (для тестов)"""
        self.virtual_balance = self.config["finance"].get("deposit", 10000.0)
        self.positions.clear()
        self.closed_trades.clear()
        logger.info("Виртуальный трейдер сброшен")