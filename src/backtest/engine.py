# src/backtest/engine.py
"""
Движок виртуального бэктеста для одной монеты.

Ключевые требования из ТЗ:
- Полный бэктест по истории для расчёта PR и фильтра монет
- Симуляция сигналов аномалий → inference → виртуальные сделки → TP/SL
- Учёт всех TF и окон (24/50/74/100)
- Обновление PR после каждой закрытой позиции
- Фильтр по min_pr, min_trades, min_age_months (настраивается в конфиге)
- Параллельный запуск в backtest_all.py (multiprocessing)

Логика:
1. Загружаем историю свечей со всех TF (storage)
2. Симулируем каждую 1m свечу: аномалии → inference → открытие позиции
3. Управление позицией: TP/SL/trailing/partial (через tp_sl_manager)
4. После закрытия — обновляем PR-снимок в storage
5. В конце — возвращаем статистику для whitelist
"""

import time
from typing import Dict, Any, List
import polars as pl
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from src.model.inference import InferenceEngine
from src.trading.tp_sl_manager import TPSLManager
from src.trading.risk_manager import RiskManager
from src.trading.virtual_trader import VirtualTrader
from src.features.feature_engine import FeatureEngine
from src.features.anomaly_detector import AnomalyDetector
from src.data.resampler import Resampler
from src.data.storage import Storage
from src.core.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

class BacktestEngine:
    def __init__(self, config: dict, symbol: str):
        self.config = config
        self.symbol = symbol
        self.inference = InferenceEngine(config)
        self.tp_sl_manager = TPSLManager(config)
        self.risk_manager = RiskManager(config)
        self.virtual_trader = VirtualTrader(config)
        self.feature_engine = FeatureEngine(config)
        self.anomaly_detector = AnomalyDetector(config)
        self.resampler = Resampler(config)
        self.storage = Storage(config)

        self.initial_balance = config["finance"]["deposit"]
        self.leverage = self._get_allowed_leverage()
        self.balance = self.initial_balance
        self.positions = []  # список открытых виртуальных позиций
        self.closed_trades = []  # список закрытых сделок

    def _get_allowed_leverage(self):
        """Проверяет максимальное плечо для монеты (из storage или биржи)"""
        max_lev = self.storage.get_max_leverage(self.symbol) or 125
        requested = self.config["finance"].get("leverage", 20)
        return min(requested, max_lev)

    def run_full_backtest(self) -> Dict[str, Any]:
        """
        Полный бэктест по всей доступной истории.
        Возвращает статистику для PR и фильтра монет.
        """
        start_time = time.time()

        # Загружаем историю (все TF, последние 50 000 свечей — ≈35 дней на 1m)
        candles = {}
        for tf in ["1m", "3m", "5m", "10m", "15m"]:
            df = self.storage.load_candles(self.symbol, tf, limit=50000)
            if df is None or len(df) < 200:
                continue
            candles[tf] = df.sort("open_time")
            self.resampler.cache[tf].extend([df])  # заполняем кэш

        if "1m" not in candles or len(candles["1m"]) < 1000:
            return {"error": f"недостаточно данных для {self.symbol}"}

        df_1m = candles["1m"]
        results = {
            "trades_count": 0,
            "wins": 0,
            "losses": 0,
            "profit": 0.0,
            "max_drawdown": 0.0,
            "best_tf": None,
            "best_window": None,
            "best_anomaly_type": None
        }

        equity_curve = [self.initial_balance]
        peak = self.initial_balance

        # Симуляция по каждой 1m свече
        for i in range(100, len(df_1m)):  # начинаем после достаточного lookback
            current_candle = df_1m[i]
            current_time = current_candle["open_time"]

            # Обновляем ресэмплер новой свечой
            self.resampler.add_1m_candle(current_candle.to_dict())

            # Обновляем открытые позиции
            self._update_open_positions(current_candle)

            # Проверяем сигнал на открытие
            pred = self.inference.predict(self.symbol)
            if pred["signal"]:
                self._open_virtual_position(pred, current_candle["close"])

            # Обновляем equity curve
            current_equity = self.balance + self._get_unrealized_pnl(current_candle["close"])
            equity_curve.append(current_equity)
            peak = max(peak, current_equity)
            dd = (peak - current_equity) / peak * 100
            results["max_drawdown"] = max(results["max_drawdown"], dd)

        # Финальные метрики
        results["trades_count"] = len(self.closed_trades)
        results["wins"] = sum(1 for t in self.closed_trades if t["pnl"] > 0)
        results["losses"] = results["trades_count"] - results["wins"]
        results["winrate"] = results["wins"] / results["trades_count"] if results["trades_count"] > 0 else 0
        results["profit"] = sum(t["pnl"] for t in self.closed_trades)
        total_profit = sum(t["pnl"] for t in self.closed_trades if t["pnl"] > 0)
        total_loss = sum(abs(t["pnl"]) for t in self.closed_trades if t["pnl"] < 0)
        results["profit_factor"] = total_profit / total_loss if total_loss > 0 else float("inf")

        logger.info(f"Бэктест {self.symbol} завершён за {time.time() - start_time:.1f} сек | "
                    f"сделок: {results['trades_count']} | PR: {results['profit_factor']:.2f} | "
                    f"winrate: {results['winrate']:.2%} | max_dd: {results['max_drawdown']:.2f}%")

        # Сохраняем PR-снимок
        self.storage.save_pr_snapshot(self.symbol, results)

        return results

    def _open_virtual_position(self, pred: Dict, entry_price: float):
        """Открывает виртуальную позицию по сигналу"""
        direction = "L" if pred["prob"] > 0.5 else "S"
        size = self.risk_manager.calculate_size(self.balance, entry_price, direction)

        tp_sl = self.tp_sl_manager.calculate_levels(
            entry_price=entry_price,
            direction=direction,
            avg_candle_pct=pred.get("avg_candle_pct", 0.8),
            position_size=size
        )

        position = {
            "symbol": self.symbol,
            "entry_time": time.time(),
            "entry_price": entry_price,
            "direction": direction,
            "size": size,
            "tp_sl_levels": tp_sl,
            "current_sl": tp_sl.get("initial_sl", entry_price * (0.99 if direction == "L" else 1.01)),
            "unrealized_pnl": 0.0
        }

        self.positions.append(position)
        logger.debug(f"[VIRTUAL] Открыта позиция {direction} на {self.symbol} @ {entry_price}")

    def _update_open_positions(self, current_candle: pl.DataFrame):
        """Обновляет все открытые позиции на новой свече"""
        to_close = []
        for pos in self.positions:
            current_price = current_candle["close"]
            high = current_candle["high"]
            low = current_candle["low"]

            if pos["direction"] == "L":
                if high >= pos["tp_sl_levels"].get("tp_price", float("inf")):
                    pnl = (pos["tp_sl_levels"]["tp_price"] - pos["entry_price"]) * pos["size"]
                    self._close_position(pos, pnl, "TP")
                    to_close.append(pos)
                elif low <= pos["current_sl"]:
                    pnl = (pos["current_sl"] - pos["entry_price"]) * pos["size"]
                    self._close_position(pos, pnl, "SL")
                    to_close.append(pos)
            else:  # Short
                if low <= pos["tp_sl_levels"].get("tp_price", -float("inf")):
                    pnl = (pos["entry_price"] - pos["tp_sl_levels"]["tp_price"]) * pos["size"]
                    self._close_position(pos, pnl, "TP")
                    to_close.append(pos)
                elif high >= pos["current_sl"]:
                    pnl = (pos["entry_price"] - pos["current_sl"]) * pos["size"]
                    self._close_position(pos, pnl, "SL")
                    to_close.append(pos)

            # Trailing update
            new_sl = self.tp_sl_manager.update_trailing(current_price, pos)
            if new_sl:
                pos["current_sl"] = new_sl

        for pos in to_close:
            self.positions.remove(pos)

    def _close_position(self, pos: Dict, pnl: float, reason: str):
        self.balance += pnl
        self.closed_trades.append({
            "symbol": pos["symbol"],
            "entry_price": pos["entry_price"],
            "exit_price": pos["tp_sl_levels"].get("tp_price") or pos["current_sl"],
            "pnl": pnl,
            "reason": reason,
            "direction": pos["direction"]
        })
        logger.debug(f"Закрыта позиция {pos['direction']} | PNL: {pnl:.2f} | reason: {reason}")

    def _get_unrealized_pnl(self, current_price: float) -> float:
        pnl = 0.0
        for pos in self.positions:
            if pos["direction"] == "L":
                pnl += (current_price - pos["entry_price"]) * pos["size"]
            else:
                pnl += (pos["entry_price"] - current_price) * pos["size"]
        return pnl


if __name__ == "__main__":
    config = load_config()
    engine = BacktestEngine(config, "BTCUSDT")
    results = engine.run_full_backtest()
    print(results)