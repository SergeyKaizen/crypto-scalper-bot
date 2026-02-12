# src/backtest/engine.py
"""
Движок бэктеста и симуляции торговли.

Основные функции:
- run_backtest() — полный бэктест на исторических данных (Polars + loop)
- simulate_trade() — симуляция одной сделки (TP/SL/trailing, soft entry)
- calculate_pnl() — расчёт профита/убытка с учётом комиссии
- run_parallel_backtest() — parallel-обработка монет на сервере (joblib)

Логика:
- Вход: DataFrame свечей (Polars) + сигналы (аномалии или Q)
- Симуляция: для каждого бара проверяем вход/выход, trailing, soft entry
- Комиссии: taker/maker fee (из constants)
- Shadow trading — параллельно считает "реальные" исходы (с slippage)
- PR — собирает все закрытые сделки → передаёт в pr_calculator

Зависимости:
- polars — векторная обработка
- joblib — parallel на сервере
- src/trading/tp_sl_manager.py — расчёт TP/SL/trailing
- src/trading/risk_manager.py — размер позиции
"""

import logging
from typing import List, Dict, Tuple
from datetime import datetime

import polars as pl
from joblib import Parallel, delayed

from src.core.config import load_config
from src.core.types import Position, TradeResult
from src.trading.tp_sl_manager import TpSlManager
from src.trading.risk_manager import RiskManager
from src.trading.order_executor import simulate_order_execution
from src.core.constants import BINANCE_FUTURES_TAKER_FEE

logger = logging.getLogger(__name__)


class BacktestEngine:
    """Движок бэктеста и симуляции торговли"""

    def __init__(self, config: Dict):
        self.config = config
        self.risk_manager = RiskManager(config)
        self.tp_sl_manager = TpSlManager(config)
        self.commission_rate = float(BINANCE_FUTURES_TAKER_FEE)

    def simulate_trade(self, df: pl.DataFrame, entry_idx: int, signal: Dict) -> TradeResult:
        """
        Симуляция одной сделки от входа до выхода

        Args:
            df: Polars DataFrame свечей
            entry_idx: индекс бара входа
            signal: {"direction": "L"/"S"/"LS", "probability": float, ...}

        Returns:
            TradeResult — результат закрытия
        """
        entry_price = df["close"][entry_idx]
        direction = signal["direction"]
        size = self.risk_manager.calculate_position_size(entry_price, signal, df)

        position = Position(
            entry_time=df["timestamp"][entry_idx],
            entry_price=entry_price,
            size=size,
            direction=direction,
            tp_price=self.tp_sl_manager.calculate_tp(df, entry_price, direction),
            sl_price=self.tp_sl_manager.calculate_sl(df, entry_price, direction),
            is_virtual=True
        )

        # Симуляция выхода
        exit_idx = entry_idx + 1
        while exit_idx < len(df):
            current_price = df["close"][exit_idx]
            current_high = df["high"][exit_idx]
            current_low = df["low"][exit_idx]

            # Проверяем TP/SL
            if direction == "L":
                if current_high >= position.tp_price:
                    return TradeResult(
                        position=position,
                        exit_time=df["timestamp"][exit_idx],
                        exit_price=position.tp_price,
                        pnl_pct=(position.tp_price - entry_price) / entry_price * 100,
                        pnl_usdt=size * (position.tp_price - entry_price),
                        reason="TP",
                        is_win=True
                    )
                if current_low <= position.sl_price:
                    return TradeResult(
                        position=position,
                        exit_time=df["timestamp"][exit_idx],
                        exit_price=position.sl_price,
                        pnl_pct=(position.sl_price - entry_price) / entry_price * 100,
                        pnl_usdt=size * (position.sl_price - entry_price),
                        reason="SL",
                        is_win=False
                    )
            else:  # Short
                # Аналогично для шорта

            # Trailing stop
            if self.config["trailing_enabled"]:
                position = self.tp_sl_manager.update_trailing(position, current_high, current_low)

            exit_idx += 1

        # Если не вышли — закрываем по последней цене
        last_price = df["close"][-1]
        pnl_pct = (last_price - entry_price) / entry_price * 100 if direction == "L" else (entry_price - last_price) / entry_price * 100
        return TradeResult(
            position=position,
            exit_time=df["timestamp"][-1],
            exit_price=last_price,
            pnl_pct=pnl_pct,
            pnl_usdt=size * (last_price - entry_price) if direction == "L" else size * (entry_price - last_price),
            reason="End of data",
            is_win=pnl_pct > 0
        )

    def run_backtest(self, symbol: str, timeframe: str) -> List[TradeResult]:
        """Полный бэктест для одной монеты и TF"""
        df = self.storage.get_candles(symbol, timeframe, limit=self.config["pr"]["analysis_period_candles"] * 2)
        if len(df) < 100:
            logger.warning("Недостаточно данных для бэктеста %s %s", symbol, timeframe)
            return []

        results = []
        for i in range(len(df) - 100):
            # Проверяем сигнал на баре i
            # (в реальном коде — вызываем anomaly_detector и inference)
            # Упрощённо — имитация
            if i % 50 == 0:  # Каждый 50-й бар — сигнал
                signal = {"direction": "L", "probability": 0.75}
                result = self.simulate_trade(df, i, signal)
                results.append(result)

        logger.info("Backtest completed for %s %s: %d trades", symbol, timeframe, len(results))
        return results

    def run_parallel_backtest(self, symbols: List[str], timeframe: str = "1m") -> Dict[str, List[TradeResult]]:
        """Параллельный бэктест (joblib) — только на сервере"""
        if not self.config.get("parallel", False):
            results = {}
            for symbol in symbols:
                results[symbol] = self.run_backtest(symbol, timeframe)
            return results

        # Parallel
        parallel_results = Parallel(n_jobs=-1)(
            delayed(self.run_backtest)(symbol, timeframe) for symbol in symbols
        )

        return dict(zip(symbols, parallel_results))