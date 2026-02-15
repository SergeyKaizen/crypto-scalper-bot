# src/backtest/engine.py
"""
Движок виртуального бэктеста для одной монеты.

Ключевые требования из ТЗ:
- Бэктест по истории для расчёта PR и фильтра монет
- Симуляция сигналов аномалий → виртуальные сделки → TP/SL
- Учёт всех TF и окон
- Обновление PR после каждой закрытой позиции
- Фильтр по min_pr, min_trades, min_age_months
- Параллельный запуск в backtest_all.py

Логика:
1. Загружаем историю свечей по всем TF
2. Симулируем каждую свечу: проверка аномалий → inference → открытие позиции
3. Управление позицией: TP/SL, trailing, partial
4. После закрытия — обновляем PR
5. В конце — возвращаем статистику
"""

import time
from typing import Dict, Any, List
import polars as pl
import numpy as np

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
        """Проверяет максимальное плечо для монеты"""
        max_lev = self.storage.get_max_leverage(self.symbol)
        requested = self.config["finance"]["leverage"]
        return min(requested, max_lev)

    def run_full_backtest(self) -> Dict[str, Any]:
        """
        Полный бэктест по всей доступной истории.
        Возвращает статистику для PR.
        """
        start_time = time.time()
        
        # Загружаем историю (все TF)
        candles = {}
        for tf in ["1m", "3m", "5m", "10m", "15m"]:
            df = self.storage.load_candles(self.symbol, tf, limit=50000)
            if df is not None and len(df) > 200:
                candles[tf] = df.sort("open_time")
                self.resampler.cache[tf].extend([df])  # заполняем кэш
        
        if not candles:
            return {"error": "нет данных"}

        # Симулируем по 1m свечам (основной таймфрейм)
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

        for i in range(100, len(df_1m)):  # начинаем после достаточного lookback
            current_candle = df_1m[i]
            current_time = current_candle["open_time"]
            
            # Обновляем ресэмплер новой свечой
            self.resampler.add_1m_candle(current_candle.to_dict())
            
            # Проверяем позицию (если открыта)
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
        results["profit_factor"] = abs(sum(t["pnl"] for t in self.closed_trades if t["pnl"] > 0) / 
                                      sum(abs(t["pnl"]) for t in self.closed_trades if t["pnl"] < 0) or 1)

        logger.info(f"Бэктест {self.symbol} завершён за {time.time() - start_time:.1f} сек | "
                    f"сделок: {results['trades_count']} | PR: {results['profit_factor']:.2f}")
        
        return results

    def _open_virtual_position(self, pred: Dict, entry_price: float):
        """Открывает виртуальную позицию по сигналу"""
        direction = "L" if pred["prob"] > 0.5 else "S"  # упрощённо, можно улучшить
        size = self.risk_manager.calculate_size(self.balance, entry_price, direction)
        
        tp_sl = self.tp_sl_manager.calculate_levels(
            entry_price=entry_price,
            direction=direction,
            avg_candle_pct=0.8,  # заглушка, в реальности из feature_engine
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
        logger.debug(f"Открыта виртуальная позиция {direction} на {self.symbol} @ {entry_price}")

    def _update_open_positions(self, current_candle: pl.DataFrame):
        """Обновляет все открытые позиции на новой свече"""
        to_close = []
        for pos in self.positions:
            current_price = current_candle["close"]
            high = current_candle["high"]
            low = current_candle["low"]
            
            # Проверяем TP/SL
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