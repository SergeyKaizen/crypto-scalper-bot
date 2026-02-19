# src/backtest/engine.py
"""
Движок виртуального бэктеста для одной монеты.

Реализовано строго по ТЗ и твоим последним уточнениям:
- Бэктест работает только по последним 250 свечам на каждом из 5 таймфреймов (1m, 3m, 5m, 10m, 15m)
- Симуляция идёт по времени 1m свечам (основной ряд)
- Для каждой аномалии inference получает последние 250 свечей каждого TF
- PR считается по формулам ТЗ:
    PR_L  = (Кол-во лонг по TP × длина TP) - (Кол-во лонг по SL × длина SL)
    PR_S  = (Кол-во шорт по TP × длина TP) - (Кол-во шорт по SL × длина SL)
    PR_LS = ((лонг TP + шорт TP) × длина TP) - ((лонг SL + шорт SL) × длина SL)
- Длина TP/SL — расстояние от входа до уровня в % (абсолютное значение)
- После бэктеста сохраняет PR_LS в storage для формирования whitelist
- Никаких дополнительных метрик — только то, что в ТЗ
"""

import time
from typing import Dict, Any
import polars as pl

from src.model.inference import InferenceEngine
from src.trading.tp_sl_manager import TPSLManager
from src.trading.risk_manager import RiskManager
from src.trading.virtual_trader import VirtualTrader
from src.features.anomaly_detector import AnomalyDetector
from src.data.resampler import Resampler
from src.data.storage import Storage
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
        self.anomaly_detector = AnomalyDetector(config)
        self.resampler = Resampler(config)
        self.storage = Storage(config)

        # Последние 250 свечей на каждом TF — фиксировано по ТЗ
        self.LAST_N_CANDLES = 250
        self.last_n = {}  # tf → DataFrame с последними 250 свечами

    def run_full_backtest(self) -> Dict[str, Any]:
        """
        Полный бэктест по последним 250 свечам на каждом TF.
        Возвращает PR_L, PR_S, PR_LS строго по формулам ТЗ.
        """
        logger.info(f"Запуск бэктеста для {self.symbol} | Последние {self.LAST_N_CANDLES} свечей на каждом TF")

        # 1. Загружаем последние 250 свечей для каждого TF
        for tf in ["1m", "3m", "5m", "10m", "15m"]:
            df = self.storage.load_candles(self.symbol, tf, limit=self.LAST_N_CANDLES)
            if df is None or len(df) < 100:
                logger.warning(f"Недостаточно данных для {tf} ({len(df) if df is not None else 0} свечей)")
                continue
            self.last_n[tf] = df.sort("open_time")

        if "1m" not in self.last_n or len(self.last_n["1m"]) < 100:
            return {"error": f"Недостаточно данных для {self.symbol}"}

        df_1m = self.last_n["1m"]

        # Статистика для расчёта PR по формулам ТЗ
        long_tp_count = 0
        long_sl_count = 0
        short_tp_count = 0
        short_sl_count = 0

        long_tp_total_length = 0.0
        long_sl_total_length = 0.0
        short_tp_total_length = 0.0
        short_sl_total_length = 0.0

        # Симуляция по 1m свечам
        for i in range(100, len(df_1m)):  # начинаем после достаточного lookback
            current_candle = df_1m[i]

            # Обновляем ресэмплер новой свечой (чтобы модель видела актуальные 250 свечей на всех TF)
            self.resampler.add_1m_candle(current_candle.to_dict())

            # Проверяем аномалии и делаем предсказание
            pred = self.inference.predict(self.symbol)
            if not pred["signal"]:
                continue

            # Открываем виртуальную позицию
            self.virtual_trader.open_position(self.symbol, pred, current_candle["close"])

            # Обновляем все позиции на текущей свече
            self.virtual_trader.update_positions(self.symbol, current_candle.to_dict())

        # Подсчёт по формулам ТЗ
        for trade in self.virtual_trader.closed_trades:
            length = abs(trade["exit_price"] - trade["entry_price"]) / trade["entry_price"] * 100  # в %

            if trade["direction"] == "L":
                if trade["reason"] == "TP":
                    long_tp_count += 1
                    long_tp_total_length += length
                elif trade["reason"] == "SL":
                    long_sl_count += 1
                    long_sl_total_length += length
            else:  # "S"
                if trade["reason"] == "TP":
                    short_tp_count += 1
                    short_tp_total_length += length
                elif trade["reason"] == "SL":
                    short_sl_count += 1
                    short_sl_total_length += length

        # Расчёт PR по формулам ТЗ
        pr_l = (long_tp_count * (long_tp_total_length / long_tp_count if long_tp_count > 0 else 0)) - \
               (long_sl_count * (long_sl_total_length / long_sl_count if long_sl_count > 0 else 0))

        pr_s = (short_tp_count * (short_tp_total_length / short_tp_count if short_tp_count > 0 else 0)) - \
               (short_sl_count * (short_sl_total_length / short_sl_count if short_sl_count > 0 else 0))

        pr_ls = ((long_tp_count + short_tp_count) * 
                 ((long_tp_total_length + short_tp_total_length) / (long_tp_count + short_tp_count) 
                  if (long_tp_count + short_tp_count) > 0 else 0)) - \
                ((long_sl_count + short_sl_count) * 
                 ((long_sl_total_length + short_sl_total_length) / (long_sl_count + short_sl_count) 
                  if (long_sl_count + short_sl_count) > 0 else 0))

        result = {
            "symbol": self.symbol,
            "pr_l": round(pr_l, 4),
            "pr_s": round(pr_s, 4),
            "pr_ls": round(pr_ls, 4),          # основной показатель для whitelist
            "long_tp_count": long_tp_count,
            "long_sl_count": long_sl_count,
            "short_tp_count": short_tp_count,
            "short_sl_count": short_sl_count,
            "total_trades": len(self.virtual_trader.closed_trades)
        }

        # Сохраняем PR_LS в storage
        self.storage.save_pr_snapshot(self.symbol, result)

        logger.info(f"Бэктест {self.symbol} завершён | PR_LS = {pr_ls:.4f} | Сделок: {result['total_trades']}")

        return result


if __name__ == "__main__":
    config = load_config()
    engine = BacktestEngine(config, "BTCUSDT")
    result = engine.run_full_backtest()
    print(result)