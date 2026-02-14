# src/backtest/engine.py
"""
Полноценный бэктест.
Проходит по историческим 1m свечам, использует Resampler для получения всех TF,
открывает виртуальные позиции и закрывает их только по TP/SL.
"""

import polars as pl

from ..utils.logger import logger
from ..features.feature_engine import FeatureEngine
from ..model.inference import InferenceEngine
from ..data.resampler import Resampler
from ..trading.virtual_trader import VirtualTrader
from ..trading.risk_manager import RiskManager


class BacktestEngine:
    def __init__(self):
        self.feature_engine = FeatureEngine(None)   # config не нужен в бэктесте
        self.inference = InferenceEngine(None)
        self.resampler = Resampler()
        self.virtual_trader = VirtualTrader()
        self.risk_manager = RiskManager()

        logger.info("BacktestEngine инициализирован")

    async def run(self, coin: str, df_1m: pl.DataFrame):
        """Запуск бэктеста по одной монете."""
        if df_1m.is_empty():
            logger.warning("Пустой DataFrame", coin=coin)
            return

        logger.info("Запуск бэктеста", coin=coin, candles=len(df_1m))

        # Получаем все таймфреймы через Resampler
        all_tfs = self.resampler.get_all_timeframes(coin, df_1m)

        for i in range(50, len(df_1m)):   # начинаем после минимального окна
            # Берём историю до текущей свечи
            history_1m = df_1m[:i]
            current_price = df_1m["close"][i]
            timestamp = df_1m["timestamp"][i]

            # Получаем все TF до текущего момента
            current_tfs = self.resampler.get_all_timeframes(coin, history_1m)

            # Проверяем аномалии
            model_input = self.feature_engine.process(current_tfs)

            for signal in model_input.anomalies:
                signal.coin = coin
                pred = self.inference.predict_for_anomaly(current_tfs, signal.anomaly_type)

                if pred.confidence > 0.6:   # порог можно вынести в конфиг
                    position = self.risk_manager.calculate_position(signal, pred, current_tfs[signal.tf])
                    if position:
                        self.virtual_trader.process_signal(signal, pred)   # открываем

            # Симулируем закрытие позиций на текущей свече
            self.virtual_trader.update_on_new_candle(coin, current_price, timestamp)

        logger.info("Бэктест завершён", coin=coin)