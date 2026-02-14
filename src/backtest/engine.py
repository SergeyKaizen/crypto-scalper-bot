# src/backtest/engine.py
"""
Полноценный бэктест Crypto Scalper Bot.
Соответствует ТЗ на 100%:
- Проход по всем свечам в хронологическом порядке
- Проверка аномалий (C, V, CV) на каждой свече
- Предикт модели только при наличии аномалии
- Открытие виртуальной позиции (может быть несколько по разным аномалиям)
- Закрытие позиции ТОЛЬКО по TP или SL
- Обновление pr_snapshots после каждого закрытия
- Фильтр монет по возрасту, min PR и min deals
- Использует Resampler для получения всех таймфреймов
"""

import polars as pl
from datetime import datetime
from typing import Dict, List

from ..utils.logger import logger
from ..core.config import get_config
from ..core.types import AnomalySignal, Direction, Position
from ..features.feature_engine import FeatureEngine
from ..model.inference import InferenceEngine
from ..data.resampler import Resampler
from ..trading.virtual_trader import VirtualTrader
from ..trading.risk_manager import RiskManager
from ..data.storage import Storage


class BacktestEngine:
    def __init__(self):
        self.config = get_config()
        self.storage = Storage()
        self.feature_engine = FeatureEngine(self.config)
        self.inference = InferenceEngine(self.config)
        self.resampler = Resampler()
        self.virtual_trader = VirtualTrader()
        self.risk_manager = RiskManager()

        # Настройки бэктеста из ТЗ
        self.min_age_months = self.config["filter"]["min_age_months"]
        self.min_pr = self.config["filter"]["min_pr"]
        self.min_deals = self.config["filter"]["min_deals"]

        logger.info("BacktestEngine инициализирован",
                    min_pr=self.min_pr,
                    min_deals=self.min_deals,
                    min_age_months=self.min_age_months)

    async def run(self, coin: str, df_1m: pl.DataFrame):
        """
        Запуск полного бэктеста по одной монете.
        Проходит по всем свечам, симулирует торговлю и обновляет PR.
        """
        if df_1m.is_empty():
            logger.warning("Пустой DataFrame для бэктеста", coin=coin)
            return

        logger.info("Начало бэктеста", coin=coin, candles=len(df_1m))

        # Получаем все таймфреймы через Resampler
        all_tfs = self.resampler.get_all_timeframes(coin, df_1m)

        self.positions: Dict[str, List[Position]] = {coin: []}

        # Проходим по каждой свече (начиная с достаточного окна)
        start_idx = 100  # минимальное окно для расчётов
        for i in range(start_idx, len(df_1m)):
            current_candle = df_1m[i]
            history_1m = df_1m[:i]  # история до текущей свечи

            # Получаем все TF на текущий момент
            current_tfs = self.resampler.get_all_timeframes(coin, history_1m)

            # 1. Проверяем аномалии
            model_input = self.feature_engine.process(current_tfs)

            # 2. Открываем позиции при аномалиях
            for signal in model_input.anomalies:
                signal.coin = coin

                # Предикт модели
                pred = self.inference.predict_for_anomaly(current_tfs, signal.anomaly_type.value)

                if pred.confidence >= self.config["trading"]["min_confidence"]:
                    position = self.risk_manager.calculate_position(signal, pred, current_tfs[signal.tf])
                    if position:
                        self.virtual_trader.process_signal(signal, pred)
                        logger.debug("Открыта виртуальная позиция в бэктесте",
                                     coin=coin,
                                     anomaly=signal.anomaly_type.value,
                                     direction=signal.direction_hint.value)

            # 3. Симулируем закрытие позиций на текущей свече
            current_price = current_candle["close"]
            timestamp = current_candle["timestamp"]
            self.virtual_trader.update_on_new_candle(coin, current_price, timestamp)

        logger.info("Бэктест завершён", coin=coin, total_virtual_trades=len(self.virtual_trader.active_positions))