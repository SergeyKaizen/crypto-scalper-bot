# src/features/feature_engine.py
"""
Центральный модуль подготовки признаков для модели.

Функционал:
- Сбор 12 базовых признаков по половинам 100 свечей (по ТЗ)
- Добавление 4 бинарных условий (C, V, CV, Q)
- Учёт 4 окон (24, 50, 74, 100) — momentum, acceleration, последовательные свечи
- Подготовка sequence (сырые свечи) + aggregated features (для GRU/Conv1D)
- Ограничение по конфигу (max_tf, max_windows) — для телефона

Используется в:
- live_loop.py — перед inference
- trainer.py — для создания датасета обучения

Логика:
- Основной период — 100 свечей (по ТЗ)
- Половины — левая 50 / правая 50
- 12 признаков — delta %, volume, bid/ask, volatility, price change и т.д.
- 4 условия — бинарные (0/1) — C, V, CV, Q (Q заполняется в inference)
- Окна — momentum = (close[-1] - close[-10]) / close[-10], acceleration = momentum[-5] - momentum[-10]
- Sequence — последние seq_len свечей (50–100) × num_tf

Все расчёты асинхронно и на Polars
"""

import logging
from typing import Dict, List, Tuple

import polars as pl

from src.core.config import load_config
from src.features.anomalies import AnomalyDetector
from src.features.channels import ChannelsCalculator
from src.features.volume_features import VolumeFeaturesCalculator

logger = logging.getLogger(__name__)

class FeatureEngine:
    """Подготовка всех признаков для модели"""

    def __init__(self, config: Dict):
        self.config = config
        self.anomaly_detector = AnomalyDetector(config)
        self.channel_calc = ChannelsCalculator(config)
        self.volume_calc = VolumeFeaturesCalculator(config)
        self.max_tf = config["max_tf"]
        self.max_windows = config["max_windows"]
        self.seq_len = config["seq_len"]

    async def build_features(self, data: Dict[str, pl.DataFrame]) -> Dict:
        """
        Главная функция: собирает признаки для всех TF

        Args:
            data: Dict[tf: DataFrame] — данные по каждому TF (1m, 3m, 5m, 10m, 15m)

        Returns:
            Dict[tf: Dict] — признаки + sequence для каждого TF
        """
        features = {}
        sequences = {}

        for tf in self.config["timeframes"][:self.max_tf]:
            df = data.get(tf)
            if df is None or len(df) < self.seq_len:
                logger.warning("Недостаточно данных для TF %s", tf)
                continue

            # 1. Последовательность свечей (sequence)
            sequence = df.tail(self.seq_len)[["open", "high", "low", "close", "volume", "bid_volume", "ask_volume"]].to_numpy()
            sequences[tf] = sequence  # shape: (seq_len, 7)

            # 2. Аггрегированные признаки по половинам
            half = len(df) // 2
            left = df.slice(0, half)
            right = df.slice(half, half)

            agg_features = {}

            # Volume & Delta
            vol_left = left["volume"].sum()
            vol_right = right["volume"].sum()
            agg_features["volume_delta_pct"] = (vol_right - vol_left) / vol_left * 100 if vol_left > 0 else 0.0

            # Bid/Ask delta
            bid_right = right["bid_volume"].sum()
            ask_right = right["ask_volume"].sum()
            total_right = vol_right
            agg_features["bid_ask_delta"] = (bid_right - ask_right) / total_right * 100 if total_right > 0 else 0.0

            # Price change
            agg_features["price_change_pct"] = (right["close"][-1] - left["close"][-1]) / left["close"][-1] * 100

            # Volatility
            agg_features["volatility_right"] = right["close"].std()

            # ... остальные 8–12 признаков аналогично (по ТЗ)

            # 3. Аномалии (C/V/CV)
            anomaly_flags = self.anomaly_detector.detect(df.tail(100))
            agg_features.update(anomaly_flags)

            # 4. Каналы и VA (добавляем как признаки)
            channel = self.channel_calc.calculate_price_channel(df.tail(100))
            va = self.channel_calc.calculate_value_area(df.tail(100))
            agg_features.update(channel)
            agg_features.update(va)

            # 5. Окна (momentum, acceleration) — только max_windows
            for window in self.config["windows"][:self.max_windows]:
                win_df = df.tail(window)
                if len(win_df) < 10:
                    continue
                momentum = (win_df["close"][-1] - win_df["close"][-10]) / win_df["close"][-10] * 100
                acceleration = (momentum - (win_df["close"][-6] - win_df["close"][-10]) / win_df["close"][-10] * 100)
                agg_features[f"momentum_w{window}"] = momentum
                agg_features[f"acceleration_w{window}"] = acceleration

            features[tf] = agg_features

        return {
            "sequences": sequences,   # Dict[tf: np.array(seq_len, 7)]
            "features": features      # Dict[tf: Dict[str, float]]
        }

# Пример использования (для тестов)
if __name__ == "__main__":
    config = load_config()
    engine = FeatureEngine(config)

    # Пример данных
    dummy_df = pl.DataFrame({
        "timestamp": [i for i in range(100)],
        "open": [60000 + i*10 for i in range(100)],
        "high": [60050 + i*10 for i in range(100)],
        "low": [59950 + i*10 for i in range(100)],
        "close": [60020 + i*10 for i in range(100)],
        "volume": [100 + i*5 for i in range(100)],
        "bid_volume": [60 + i*3 for i in range(100)],
        "ask_volume": [40 + i*2 for i in range(100)],
    })

    data = {"1m": dummy_df, "5m": dummy_df}  # Имитация

    result = engine.build_features(data)
    print("Признаки для 1m:", result["features"].get("1m", {}))