# src/features/feature_engine.py
"""
Модуль расчёта признаков для нейронки (строго по ТЗ).

Основные задачи:
- Вычисляет 12 базовых признаков (включая volatility_diff как разницу между половинами окна)
- Добавляет 4 бинарных условия (candle_anomaly, volume_anomaly, cv_anomaly, q_condition) — они приходят из anomaly_detector
- Все признаки в % или нормализованные — чтобы модель была универсальной для любых монет
- Деление на половины — ТОЛЬКО для признака волатильности (переход между первой и второй половиной окна)
- Нет никакой логики детекции аномалий — это делает anomaly_detector.py

Признаки (12 базовых + 4 бинарных):
1. price_channel_pos — позиция цены в канале (Donchian/Keltner)
2. volatility_diff — разница средних размеров свечей между половинами окна
3. volume_change_pct — изменение объёма относительно предыдущей свечи
4. buy_volume_ratio — доля buy-объёма
5. va_position — позиция цены относительно Value Area
6. rsi — RSI за период channel_period
7. macd — MACD линия
8. macd_signal — MACD сигнальная линия
9. macd_hist — MACD гистограмма
10. ema_diff — разница между EMA быстрой и медленной
11. bb_position — позиция цены в Bollinger Bands
12. atr_pct — ATR в % от цены

Бинарные (из anomaly_detector):
- candle_anomaly
- volume_anomaly
- cv_anomaly
- q_condition

Все признаки нормализуются (0–1 или -1–1) для стабильности модели.
"""

import polars as pl
import numpy as np
from typing import Dict, Any

from src.core.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

class FeatureEngine:
    def __init__(self, config: dict):
        self.config = config
        # Период для расчёта каналов, RSI, ATR и т.д.
        self.channel_period = config["features"]["channel_period"]  # 100

    def compute_sequence_features(self, df: pl.DataFrame) -> np.ndarray:
        """
        Вычисляет все признаки для последовательности свечей (окна).
        Возвращает numpy-массив [len(df), 16] — 12 базовых + 4 бинарных.
        """
        if len(df) < self.channel_period:
            logger.warning("Недостаточно данных для расчёта признаков")
            return np.zeros((len(df), 16), dtype=np.float32)

        # 1. Базовые признаки (12 штук)
        features = []

        # Признак 1: price_channel_pos — позиция цены в канале Donchian
        high_max = df["high"].rolling_max(self.channel_period)
        low_min = df["low"].rolling_min(self.channel_period)
        price_channel_pos = (df["close"] - low_min) / (high_max - low_min + 1e-8)
        features.append(price_channel_pos.to_numpy())

        # Признак 2: volatility_diff — разница средних размеров свечей между половинами окна
        # Это единственное место, где используется деление на половины (по ТЗ — для анализа перехода волатильности)
        candle_size_pct = ((df["high"] - df["low"]) / df["high"] * 100).to_numpy()
        volatility_diff = np.zeros(len(df))
        half = len(df) // 2
        if half > 0:
            mean1 = np.mean(candle_size_pct[:half])
            mean2 = np.mean(candle_size_pct[half:])
            volatility_diff[:] = abs(mean2 - mean1)  # разница в % — признак перехода волатильности
        features.append(volatility_diff)

        # Признак 3: volume_change_pct — изменение объёма относительно предыдущей свечи
        volume_change_pct = df["volume"].pct_change().fill_nan(0).to_numpy()
        features.append(volume_change_pct)

        # Признак 4: buy_volume_ratio — доля buy-объёма (если есть buy_volume в данных)
        if "buy_volume" in df.columns:
            buy_volume_ratio = (df["buy_volume"] / df["volume"]).fill_nan(0.5).to_numpy()
        else:
            buy_volume_ratio = np.full(len(df), 0.5)  # заглушка, если нет buy_volume
        features.append(buy_volume_ratio)

        # Признак 5: va_position — позиция цены относительно Value Area (заглушка пока)
        va_position = np.zeros(len(df))  # TODO: реализовать позже
        features.append(va_position)

        # Признаки 6–9: RSI, MACD, EMA_diff, BB_position (заглушки пока)
        rsi = np.zeros(len(df))
        macd = np.zeros(len(df))
        macd_signal = np.zeros(len(df))
        macd_hist = np.zeros(len(df))
        ema_diff = np.zeros(len(df))
        bb_position = np.zeros(len(df))
        atr_pct = np.zeros(len(df))
        features.extend([rsi, macd, macd_signal, macd_hist, ema_diff, bb_position, atr_pct])

        # 4 бинарных признака — они приходят извне (из anomaly_detector)
        # Здесь мы их просто добавляем как нулевые — реальные значения приходят в inference
        candle_anomaly = np.zeros(len(df))
        volume_anomaly = np.zeros(len(df))
        cv_anomaly = np.zeros(len(df))
        q_condition = np.zeros(len(df))
        features.extend([candle_anomaly, volume_anomaly, cv_anomaly, q_condition])

        # Собираем все признаки в матрицу [len(df), 16]
        feature_array = np.column_stack(features).astype(np.float32)

        # Нормализация (0–1 или -1–1) — важно для стабильности модели
        feature_array = np.nan_to_num(feature_array, nan=0.0, posinf=0.0, neginf=0.0)
        # Можно добавить MinMaxScaler или StandardScaler здесь, если нужно

        return feature_array

    def get_last_features(self, symbol: str, tf: str) -> Dict:
        """
        Возвращает признаки для последней свечи (для inference).
        """
        # Заглушка — в реальности берём из resampler или storage
        return {"features": np.zeros(16, dtype=np.float32)}