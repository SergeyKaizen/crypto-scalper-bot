# src/features/anomaly_detector.py
"""
Детектор аномалий свечных и объёмных (строго по ТЗ).

Свечная аномалия (точная формула из ТЗ):
- Размер свечи = (high - low) / high * 100
- Средний размер = mean всех размеров за candle_lookback свечей
- Отклонение каждой свечи = |размер свечи - средний размер|
- Среднее отклонение = mean всех отклонений
- Порог = средний размер + среднее отклонение
- Если текущая свеча > порога → аномалия

Quiet-режим:
- Если quiet_mode = true в конфиге → q_condition = true всегда
- Нет фильтра по волатильности — даже с аномалиями могут быть хорошие паттерны

Объёмная аномалия — полная заглушка (реализуем позже по процентилю)
cv_anomaly — полная заглушка (реализуем позже)
"""

import polars as pl
import numpy as np
from typing import List, Dict

from src.core.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

class AnomalyDetector:
    def __init__(self, config: dict):
        # Инициализация конфига
        self.config = config
        # Период для расчёта свечной аномалии
        self.candle_lookback = config["anomalies"]["candle_lookback"]  # 100
        # Период для расчёта объёмной аномалии (пока заглушка)
        self.volume_lookback = config["anomalies"]["volume_lookback"]  # 25
        # Quiet-режим из конфига
        self.quiet_mode = config["general"].get("quiet_mode", False)   # false по умолчанию

    def detect(self, df: pl.DataFrame) -> List[Dict]:
        """
        Детектирует аномалии для каждой свечи в df.
        Возвращает список словарей для каждой свечи.
        """
        results = []

        if len(df) < self.candle_lookback:
            # Если данных меньше, чем lookback — нет аномалий
            return [{"anomaly_type": None, "q_condition": False} for _ in range(len(df))]

        # Добавляем размер свечи в % для всех свечей
        df = df.with_columns(
            candle_size_pct = ((pl.col("high") - pl.col("low")) / pl.col("high") * 100)
        )

        for i in range(self.candle_lookback - 1, len(df)):
            # Берём последние candle_lookback свечей до текущей (не включая текущую)
            past = df.slice(i - self.candle_lookback + 1, self.candle_lookback - 1)

            # Средний размер свечи за весь период
            mean_candle = past["candle_size_pct"].mean()

            # Абсолютные отклонения от среднего
            deviations = (past["candle_size_pct"] - mean_candle).abs()

            # Среднее отклонение
            mean_dev = deviations.mean()

            # Порог аномалии
            threshold = mean_candle + mean_dev

            # Текущая свеча
            current_size_pct = df["candle_size_pct"][i]

            is_candle_anomaly = current_size_pct > threshold

            # Объёмная аномалия — полная заглушка (реализуем позже по процентилю)
            is_volume_anomaly = False

            # cv_anomaly — полная заглушка (реализуем позже)
            is_cv_anomaly = False

            # Quiet-режим: если включён — q_condition = true всегда
            is_q = self.quiet_mode

            anomaly_type = None
            if is_candle_anomaly:
                anomaly_type = "candle"
            if is_volume_anomaly:
                anomaly_type = "volume" if anomaly_type is None else f"{anomaly_type}+volume"
            if is_cv_anomaly:
                anomaly_type = "cv" if anomaly_type is None else f"{anomaly_type}+cv"

            results.append({
                "anomaly_type": anomaly_type,
                "q_condition": is_q
            })

        # Для первых свечей (до lookback) — нет аномалий
        padding = [{"anomaly_type": None, "q_condition": False}] * (self.candle_lookback - 1)
        return padding + results