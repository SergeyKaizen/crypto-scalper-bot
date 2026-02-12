# src/features/volume_features.py
"""
Модуль расчёта объёмных признаков и объёмной аномалии.

Основные функции:
- calculate_volume_features() — 4–6 объёмных признаков (delta %, momentum, cumulative delta и т.д.)
- is_volume_anomaly() — объёмная аномалия (total volume + delta bid/ask)
- prepare_delta_sign() — знак дельты (positive → bid-доминирование → шорт-сигнал)

Используется в:
- feature_engine.py — добавляет признаки в 12 базовых
- anomalies.py — для V и CV аномалий

Логика:
- Delta bid/ask = (bid_volume - ask_volume) / total_volume × 100%
- Volume anomaly: total_volume > 95-й перцентиль + (если use_delta) |delta| > threshold
- Cumulative delta — накопленная дельта за последние N свечей (показывает давление)
- Volume momentum — изменение объёма за последние 5–10 свечей

Все расчёты на последних 100 свечах (основной период)
"""

import logging
from typing import Dict

import polars as pl

from src.core.config import load_config

logger = logging.getLogger(__name__)

class VolumeFeaturesCalculator:
    """Расчёт всех объёмных признаков и аномалий"""

    def __init__(self, config: Dict):
        self.config = config
        self.use_delta = config["features"].get("use_delta_in_volume_anomaly", True)
        self.delta_threshold = config["features"].get("volume_delta_threshold", 25.0)  # % — пример, можно отключить

    def calculate_volume_features(self, df: pl.DataFrame) -> Dict[str, float]:
        """
        Расчёт 4–6 объёмных признаков (по половинам и последним свечам)

        Возвращает:
        {
            "volume_delta_pct": float,           # (right - left) / left * 100
            "bid_ask_delta_last": float,         # delta bid/ask за последнюю свечу
            "bid_ask_delta_avg": float,          # средняя delta за 10 свечей
            "volume_momentum": float,            # изменение объёма за последние 5 свечей
            "cumulative_delta": float,           # накопленная delta bid/ask за 20 свечей
            "volume_spike_ratio": float          # last_volume / mean_volume
        }
        """
        if len(df) < 20:
            logger.warning("Недостаточно свечей для объёмных признаков")
            return {
                "volume_delta_pct": 0.0,
                "bid_ask_delta_last": 0.0,
                "bid_ask_delta_avg": 0.0,
                "volume_momentum": 0.0,
                "cumulative_delta": 0.0,
                "volume_spike_ratio": 0.0
            }

        # 1. Делим на половины (по ТЗ — 50/50 из 100)
        half = len(df) // 2
        left = df.slice(0, half)
        right = df.slice(half, half)

        volume_left = left["volume"].sum()
        volume_right = right["volume"].sum()
        volume_delta_pct = (volume_right - volume_left) / volume_left * 100 if volume_left > 0 else 0.0

        # 2. Delta bid/ask за последнюю свечу
        last_total = df["volume"][-1]
        bid_ask_delta_last = 0.0
        if last_total > 0:
            bid_ask_delta_last = (df["bid_volume"][-1] - df["ask_volume"][-1]) / last_total * 100

        # 3. Средняя delta за последние 10 свечей
        recent = df.tail(10)
        bid_ask_delta_avg = 0.0
        if not recent.is_empty():
            recent_total = recent["volume"].sum()
            if recent_total > 0:
                bid_ask_delta_avg = (recent["bid_volume"].sum() - recent["ask_volume"].sum()) / recent_total * 100

        # 4. Volume momentum (изменение объёма за последние 5 свечей)
        last_5 = df.tail(5)["volume"]
        volume_momentum = (last_5[-1] - last_5[0]) / last_5[0] * 100 if last_5[0] > 0 else 0.0

        # 5. Cumulative delta (накопленная delta bid/ask за 20 свечей)
        recent_20 = df.tail(20)
        cumulative_delta = (recent_20["bid_volume"].sum() - recent_20["ask_volume"].sum()) / recent_20["volume"].sum() * 100

        # 6. Volume spike ratio (последний объём / средний)
        mean_volume = df["volume"].mean()
        volume_spike_ratio = df["volume"][-1] / mean_volume if mean_volume > 0 else 0.0

        features = {
            "volume_delta_pct": volume_delta_pct,
            "bid_ask_delta_last": bid_ask_delta_last,
            "bid_ask_delta_avg": bid_ask_delta_avg,
            "volume_momentum": volume_momentum,
            "cumulative_delta": cumulative_delta,
            "volume_spike_ratio": volume_spike_ratio
        }

        logger.debug("Объёмные признаки: delta_last=%.1f%%, spike=%.2f, momentum=%.1f%%", 
                     bid_ask_delta_last, volume_spike_ratio, volume_momentum)

        return features

    def is_volume_anomaly(self, df: pl.DataFrame) -> bool:
        """
        Объёмная аномалия (V)

        Формула:
        1. last_volume > 95-й перцентиль за 100 свечей
        2. Если use_delta — |bid_ask_delta_last| > delta_threshold
        """
        if len(df) < 50:
            return False

        last_volume = df["volume"][-1]
        volumes = df["volume"]
        percentile_95 = volumes.quantile(0.95)

        volume_condition = last_volume > percentile_95

        if not self.use_delta:
            return volume_condition

        last_total = df["volume"][-1]
        if last_total == 0:
            return volume_condition

        delta_pct = (df["bid_volume"][-1] - df["ask_volume"][-1]) / last_total * 100
        delta_condition = abs(delta_pct) > self.delta_threshold

        if volume_condition and delta_condition:
            direction = "bid-dominant (short)" if delta_pct > 0 else "ask-dominant (long)"
            logger.debug("Объёмная аномалия + delta: %s, delta=%.1f%%, volume=%.2f", 
                         direction, delta_pct, last_volume)
            return True

        return volume_condition

    def get_volume_spike_direction(self, df: pl.DataFrame) -> str:
        """Направление всплеска объёма (для логов и отладки)"""
        if not self.is_volume_anomaly(df):
            return "No spike"

        delta_pct = (df["bid_volume"][-1] - df["ask_volume"][-1]) / df["volume"][-1] * 100
        if delta_pct > self.delta_threshold:
            return "bid-dominant (short likely)"
        elif delta_pct < -self.delta_threshold:
            return "ask-dominant (long likely)"
        else:
            return "balanced spike"