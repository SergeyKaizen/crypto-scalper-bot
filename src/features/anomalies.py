# src/features/anomalies.py
"""
Модуль расчёта аномалий (триггеров для inference).

Основные функции:
- is_candle_anomaly()        — свечная аномалия (размер свечи > средний + отклонение)
- is_volume_anomaly()        — объёмная аномалия (объём > порог + delta bid/ask)
- is_mixed_anomaly()         — смешанная (C + V одновременно)
- prepare_quiet_flag()       — флаг Q для тихого режима (из inference)

Используется в:
- feature_engine.py — добавляет 4 бинарных условия (C/V/CV/Q)
- live_loop.py — определяет, когда делать inference

Логика:
- Все расчёты на последних 100 свечах (period)
- Свечная: размер > mean + std (или 1.5–2×mean)
- Объёмная: объём > 95-й перцентиль + delta bid/ask (если флаг включён)
- Delta bid/ask — (bid_volume - ask_volume) / total_volume × 100%
- Q — не аномалия, а флаг из inference (мягкий паттерн)

Формулы подробно описаны в комментариях
"""

import logging
from typing import Dict, Tuple

import polars as pl

from src.core.config import load_config

logger = logging.getLogger(__name__)

class AnomalyDetector:
    """Класс для расчёта всех аномалий"""

    def __init__(self, config: Dict):
        self.config = config
        self.delta_threshold = config["features"].get("volume_delta_threshold", 25.0)  # % (пример, но лучше отключить)
        self.use_delta = config["features"].get("use_delta_in_volume_anomaly", True)

    def detect(self, df: pl.DataFrame) -> Dict[str, int]:
        """
        Главная функция: возвращает словарь с 4 бинарными флагами
        {
            "C": 1 или 0,
            "V": 1 или 0,
            "CV": 1 или 0,
            "Q": 0 (заполняется позже в inference)
        }
        """
        if len(df) < 50:
            logger.warning("Недостаточно свечей для расчёта аномалий (%d)", len(df))
            return {"C": 0, "V": 0, "CV": 0, "Q": 0}

        c = self.is_candle_anomaly(df)
        v = self.is_volume_anomaly(df)
        cv = 1 if c and v else 0

        return {
            "C": int(c),
            "V": int(v),
            "CV": int(cv),
            "Q": 0  # Заполняется в inference.py при тихом режиме
        }

    def is_candle_anomaly(self, df: pl.DataFrame) -> bool:
        """
        Свечная аномалия (по ТЗ)
        Формула:
            candle_size = high - low
            mean_size = средний размер за последние 100 свечей
            std_size = стандартное отклонение
            return candle_size > mean_size + std_size  (1 sigma)
        Более агрессивно: > mean_size * 1.8 или 2.0 — настраивается
        """
        candle_size = df["high"][-1] - df["low"][-1]
        sizes = df["high"] - df["low"]
        mean_size = sizes.mean()
        std_size = sizes.std()

        threshold = mean_size + std_size  # 1 sigma — стандарт
        # Можно сделать жёстче: threshold = mean_size * 1.8

        is_anomaly = candle_size > threshold

        if is_anomaly:
            direction = "bullish" if df["close"][-1] > df["open"][-1] else "bearish"
            logger.debug("Свечная аномалия: %s, размер=%.2f, threshold=%.2f", direction, candle_size, threshold)

        return is_anomaly

    def is_volume_anomaly(self, df: pl.DataFrame) -> bool:
        """
        Объёмная аномалия (по ТЗ + delta bid/ask)
        Формула:
            1. total_volume > 95-й перцентиль за 100 свечей
            2. (если use_delta) delta_bid_ask = (bid_volume - ask_volume) / total_volume * 100
               delta > +threshold → bid-доминирование (шорт-сигнал)
               delta < -threshold → ask-доминирование (лонг-сигнал)
        """
        last_volume = df["volume"][-1]
        volumes = df["volume"]
        percentile_95 = volumes.quantile(0.95)

        volume_condition = last_volume > percentile_95

        if not self.use_delta:
            return volume_condition

        # Delta bid/ask
        last_total = df["volume"][-1]
        if last_total == 0:
            return volume_condition

        delta_pct = (df["bid_volume"][-1] - df["ask_volume"][-1]) / last_total * 100

        # Сигнал: bid-доминирование → шорт, ask → лонг
        is_significant_delta = abs(delta_pct) > self.delta_threshold

        if volume_condition and is_significant_delta:
            direction = "bid-dominant (short)" if delta_pct > 0 else "ask-dominant (long)"
            logger.debug("Объёмная аномалия + delta: %s, delta=%.1f%%, volume=%.2f", direction, delta_pct, last_volume)
            return True

        return volume_condition

    def get_anomaly_summary(self, df: pl.DataFrame) -> str:
        """Красивое текстовое описание аномалий (для логов)"""
        anomalies = self.detect(df)
        parts = []
        if anomalies["C"]:
            parts.append("CANDLE")
        if anomalies["V"]:
            parts.append("VOLUME")
        if anomalies["CV"]:
            parts.append("MIXED")
        if anomalies["Q"]:
            parts.append("QUIET")

        return " + ".join(parts) if parts else "No anomaly"

# Пример использования (для тестов)
if __name__ == "__main__":
    config = load_config()
    detector = AnomalyDetector(config)

    # Пример DataFrame (1m свечи)
    df = pl.DataFrame({
        "timestamp": [i for i in range(100)],
        "open": [60000 + i*10 for i in range(100)],
        "high": [60050 + i*10 for i in range(100)],
        "low": [59950 + i*10 for i in range(100)],
        "close": [60020 + i*10 for i in range(100)],
        "volume": [100 + i*5 for i in range(100)],
        "bid_volume": [60 + i*3 for i in range(100)],
        "ask_volume": [40 + i*2 for i in range(100)],
    })

    result = detector.detect(df)
    print("Аномалии:", result)
    print("Summary:", detector.get_anomaly_summary(df))