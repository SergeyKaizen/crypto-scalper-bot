# src/features/anomaly_detector.py
"""
Детектор аномалий — основной триггер для вызова модели.

По ТЗ реализовано 4 условия:
1. candle_anomaly     — размер свечи сильно превышает средний
2. volume_anomaly     — всплеск объёма (адаптивный алгоритм)
3. cv_anomaly         — комбинация свечной + объёмной аномалии
4. q_condition        — Quiet mode: нет ни одной из трёх аномалий (только если quiet_mode включён в конфиге)

Аномалия — это сигнал для подачи последовательности свечей в модель.
Без аномалии (и без quiet_mode) предсказание не запускается.

Параметры берутся из конфига и могут быть разными для phone_tiny / server.
"""

import polars as pl
from typing import Dict, Tuple, Any
import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)

class AnomalyDetector:
    def __init__(self, config: dict):
        anomalies_cfg = config.get("anomalies", {})

        # Периоды расчёта (из ТЗ)
        self.candle_lookback = anomalies_cfg.get("candle_lookback", 100)   # для свечной аномалии
        self.volume_lookback = anomalies_cfg.get("volume_lookback", 25)    # для объёмной

        # Quiet mode
        self.quiet_mode_enabled = config.get("quiet_mode", False)

        # Пороги
        self.candle_std_multiplier = anomalies_cfg.get("candle_std_multiplier", 1.5)
        self.volume_perc_threshold = anomalies_cfg.get("volume_perc_threshold", 75)
        self.norm_factor = anomalies_cfg.get("norm_factor", 1.5)

        # Опциональный фильтр: игнорировать сверхсильные импульсы
        self.max_candle_multiplier = anomalies_cfg.get("max_candle_multiplier", None)  # None = отключено

        self.debug = config.get("debug", False)

    def detect(self, df: pl.DataFrame) -> Dict[str, Any]:
        """
        Основной метод.
        Проверяет последнюю свечу на 4 условия.
        Возвращает словарь с флагами и типом аномалии.
        """
        if len(df) < max(self.candle_lookback, self.volume_lookback):
            return {
                "candle_anomaly": False,
                "volume_anomaly": False,
                "cv_anomaly": False,
                "q_condition": False,
                "anomaly_type": None,
                "strength": 0.0,
                "details": {"error": "недостаточно свечей"}
            }

        # 1. Свечная аномалия
        candle_anomaly, candle_strength = self._detect_candle_anomaly(df)

        # 2. Объёмная аномалия
        volume_anomaly, volume_strength = self._detect_volume_anomaly(df)

        # 3. Комбинированная
        cv_anomaly = candle_anomaly and volume_anomaly

        # 4. Quiet condition
        q_condition = self.quiet_mode_enabled and not (candle_anomaly or volume_anomaly or cv_anomaly)

        # Определяем основной тип аномалии (приоритет: cv > candle > volume > q)
        if cv_anomaly:
            anomaly_type = "cv"
            strength = max(candle_strength, volume_strength)
        elif candle_anomaly:
            anomaly_type = "candle"
            strength = candle_strength
        elif volume_anomaly:
            anomaly_type = "volume"
            strength = volume_strength
        elif q_condition:
            anomaly_type = "q"
            strength = 1.0
        else:
            anomaly_type = None
            strength = 0.0

        result = {
            "candle_anomaly": candle_anomaly,
            "volume_anomaly": volume_anomaly,
            "cv_anomaly": cv_anomaly,
            "q_condition": q_condition,
            "anomaly_type": anomaly_type,
            "strength": min(strength, 5.0),
            "details": {
                "candle_strength": round(candle_strength, 3),
                "volume_strength": round(volume_strength, 3)
            }
        }

        if self.debug and anomaly_type:
            logger.debug(f"[ANOMALY] {anomaly_type.upper()} | strength={strength:.3f} | symbol={df.get_column('symbol')[0] if 'symbol' in df.columns else 'unknown'}")

        return result

    def _detect_candle_anomaly(self, df: pl.DataFrame) -> Tuple[bool, float]:
        """Свечная аномалия: размер свечи > mean + N*std"""
        recent = df.tail(self.candle_lookback)

        tr_pct = ((recent["high"] - recent["low"]) / recent["close"]) * 100
        body_pct = ((recent["close"] - recent["open"]).abs() / recent["close"]) * 100
        sizes = pl.max_horizontal(tr_pct, body_pct)

        mean_size = sizes.mean()
        std_size = sizes.std()
        last_size = sizes[-1]

        threshold = mean_size + self.candle_std_multiplier * std_size

        is_anomaly = last_size > threshold
        strength = last_size / mean_size if mean_size > 0 else 0.0

        # Фильтр на сверхсильные импульсы
        if is_anomaly and self.max_candle_multiplier is not None:
            if strength > self.max_candle_multiplier:
                is_anomaly = False
                strength = 0.0
                if self.debug:
                    logger.debug(f"[IGNORED] Candle anomaly too strong: {strength:.2f}x")

        return is_anomaly, strength

    def _detect_volume_anomaly(self, df: pl.DataFrame) -> Tuple[bool, float]:
        """Объёмная аномалия — адаптивный алгоритм"""
        recent = df.tail(self.volume_lookback)

        total_vol = recent["volume"].sum()
        if total_vol == 0:
            return False, 0.0

        last_vol = recent["volume"][-1]
        avg_vol = recent["volume"].mean()
        vol_ratio = last_vol / avg_vol if avg_vol > 0 else 0.0

        perc = (recent["volume"] <= last_vol).mean() * 100

        trend_factor = 1.0
        if len(recent) >= 5:
            if recent["close"][-1] > recent["close"][-5]:
                trend_factor = 1.2
            elif recent["close"][-1] < recent["close"][-5]:
                trend_factor = 0.8

        threshold = self.volume_perc_threshold + (100 - self.volume_perc_threshold) * (1 - 1 / self.norm_factor)

        is_anomaly = (perc >= threshold) and (vol_ratio > trend_factor)
        strength = (perc / 100) * vol_ratio

        return is_anomaly, strength


# ────────────────────────────────────────────────────────────────
# Тестовый запуск
# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from src.core.config import load_config
    config = load_config()
    config["quiet_mode"] = True                     # для теста Q-режима
    config["debug"] = True

    detector = AnomalyDetector(config)

    # Тестовый датафрейм (без аномалий → должен сработать q_condition)
    df = pl.DataFrame({
        "open_time": list(range(100)),
        "open": [60000.0] * 100,
        "high": [60100.0] * 100,
        "low": [59900.0] * 100,
        "close": [60000.0 + i * 0.1 for i in range(100)],
        "volume": [10.0] * 100
    })

    result = detector.detect(df)
    print(result)