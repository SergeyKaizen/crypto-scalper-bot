# src/features/anomaly_detector.py
"""
Детектор аномалий — основной триггер для вызова модели.

По ТЗ:
- 3 основных условия: свечная аномалия, объёмная, свечная+объёмная
- Добавлено 4-е условие: q_condition (Quiet mode) — когда НЕТ ни одной из 3-х аномалий
- q_condition активируется только если quiet_mode включён в конфиге
- Обучение на всех TF и окнах, выявление лучшего TF в момент аномалии
- Аномалия — сигнал для подачи последовательности в модель

Вывод: словарь с 4-мя флагами (candle, volume, cv, q) + тип + сила
"""

import polars as pl
from typing import Dict, Tuple, Any
from src.core.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

class AnomalyDetector:
    def __init__(self, config: dict):
        anomalies_cfg = config.get("anomalies", {})

        # Периоды расчёта (из ТЗ)
        self.candle_lookback       = anomalies_cfg.get("candle_lookback",      100)  # для свечной аномалии
        self.volume_lookback       = anomalies_cfg.get("volume_lookback",      25)   # для объёмной
        self.quiet_mode_enabled    = config.get("quiet_mode", False)                  # флаг Q-режима

        # Пороги
        self.candle_std_multiplier = anomalies_cfg.get("candle_std_multiplier", 1.5)
        self.volume_perc_threshold = anomalies_cfg.get("volume_perc_threshold", 75)   # перцентиль
        self.norm_factor           = anomalies_cfg.get("norm_factor",           1.5)   # тренд/флет

        # Опционально: игнорировать сверхсильные импульсы (пампы/дампы)
        self.max_candle_multiplier = anomalies_cfg.get("max_candle_multiplier", None)  # None = отключено

        self.debug = config.get("debug", False)

    def detect(self, df: pl.DataFrame) -> Dict[str, Any]:
        """
        Проверяет последнюю свечу на 4 условия:
        - candle_anomaly
        - volume_anomaly
        - cv_anomaly (комбинированная)
        - q_condition (только если quiet_mode включён и нет других аномалий)

        Возвращает:
        {
            "candle_anomaly": bool,
            "volume_anomaly": bool,
            "cv_anomaly":     bool,
            "q_condition":    bool,
            "anomaly_type":   str or None,  # "candle", "volume", "cv", "q" или None
            "strength":       float,
            "details":        dict
        }
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

        # Последняя свеча
        last = df[-1]

        # 1. Свечная аномалия
        candle_anomaly, candle_strength = self._detect_candle_anomaly(df)

        # 2. Объёмная аномалия
        volume_anomaly, volume_strength = self._detect_volume_anomaly(df)

        # 3. Комбинированная
        cv_anomaly = candle_anomaly and volume_anomaly

        # 4. Quiet condition — только если включён режим и нет других аномалий
        q_condition = self.quiet_mode_enabled and not (candle_anomaly or volume_anomaly or cv_anomaly)

        # Определяем основной тип (приоритет: cv > candle > volume > q)
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
            strength = 1.0  # базовая сила для quiet
        else:
            anomaly_type = None
            strength = 0.0

        result = {
            "candle_anomaly": candle_anomaly,
            "volume_anomaly": volume_anomaly,
            "cv_anomaly":     cv_anomaly,
            "q_condition":    q_condition,
            "anomaly_type":   anomaly_type,
            "strength":       min(strength, 5.0),
            "details": {
                "candle_strength": candle_strength,
                "volume_strength": volume_strength,
                "last_candle_size_pct": candle_strength * 100 if candle_anomaly else 0,
                "last_volume_pct": volume_strength * 100 if volume_anomaly else 0
            }
        }

        if self.debug and anomaly_type:
            logger.debug(f"[ANOMALY] {anomaly_type.upper()} | strength: {strength:.2f} | symbol: {df['symbol'][0] if 'symbol' in df.columns else 'unknown'}")

        return result

    def _detect_candle_anomaly(self, df: pl.DataFrame) -> Tuple[bool, float]:
        """Свечная аномалия: размер свечи > mean + N*std"""
        recent = df.tail(self.candle_lookback)

        tr_pct = ((recent["high"] - recent["low"]) / recent["close"]) * 100
        body_pct = ((recent["close"] - recent["open"]).abs() / recent["close"]) * 100
        sizes = pl.max_horizontal(tr_pct, body_pct)

        mean_size = sizes.mean()
        std_size  = sizes.std()
        last_size = sizes[-1]

        threshold = mean_size + self.candle_std_multiplier * std_size

        is_anomaly = last_size > threshold
        strength   = last_size / mean_size if mean_size > 0 else 0.0

        # Опциональный фильтр на сверхсильные движения
        if is_anomaly and self.max_candle_multiplier is not None:
            if strength > self.max_candle_multiplier:
                is_anomaly = False
                strength = 0.0
                if self.debug:
                    logger.debug(f"[IGNORED] Candle too strong: {strength:.2f}x")

        return is_anomaly, strength

    def _detect_volume_anomaly(self, df: pl.DataFrame) -> Tuple[bool, float]:
        """Объёмная аномалия — адаптивный алгоритм (Pine-подобный)"""
        recent = df.tail(self.volume_lookback)

        total_vol = recent["volume"].sum()
        if total_vol == 0:
            return False, 0.0

        last_vol = recent["volume"][-1]
        avg_vol  = recent["volume"].mean()
        vol_ratio = last_vol / avg_vol if avg_vol > 0 else 0.0

        # Перцентиль объёма
        perc = (recent["volume"] <= last_vol).mean() * 100

        # Учёт тренда (последние 5 свечей)
        trend_factor = 1.0
        if len(recent) >= 5:
            if recent["close"][-1] > recent["close"][-5]:
                trend_factor = 1.2
            elif recent["close"][-1] < recent["close"][-5]:
                trend_factor = 0.8

        threshold = self.volume_perc_threshold + (100 - self.volume_perc_threshold) * (1 - 1/self.norm_factor)

        is_anomaly = (perc >= threshold) and (vol_ratio > trend_factor)
        strength   = (perc / 100) * vol_ratio

        return is_anomaly, strength


# Тестовый запуск
if __name__ == "__main__":
    config = load_config()
    config["quiet_mode"] = True  # для теста Q-режима
    detector = AnomalyDetector(config)

    # Тестовый датафрейм (без аномалий → q_condition должен быть True)
    df = pl.DataFrame({
        "open_time": list(range(100)),
        "open":  [60000.0] * 100,
        "high":  [60100.0] * 100,
        "low":   [59900.0] * 100,
        "close": [60000.0 + i*0.1 for i in range(100)],
        "volume": [10.0] * 100
    })

    result = detector.detect(df)
    print(result)