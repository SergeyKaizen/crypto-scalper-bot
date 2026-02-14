# ================================================
# 2. src/features/anomaly_detector.py
# ================================================
"""
ЕДИНЫЙ ДЕТЕКТОР всех трёх аномалий (C, V, CV).
Работает на любом таймфрейме, возвращает унифицированный список сигналов.
"""

from dataclasses import dataclass
from enum import Enum
import polars as pl
from typing import List, Tuple

class AnomalyType(Enum):
    CANDLE = "C"
    VOLUME = "V"
    CANDLE_VOLUME = "CV"

class Direction(Enum):
    LONG = "L"
    SHORT = "S"

@dataclass
class AnomalySignal:
    """Унифицированный сигнал аномалии."""
    tf: str
    anomaly_type: AnomalyType
    direction_hint: Direction      # подсказка для отскока
    strength: float                # 0.0–1.0
    threshold: float
    current_volume: float
    candle_size_pct: float
    timestamp: int                 # unix ms закрытия свечи


class AnomalyDetector:
    """Центральный класс обнаружения аномалий по ТЗ."""

    def __init__(self, config: dict):
        """Загружаем параметры из выбранного режима торговли."""
        mode = config["trading_mode"]
        self.cfg = config["anomalies"][mode]
        self.lookback = self.cfg["lookback"]          # 25
        self.min_perc = self.cfg["min_perc"]          # 0.94
        self.max_perc = self.cfg["max_perc"]          # 0.99
        self.flat_mult = self.cfg["flat_mult"]        # 1.18

    def detect_all(self, dfs: dict[str, pl.DataFrame]) -> List[AnomalySignal]:
        """
        Главный метод.
        Принимает словарь датафреймов по всем TF и возвращает все найденные аномалии.
        """
        signals = []
        
        for tf, df in dfs.items():
            if len(df) < self.lookback + 10:
                continue
                
            tf_minutes = int(tf[:-1]) if tf.endswith("m") else 1
            
            # Свечная аномалия
            is_c, strength_c = self._candle_anomaly(df)
            if is_c:
                signals.append(self._create_signal(tf, AnomalyType.CANDLE, df, strength_c))
            
            # Объёмная аномалия
            is_v, strength_v, _ = self._volume_anomaly(df, tf_minutes)
            if is_v:
                signals.append(self._create_signal(tf, AnomalyType.VOLUME, df, strength_v))
            
            # Комбинированная CV
            if is_c and is_v:
                signals.append(self._create_signal(tf, AnomalyType.CANDLE_VOLUME, df, max(strength_c, strength_v)))

        return signals

    def _candle_anomaly(self, df: pl.DataFrame) -> Tuple[bool, float]:
        """Реализация свечной аномалии точно по описанию в ТЗ (PDF стр.6)."""
        period = 100
        if len(df) < period:
            return False, 0.0
            
        sizes = ((df["high"] - df["low"]) / df["close"] * 100).tail(period)
        mean_size = sizes.mean()
        deviations = (sizes - mean_size).abs()
        mean_dev = deviations.mean()
        anomaly_threshold = mean_size + mean_dev
        
        last_size = sizes[-1]
        is_anomaly = last_size > anomaly_threshold
        strength = min(1.0, last_size / anomaly_threshold) if is_anomaly else 0.0
        
        return is_anomaly, strength

    def _volume_anomaly(self, df: pl.DataFrame, tf_minutes: int) -> Tuple[bool, float, float]:
        """Полностью адаптированный код из PDF (стр.7-9) на Polars."""
        past = df.tail(self.lookback + 1).head(self.lookback)
        if len(past) < self.lookback:
            return False, 0.0, 0.0
            
        vols = past["volume"]
        ranges = past["high"] - past["low"]
        
        med_vol = vols.median()
        med_range = ranges.median()
        
        # Масштабирование перцентиля по TF (как в Pine Script)
        tf_factor = max(1.0, tf_minutes / 5.0)
        perc_level = max(self.min_perc, self.max_perc - (1.0 / (self.lookback / tf_factor)))
        
        perc_threshold = vols.sort().quantile(perc_level)
        
        # Нормализация по ширине текущей свечи
        current = df.tail(1)
        current_range_pct = ((current["high"] - current["low"]) / current["close"])[0]
        avg_range = ranges.mean()
        norm_factor = avg_range / current_range_pct if current_range_pct > 0 else 1.0
        
        adjusted_threshold = perc_threshold * max(1.0, norm_factor * 0.8)
        
        current_vol = current["volume"][0]
        is_high_vol = current_vol > adjusted_threshold
        
        if not is_high_vol:
            return False, 0.0, perc_threshold
            
        # Проверка трендовости
        price_change = abs(current["close"][0] - df["close"][-self.lookback-1])
        trend_threshold = med_range * (self.lookback / 3.8)
        is_trend = price_change > trend_threshold
        
        is_anomaly = is_trend or (current_vol > med_vol * self.flat_mult)
        strength = min(1.0, current_vol / adjusted_threshold) if is_anomaly else 0.0
        
        return is_anomaly, strength, perc_threshold

    def _create_signal(self, tf: str, atype: AnomalyType, df: pl.DataFrame, strength: float) -> AnomalySignal:
        """Создаёт унифицированный сигнал с правильной направленностью для отскока."""
        current = df.tail(1)
        is_bullish = current["close"][0] > current["open"][0]
        
        direction = Direction.LONG if is_bullish else Direction.SHORT
        # Для отскока: бычья свеча → шорт, медвежья → лонг
        if atype in (AnomalyType.CANDLE, AnomalyType.CANDLE_VOLUME):
            direction = Direction.SHORT if is_bullish else Direction.LONG
            
        return AnomalySignal(
            tf=tf,
            anomaly_type=atype,
            direction_hint=direction,
            strength=strength,
            threshold=0.0,
            current_volume=current["volume"][0],
            candle_size_pct=((current["high"]-current["low"])/current["close"]*100)[0],
            timestamp=int(current["timestamp"][0])
        )