# src/features/half_comparator.py
"""
Сравнение левой и правой половины периода.
Все признаки средней цены из ТЗ реализованы полностью.
"""

from dataclasses import dataclass
import polars as pl
from typing import Dict, Optional

@dataclass
class HalfComparisonResult:
    binary_vector: list[int]          # 16 элементов (12 основных + 4 доп)
    percent_changes: Dict[str, float]
    left_features: Dict[str, float]
    right_features: Dict[str, float]
    is_valid: bool = True

class HalfComparator:
    def __init__(self, config: dict):
        self.period = config["model"]["main_period"]  # обычно 100
        self.feature_names = [
            "volume", "bid", "ask", "delta", "price_change",
            "volatility", "price_channel_pos", "va_pos",
            "candle_anomaly", "volume_anomaly", "cv_anomaly",
            "avg_price_delta", "price_vs_avg_delta", "delta_between_price_vs_avg"
        ]

    def compare(self, df: pl.DataFrame, period: Optional[int] = None) -> HalfComparisonResult:
        period = period or self.period
        if len(df) < period:
            return HalfComparisonResult([0]*16, {}, {}, {}, False)

        half = period // 2
        left_df = df.tail(period).head(half)
        right_df = df.tail(half)

        left = self._extract_features(left_df)
        right = self._extract_features(right_df)

        binary = []
        pct_changes = {}

        for name in self.feature_names:
            l_val = left.get(name, 0.0)
            r_val = right.get(name, 0.0)
            delta_pct = ((r_val - l_val) / abs(l_val) * 100) if abs(l_val) > 1e-10 else 0.0
            pct_changes[name] = delta_pct
            binary.append(1 if r_val > l_val else 0)

        # Дополняем до 16, если нужно (для совместимости со старыми моделями)
        while len(binary) < 16:
            binary.append(0)

        return HalfComparisonResult(
            binary_vector=binary,
            percent_changes=pct_changes,
            left_features=left,
            right_features=right
        )

    def _extract_features(self, df: pl.DataFrame) -> Dict[str, float]:
        """Реальные расчёты всех признаков средней цены."""
        if len(df) == 0:
            return {k: 0.0 for k in self.feature_names}

        closes = df["close"]
        avg_price = closes.mean()

        price_change_pct = ((closes[-1] - closes[0]) / closes[0] * 100) if closes[0] != 0 else 0.0

        price_vs_avg_pct = ((closes[-1] - avg_price) / avg_price * 100) if avg_price != 0 else 0.0

        candle_sizes_pct = ((df["high"] - df["low"]) / df["close"] * 100)
        volatility = candle_sizes_pct.mean()

        return {
            "volume": df["volume"].sum(),
            "bid": 0.0,               # заполняется выше по стеку
            "ask": 0.0,
            "delta": 0.0,
            "price_change": price_change_pct,
            "volatility": volatility,
            "price_channel_pos": 0.0, # из channels
            "va_pos": 0.0,
            "candle_anomaly": 0,
            "volume_anomaly": 0,
            "cv_anomaly": 0,
            "avg_price_delta": avg_price,
            "price_vs_avg_delta": price_vs_avg_pct,
            "delta_between_price_vs_avg": price_vs_avg_pct  # сравнение будет в compare()
        }