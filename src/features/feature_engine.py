"""
src/features/feature_engine.py

=== Основной принцип работы файла ===

FeatureEngine — основной модуль генерации признаков для модели.
Теперь поддерживает feature_selection из конфига + все окна, VA, volume_percentile.
"""

import polars as pl
import numpy as np
from typing import Dict, Optional
from src.core.config import load_config
from src.features.anomaly_detector import AnomalyDetector
from src.features.channels import PriceChannel
from src.features.half_comparator import HalfComparator
from src.utils.logger import setup_logger

logger = setup_logger('feature_engine', logging.INFO)

class FeatureEngine:
    def __init__(self, config: dict):
        self.config = config
        self.anomaly_detector = AnomalyDetector(config)
        self.channel = PriceChannel(config)
        self.half_comparator = HalfComparator(config)

        # === НОВАЯ НАСТРОЙКА: feature_selection ===
        self.feature_selection = config["features"].get("selection", "all")
        self.windows_list = config["features"].get("windows_list", [24, 50, 74, 100])
        self.timeframes_list = config["features"].get("timeframes_list", ["1m", "3m", "5m", "10m", "15m"])
        self.va_percentage = config["features"].get("va_percentage", 60)
        self.volume_percentile = config["features"].get("volume_percentile", 95)
        self.half_comparison_period = config["features"].get("half_comparison_period", 100)

    async def build_features(self, windows: Dict[str, pl.DataFrame]) -> Dict:
        """
        Основной метод (вызывается из live_loop и engine).
        data = {tf: df}
        Возвращает агрегированные sequences + features по всем TF
        """
        result = {"sequences": {}, "features": {}}
        for tf, df in windows.items():
            if df is None or df.is_empty():
                continue

            single_result = self.compute_features(df)
            result["sequences"].update(single_result["sequences"])
            result["features"][tf] = single_result["features"]
        return result

    def compute_features(self, df: pl.DataFrame) -> Dict[str, Any]:
        """
        Внутренний метод: возвращает все признаки для данного df (один таймфрейм)
        """
        if df.is_empty():
            return {"sequences": {}, "features": {}}

        features = {}
        sequences = {}

        for window in self.windows_list:
            if len(df) < window:
                continue

            window_df = df.tail(window)
            agg = self._aggregate_features(window_df)
            seq = self._normalize_sequence(window_df)

            features[window] = agg
            sequences[window] = seq

        sequences = {k: torch.tensor(v, dtype=torch.float32) for k, v in sequences.items()}

        return {
            "sequences": sequences,
            "features": features
        }

    def _aggregate_features(self, df: pl.DataFrame) -> Dict[str, float]:
        if df.is_empty():
            return {}

        features = {}
        va_info = self._compute_value_area(df)
        current_close = df["close"].last()

        if va_info["VAH"] is None or va_info["VAL"] is None or va_info["VAH"] == va_info["VAL"]:
            va_position = 0.5
        else:
            va_position = (current_close - va_info["VAL"]) / (va_info["VAH"] - va_info["VAL"])
            va_position = max(0.0, min(1.0, va_position))

        features["va_position"] = va_position
        quiet_streak = self._compute_quiet_streak(df)
        features["quiet_streak"] = quiet_streak

        return features

    def _compute_value_area(self, df: pl.DataFrame) -> Dict[str, float | None]:
        if len(df) < 10:
            return {"VAH": None, "VAL": None, "POC": None}

        poc = df.group_by("close").agg(pl.col("volume").sum()).sort("volume", descending=True).first()["close"][0]
        price_min = df["low"].min()
        price_max = df["high"].max()
        price_range = price_max - price_min
        vah = poc + (self.va_percentage / 100) * price_range
        val = poc - (self.va_percentage / 100) * price_range

        return {"VAH": vah, "VAL": val, "POC": poc}

    def _compute_quiet_streak(self, df: pl.DataFrame) -> int:
        if len(df) < 2:
            return 0

        ranges = df["high"] - df["low"]
        window = min(len(ranges), self.quiet_window)
        avg_range = ranges.tail(window).mean()

        if avg_range == 0:
            return 0

        current_range = ranges.last()
        if current_range >= self.quiet_threshold * avg_range:
            return 0

        streak = 1
        for i in range(2, len(ranges) + 1):
            prev_range = ranges[-i]
            if prev_range < self.quiet_threshold * avg_range:
                streak += 1
            else:
                break
        return streak

    def _normalize_sequence(self, df: pl.DataFrame) -> np.ndarray:
        closes = df["close"].to_numpy()
        mean = closes.mean()
        std = closes.std()
        if std == 0:
            std = 1
        return (closes - mean) / std