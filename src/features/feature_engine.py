"""
src/features/feature_engine.py

=== Основной принцип работы файла ===

Центральный модуль генерации признаков для модели.

Ключевые задачи:
- build_features(data) → основной метод для live/backtest (принимает dict {tf: df})
- compute_features(df) → внутренний для одного таймфрейма
- _aggregate_features(df) → агрегация по окнам (24, 50, 74, 100)
- VA_position, quiet_streak — без look-ahead bias
"""

import polars as pl
import numpy as np
from typing import Dict, Any
import torch

from src.core.config import load_config
from src.utils.logger import setup_logger

logger = setup_logger("feature_engine", logging.INFO)


class FeatureEngine:
    def __init__(self, config: dict = None):
        self.config = config or load_config()
        self.windows = self.config.get("windows", [24, 50, 74, 100])
        self.quiet_window = 20
        self.quiet_threshold = 0.4

    def build_features(self, data: Dict[str, pl.DataFrame]) -> Dict[str, Any]:
        """
        Основной метод (вызывается из live_loop и engine).
        data = {tf: df}
        Возвращает агрегированные sequences + features по всем TF
        """
        result = {"sequences": {}, "features": {}}
        for tf, df in data.items():
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

        for window in self.windows:
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
        vah = poc + 0.35 * price_range
        val = poc - 0.35 * price_range

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