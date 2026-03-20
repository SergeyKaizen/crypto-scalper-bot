"""
src/features/feature_engine.py

=== Основной принцип работы файла ===

FeatureEngine — основной модуль генерации признаков для модели.
Теперь поддерживает feature_selection из конфига + все окна, VA, volume_percentile.
"""

import polars as pl
import numpy as np
import torch
from typing import Dict, Any
from src.core.config import load_config
from src.features.anomaly_detector import AnomalyDetector
from src.utils.logger import setup_logger

# Импортируем реальные функции (repo-style, без классов)
from src.features.half_comparator import compare_halves
from src.features.channels import anomalous_surge_channel_feature, calculate_volume_profile_va

logger = setup_logger('feature_engine', logging.INFO)

class FeatureEngine:
    def __init__(self, config: dict):
        self.config = config
        self.anomaly_detector = AnomalyDetector()  # FIX: без config

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
        """Обновлённый метод: теперь включает half_changes + channel features"""
        if df.is_empty():
            return {}

        features = {}
        current_close = float(df["close"].last())

        # Value Area (оставлено)
        va_info = self._compute_value_area(df)
        if va_info["VAH"] is None or va_info["VAL"] is None or va_info["VAH"] == va_info["VAL"]:
            va_position = 0.5
        else:
            va_position = (current_close - va_info["VAL"]) / (va_info["VAH"] - va_info["VAL"])
            va_position = max(0.0, min(1.0, va_position))

        features["va_position"] = va_position

        # === NEW: Half Comparator (полные half_changes по ТЗ) ===
        # Используем реальную функцию из repo
        half_changes = compare_halves(
            window_df=df.to_pandas(),
            window_size=len(df),
            va_std=calculate_volume_profile_va(df.to_pandas()),
            va_delta=calculate_volume_profile_va(df.to_pandas(), use_delta=True),
            current_price=current_close
        )
        features.update(half_changes)

        # === NEW: Channel features (anomalous surge + VA) ===
        channel_df = anomalous_surge_channel_feature(df.to_pandas(), period=self.half_comparison_period)
        features["asc_norm_dist_to_upper"] = float(channel_df["asc_norm_dist_to_upper"].iloc[-1])
        features["asc_norm_dist_to_lower"] = float(channel_df["asc_norm_dist_to_lower"].iloc[-1])
        features["asc_position_in_channel"] = float(channel_df["asc_position_in_channel"].iloc[-1])

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

    def _normalize_sequence(self, df: pl.DataFrame) -> np.ndarray:
        closes = df["close"].to_numpy()
        mean = closes.mean()
        std = closes.std()
        if std == 0:
            std = 1
        return (closes - mean) / std