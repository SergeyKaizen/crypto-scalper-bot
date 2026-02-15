# src/features/feature_engine.py
"""
Модуль генерации признаков для последовательностей свечей.

Всего 13 признаков:
- OHLCV + производные
- признаки средней цены (из ТЗ)
- 4 бинарных условия: candle_anomaly, volume_anomaly, cv_anomaly, q_condition

Value Area (VA) — ровно 60 % объёма (классический Volume Profile):
- 50 ценовых бинов
- накопление от самого объёмного бина до 60 %
- POC = max volume bin
- VAH/VAL = границы 60 %
- va_pos = (close - POC) / POC * 100 (% от POC)
"""

import polars as pl
import numpy as np
from typing import Dict, Optional
from src.features.half_comparator import HalfComparator
from src.features.anomaly_detector import AnomalyDetector
from src.core.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

class FeatureEngine:
    def __init__(self, config: dict):
        self.config = config
        self.half_comparator = HalfComparator(config)
        self.anomaly_detector = AnomalyDetector(config)
        
        # 13 признаков (q_condition добавлен)
        self.feature_names = [
            "open", "high", "low", "close", "volume", "buy_volume",
            "price_change", "volatility", "price_channel_pos", "va_pos",
            "delta_volume", "delta_price",
            "avg_price_delta", "price_vs_avg_delta", "delta_between_price_vs_avg",
            "candle_anomaly", "volume_anomaly", "cv_anomaly", "q_condition"
        ]
        
        self.channel_period = config.get("features", {}).get("channel_period", 100)
        self.va_bins = config.get("features", {}).get("va_bins", 50)  # кол-во ценовых бинов

    def compute_sequence_features(self, df: pl.DataFrame) -> np.ndarray:
        """Вычисляет все признаки для каждой свечи в последовательности."""
        if len(df) < 10:
            logger.warning("Слишком короткая последовательность")
            return np.zeros((len(df), len(self.feature_names)), dtype=np.float32)

        # Базовые OHLCV + производные
        df = df.with_columns([
            (pl.col("close") / pl.col("close").shift(1) - 1).alias("price_change"),
            ((pl.col("high") - pl.col("low")) / pl.col("close")).alias("volatility"),
            (pl.col("volume") - pl.col("volume").shift(1)).alias("delta_volume"),
            (pl.col("close") - pl.col("close").shift(1)).alias("delta_price")
        ])

        # Ценовой канал (Donchian)
        df = df.with_columns([
            pl.col("high").rolling_max(window_size=self.channel_period).alias("channel_high"),
            pl.col("low").rolling_min(window_size=self.channel_period).alias("channel_low")
        ]).with_columns(
            ((pl.col("close") - pl.col("channel_low")) / 
             (pl.col("channel_high") - pl.col("channel_low") + 1e-9)).alias("price_channel_pos")
        )

        # Value Area — ровно 60 % объёма (классическая реализация)
        df = self._add_va_60_percent(df)

        # Признаки средней цены и сравнение половин
        df = self._add_half_comparison_features(df)

        # 4 бинарных условия
        df = self._add_anomaly_flags(df)

        # Заполняем NaN и null
        df = df.fill_nan(0).fill_null(strategy="forward")

        return df.select(self.feature_names).to_numpy().astype(np.float32)

    def _add_va_60_percent(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Value Area ровно 60 % объёма — классический Volume Profile.

        1. Диапазон цен → 50 бинов
        2. Объём по бинам
        3. Сортировка бинов по объёму descending
        4. Накопление до 60 %
        5. POC = max volume bin
        6. VAH/VAL = границы накопленных бинов
        7. va_pos = (close - POC_price) / POC_price * 100 (%)
        """
        period = min(50, len(df))
        if period < 10:
            return df.with_columns(pl.lit(0.0).alias("va_pos"))

        recent = df.tail(period)
        total_vol = recent["volume"].sum()
        if total_vol <= 0:
            return df.with_columns(pl.lit(0.0).alias("va_pos"))

        min_p = recent["low"].min()
        max_p = recent["high"].max()
        price_range = max_p - min_p
        if price_range <= 0:
            return df.with_columns(pl.lit(0.0).alias("va_pos"))

        bin_size = price_range / self.va_bins

        # Присваиваем бин каждой свече (по close)
        recent = recent.with_columns(
            (((pl.col("close") - min_p) / bin_size).floor().cast(pl.Int32).clip(0, self.va_bins - 1)).alias("price_bin")
        )

        # Объём по бинам + сортировка descending
        vol_by_bin = (
            recent.group_by("price_bin")
            .agg(pl.col("volume").sum().alias("vol"))
            .sort("vol", descending=True)
        )

        # Накопление до 60 %
        cum_vol = vol_by_bin.with_columns(
            pl.col("vol").cum_sum().alias("cum_vol")
        )

        target_vol = total_vol * 0.60
        va_bins_list = cum_vol.filter(pl.col("cum_vol") <= target_vol)["price_bin"].to_list()

        if not va_bins_list:
            va_bins_list = [vol_by_bin["price_bin"][0]]  # хотя бы POC

        # POC — самый объёмный бин
        poc_bin = vol_by_bin["price_bin"][0]
        poc_price = min_p + (poc_bin + 0.5) * bin_size  # середина бина

        # VAH/VAL — границы
        vah_bin = max(va_bins_list)
        val_bin = min(va_bins_list)
        vah_price = min_p + (vah_bin + 1) * bin_size
        val_price = min_p + val_bin * bin_size

        # Позиция текущей цены относительно POC (в %)
        df = df.with_columns(
            ((pl.col("close") - poc_price) / (poc_price + 1e-9) * 100).alias("va_pos")
        )

        return df

    def _add_half_comparison_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Признаки средней цены из ТЗ (усреднены по окнам)"""
        for w in [24, 50, 74, 100]:
            if len(df) < w:
                continue
            rolling = df.tail(w)
            comp = self.half_comparator.compare(rolling, period=w)
            if comp.is_valid:
                df = df.with_columns([
                    pl.lit(comp.percent_changes.get("avg_price_delta", 0.0)).alias(f"avg_price_delta_{w}"),
                    pl.lit(comp.percent_changes.get("price_vs_avg_delta", 0.0)).alias(f"price_vs_avg_delta_{w}"),
                    pl.lit(comp.percent_changes.get("delta_between_price_vs_avg", 0.0)).alias(f"delta_between_price_vs_avg_{w}")
                ])

        df = df.with_columns([
            pl.mean_horizontal([f"avg_price_delta_{w}" for w in [24, 50, 74, 100] if f"avg_price_delta_{w}" in df.columns]).alias("avg_price_delta"),
            pl.mean_horizontal([f"price_vs_avg_delta_{w}" for w in [24, 50, 74, 100] if f"price_vs_avg_delta_{w}" in df.columns]).alias("price_vs_avg_delta"),
            pl.mean_horizontal([f"delta_between_price_vs_avg_{w}" for w in [24, 50, 74, 100] if f"delta_between_price_vs_avg_{w}" in df.columns]).alias("delta_between_price_vs_avg")
        ])
        return df

    def _add_anomaly_flags(self, df: pl.DataFrame) -> pl.DataFrame:
        """4 бинарных условия"""
        anomalies = self.anomaly_detector.detect(df)
        
        q_condition = [0] * len(df)
        if self.config.get("quiet_mode", False):
            for i, row in enumerate(anomalies):
                if not (row.get("candle_anomaly") or row.get("volume_anomaly") or row.get("cv_anomaly")):
                    q_condition[i] = 1

        df = df.with_columns([
            pl.Series("candle_anomaly", [1 if row.get("candle_anomaly", False) else 0 for row in anomalies]),
            pl.Series("volume_anomaly", [1 if row.get("volume_anomaly", False) else 0 for row in anomalies]),
            pl.Series("cv_anomaly",     [1 if row.get("cv_anomaly", False) else 0 for row in anomalies]),
            pl.Series("q_condition",    q_condition)
        ])
        return df

    def get_last_features(self, symbol: str, tf: str) -> Dict[str, float]:
        df = self.storage.load_candles(symbol, tf, limit=1)
        if df.is_empty():
            return {}
        feat_array = self.compute_sequence_features(df)
        return dict(zip(self.feature_names, feat_array[-1]))


# Тест
if __name__ == "__main__":
    config = load_config()
    engine = FeatureEngine(config)
    
    df = pl.DataFrame({
        "open_time": list(range(100)),
        "open": [60000 + i*10 for i in range(100)],
        "high": [60100 + i*10 for i in range(100)],
        "low": [59900 + i*10 for i in range(100)],
        "close": [60050 + i*10 for i in range(100)],
        "volume": [10 + i*0.1 for i in range(100)],
        "buy_volume": [6 + i*0.05 for i in range(100)]
    })
    
    features = engine.compute_sequence_features(df)
    print("Shape:", features.shape)
    print("Последние 5 признаков (включая va_pos):", features[-1, -5:])