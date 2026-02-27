"""
src/features/feature_engine.py

Центральный модуль генерации признаков.
Принимает окна свечей по нескольким TF и возвращает:
- sequences: Dict[tf, torch.Tensor] — последовательности для модели
- features: Dict[tf, dict] — агрегированные фичи (статистики, индикаторы, аномалии и т.д.)

Работает асинхронно, кэширует тяжёлые расчёты (ATR, VWAP и т.д.)
Использует polars для скорости на больших окнах.
"""

import logging
import torch
import polars as pl
import numpy as np
from typing import Dict, Any, Optional

from src.core.config import load_config
from src.features.anomalies import detect_anomalies
from src.features.channels import calculate_donchian, calculate_keltner
from src.features.half_comparator import HalfComparator
from src.utils.logger import setup_logger

setup_logger()
logger = logging.getLogger(__name__)


class FeatureEngine:
    def __init__(self, config: dict):
        self.config = config
        self.seq_len = config["seq_len"]
        self.timeframes = config["timeframes"]
        self.half_comparator = HalfComparator(config)
        self.cache = {}  # простой кэш для повторяющихся окон

    async def build_features(self, windows: Dict[str, pl.DataFrame]) -> Dict[str, Any]:
        """
        Основной метод генерации признаков.
        windows: {tf: DataFrame} — окна свечей по каждому TF
        Возвращает:
        {
            "sequences": {tf: torch.Tensor [seq_len, n_features]},
            "features": {tf: dict агрегированных признаков}
        }
        """
        sequences = {}
        agg_features = {}

        for tf, df in windows.items():
            if len(df) < self.seq_len:
                logger.warning(f"Окно {tf} слишком короткое: {len(df)} < {self.seq_len}")
                continue

            # Основные OHLCV + volume
            df = df.with_columns([
                (pl.col("close") / pl.col("open") - 1).alias("return"),
                (pl.col("high") - pl.col("low")).alias("range"),
                ((pl.col("close") - pl.col("open")) / (pl.col("high") - pl.col("low") + 1e-9)).alias("body_ratio"),
            ])

            # Кэшируем тяжёлые индикаторы
            cache_key = f"{tf}_{df['open_time'].max()}"
            if cache_key in self.cache:
                seq, agg = self.cache[cache_key]
            else:
                # Рассчитываем индикаторы
                df = self._add_indicators(df)

                # Детекция аномалий
                anomalies = detect_anomalies(df)

                # Каналы
                donchian = calculate_donchian(df)
                keltner = calculate_keltner(df)

                # ← ФИКС: Вызов half_comparator после regime separation (предварительное разделение на режимы внутри compare_regimes)
                regime_features = self.half_comparator.compare_regimes(df)

                # Агрегированные фичи
                agg = self._aggregate_features(df, anomalies, donchian, keltner, regime_features)

                # Sequence — последние seq_len строк числовых колонок
                numeric_cols = df.select(pl.exclude(["open_time", "symbol"])).columns
                seq_df = df.select(numeric_cols).tail(self.seq_len)
                seq = torch.tensor(seq_df.to_numpy(), dtype=torch.float32)

                self.cache[cache_key] = (seq, agg)

            sequences[tf] = seq
            agg_features[tf] = agg

        return {
            "sequences": sequences,
            "features": agg_features
        }

    def _add_indicators(self, df: pl.DataFrame) -> pl.DataFrame:
        """Добавляет стандартные индикаторы (EMA, RSI, ATR, VWAP и т.д.)"""
        # EMA 14
        df = df.with_columns(pl.col("close").ewm_mean(span=14).alias("ema_14"))

        # ATR 14 (простая реализация)
        tr = pl.max_horizontal(
            pl.col("high") - pl.col("low"),
            (pl.col("high") - pl.col("close").shift(1)).abs(),
            (pl.col("low") - pl.col("close").shift(1)).abs()
        )
        atr = tr.ewm_mean(span=14, adjust=False).alias("atr")
        df = df.with_columns(atr)

        # RSI 14
        delta = pl.col("close").diff()
        gain = delta.clip_min(0)
        loss = (-delta).clip_min(0)
        avg_gain = gain.ewm_mean(span=14, adjust=False)
        avg_loss = loss.ewm_mean(span=14, adjust=False)
        rs = avg_gain / (avg_loss + 1e-9)
        rsi = (100 - (100 / (1 + rs))).alias("rsi_14")
        df = df.with_columns(rsi)

        return df

    def _aggregate_features(self, df: pl.DataFrame, anomalies: Dict, donchian: Dict,
                            keltner: Dict, regime: Dict) -> Dict[str, Any]:
        """Собирает агрегированные фичи за окно — теперь с реальными расчётами"""
        last_row = df.tail(1)
        window_returns = df["return"].to_numpy()

        # Volume Area (VA) position — где текущая цена относительно VAH/VAL (простая аппроксимация)
        volume_profile = df.group_by_dynamic("close", every="0.5%").agg(pl.col("volume").sum())
        if not volume_profile.is_empty():
            poc_idx = volume_profile["volume"].arg_max()
            poc_price = volume_profile["close"].item(poc_idx)
            va_high = volume_profile["close"].max()
            va_low = volume_profile["close"].min()
            va_position = (last_row["close"][0] - va_low) / (va_high - va_low + 1e-9) if va_high != va_low else 0.5
        else:
            va_position = 0.5

        # Sequential count (как в DeMark) — упрощённо: сколько свечей подряд close > prev close
        sequential_up = (df["close"] > df["close"].shift(1)).cast(pl.Int32).rle_id().agg(pl.col("close").len()).max()
        sequential_down = (df["close"] < df["close"].shift(1)).cast(pl.Int32).rle_id().agg(pl.col("close").len()).max()

        # Quiet streak — сколько последних свечей range < ATR * 0.5 (низкая волатильность)
        atr = last_row["atr"][0] if "atr" in last_row.columns else 0.001
        quiet_mask = (df["range"] < atr * 0.5)
        quiet_streak = quiet_mask.cast(pl.Int32).rle_id().agg(pl.col("range").len()).max() if quiet_mask.any() else 0

        return {
            "close": last_row["close"][0],
            "volume_mean": df["volume"].mean(),
            "volatility_atr": atr,
            "anomaly_score": anomalies.get("score", 0.0),
            "donchian_width": donchian.get("width", 0.0),
            "keltner_squeeze": keltner.get("squeeze", False),
            "regime_bull_strength": regime.get("bull_strength", 0.0),
            "regime_bear_strength": regime.get("bear_strength", 0.0),

            # Реальные значения вместо 0.0
            "va_position": float(va_position),
            "sequential_up": int(sequential_up) if sequential_up is not None else 0,
            "sequential_down": int(sequential_down) if sequential_down is not None else 0,
            "quiet_streak": int(quiet_streak),

            # Дополнительные простые статистики
            "return_mean": float(window_returns.mean()),
            "return_std": float(window_returns.std()),
            "rsi_14": last_row["rsi_14"][0] if "rsi_14" in last_row.columns else 50.0,
            "ema_distance": (last_row["close"][0] - last_row["ema_14"][0]) / last_row["ema_14"][0] if "ema_14" in last_row.columns else 0.0,
        }

    def clear_cache(self):
        """Очистка кэша при смене модели / конфига"""
        self.cache.clear()