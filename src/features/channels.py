# src/features/channels.py
"""
Модуль расчёта ценовых каналов и Value Area (VA/POC).

Основные функции:
- calculate_price_channel()      — EMA-based канал (mid, upper, lower) + slope + breakout_strength
- calculate_value_area()         — Value Area 60% (POC, VAH, VAL) — объёмный профиль
- get_dynamic_tp_level()         — следующий уровень VAH/VAL или channel upper/lower

Используется в:
- feature_engine.py — добавляет признаки (channel_slope, breakout_strength, distance_to_channel)
- tp_sl_manager.py — для dynamic TP (TP = next VAH/VAL или channel upper/lower)
- risk_manager.py — для расчёта SL/TP расстояния

Логика:
- Price channel: EMA(mid) ± multiplier × EMA(ATR) — адаптивный канал
- Breakout_strength: сколько свечей подряд цена вне канала (0–5+)
- Channel_slope: наклон EMA upper - EMA lower (трендовость)
- VA: 60% объёма вокруг POC (по 30% вверх/вниз) — классический Volume Profile

Все расчёты на последних 100 свечах (основной период)
"""

import logging
from typing import Dict, Tuple, Optional

import polars as pl
import numpy as np

from src.core.config import load_config

logger = logging.getLogger(__name__)

class ChannelsCalculator:
    """Расчёт ценовых каналов и Value Area"""

    def __init__(self, config: Dict):
        self.config = config
        self.ema_span = config["features"].get("channel_ema_span", 20)      # Период EMA mid
        self.atr_period = config["features"].get("atr_period", 14)          # Период ATR
        self.channel_multiplier = config["features"].get("channel_multiplier", 2.0)  # Ширина канала
        self.va_percentage = 60.0                                           # Value Area — 60%

    def calculate_price_channel(self, df: pl.DataFrame, period: int = 100) -> Dict[str, float]:
        """
        Расчёт EMA-based ценового канала.

        Формула:
        mid = (high + low) / 2
        ema_mid = EMA(mid, span=ema_span)
        atr = ATR(high, low, close, period=atr_period)
        upper = ema_mid + multiplier × atr
        lower = ema_mid - multiplier × atr

        Дополнительно:
        - channel_slope = (upper[-1] - upper[-10]) / upper[-10] * 100  (наклон канала)
        - breakout_strength = кол-во свечей подряд вне канала (0–5+)
        - distance_to_upper/lower = (close - upper/lower) / upper/lower * 100

        Возвращает:
        {
            "ema_mid": float,
            "upper": float,
            "lower": float,
            "channel_slope": float,
            "breakout_strength": int,
            "distance_to_upper": float,
            "distance_to_lower": float
        }
        """
        if len(df) < self.ema_span + self.atr_period:
            logger.warning("Недостаточно свечей для канала (%d < %d)", len(df), self.ema_span + self.atr_period)
            return {"upper": 0, "lower": 0, "channel_slope": 0, "breakout_strength": 0, "distance_to_upper": 0, "distance_to_lower": 0}

        # 1. Mid price
        df = df.with_columns(
            ((pl.col("high") + pl.col("low")) / 2).alias("mid")
        )

        # 2. EMA mid
        df = df.with_columns(
            pl.col("mid").ewm_mean(span=self.ema_span, adjust=False).alias("ema_mid")
        )

        # 3. ATR (True Range)
        df = df.with_columns(
            pl.max_horizontal(
                pl.col("high") - pl.col("low"),
                (pl.col("high") - pl.col("close").shift(1)).abs(),
                (pl.col("low") - pl.col("close").shift(1)).abs()
            ).alias("tr")
        ).with_columns(
            pl.col("tr").ewm_mean(span=self.atr_period, adjust=False).alias("atr")
        )

        # 4. Канал
        df = df.with_columns(
            (pl.col("ema_mid") + self.channel_multiplier * pl.col("atr")).alias("upper"),
            (pl.col("ema_mid") - self.channel_multiplier * pl.col("atr")).alias("lower")
        )

        # 5. Channel slope (наклон за последние 10 свечей)
        slope = (df["upper"][-1] - df["upper"][-10]) / df["upper"][-10] * 100 if len(df) >= 10 else 0.0

        # 6. Breakout strength (сколько свечей подряд цена вне канала)
        df = df.with_columns(
            ((pl.col("close") > pl.col("upper")) | (pl.col("close") < pl.col("lower"))).alias("out_of_channel")
        )
        breakout_strength = df["out_of_channel"][-5:].sum()  # последние 5 свечей

        # 7. Distance to channel
        distance_upper = (df["close"][-1] - df["upper"][-1]) / df["upper"][-1] * 100
        distance_lower = (df["close"][-1] - df["lower"][-1]) / df["lower"][-1] * 100

        result = {
            "ema_mid": df["ema_mid"][-1],
            "upper": df["upper"][-1],
            "lower": df["lower"][-1],
            "channel_slope": slope,
            "breakout_strength": breakout_strength,
            "distance_to_upper": distance_upper,
            "distance_to_lower": distance_lower
        }

        logger.debug("Price channel: upper=%.2f, lower=%.2f, slope=%.2f%%, breakout=%d", 
                     result["upper"], result["lower"], result["channel_slope"], result["breakout_strength"])

        return result

    def calculate_value_area(self, df: pl.DataFrame, period: int = 100, bin_size_pct: float = 0.1) -> Dict[str, float]:
        """
        Value Area (VA) — 60% объёма вокруг POC (Point of Control)

        Алгоритм:
        1. Биннинг цен (шаг bin_size_pct % от средней цены)
        2. Подсчёт объёма в каждом бине
        3. POC = бин с максимальным объёмом
        4. VA = 60% общего объёма вокруг POC (по 30% вверх и вниз)
        5. VAH/VAL — верхняя/нижняя граница VA

        Возвращает:
        {
            "poc": float,           # Point of Control (цена с max volume)
            "vah": float,           # Value Area High
            "val": float            # Value Area Low
        }
        """
        if len(df) < 50:
            return {"poc": 0, "vah": 0, "val": 0}

        # 1. Средняя цена для биннинга
        avg_price = df["close"].mean()
        bin_size = avg_price * (bin_size_pct / 100)

        # 2. Биннинг цен (round to nearest bin)
        df = df.with_columns(
            (pl.col("close") / bin_size).round().cast(pl.Int64).alias("price_bin")
        )

        # 3. Объём по бинам
        volume_profile = df.group_by("price_bin").agg(
            pl.col("volume").sum().alias("bin_volume")
        ).sort("bin_volume", descending=True)

        if volume_profile.is_empty():
            return {"poc": 0, "vah": 0, "val": 0}

        # 4. POC — цена с max объёмом
        poc_bin = volume_profile["price_bin"][0]
        poc_price = poc_bin * bin_size

        # 5. Общий объём
        total_volume = volume_profile["bin_volume"].sum()

        # 6. Накопление 60% объёма вокруг POC
        target_volume = total_volume * (self.va_percentage / 100)
        accumulated = 0.0
        vah_bin, val_bin = poc_bin, poc_bin

        for row in volume_profile.iter_rows(named=True):
            bin_vol = row["bin_volume"]
            accumulated += bin_vol
            if row["price_bin"] > vah_bin:
                vah_bin = row["price_bin"]
            if row["price_bin"] < val_bin:
                val_bin = row["price_bin"]
            if accumulated >= target_volume:
                break

        vah = vah_bin * bin_size
        val = val_bin * bin_size

        result = {
            "poc": poc_price,
            "vah": vah,
            "val": val
        }

        logger.debug("Value Area: POC=%.2f, VAH=%.2f, VAL=%.2f", poc_price, vah, val)

        return result

    def get_dynamic_tp_level(self, df: pl.DataFrame, direction: str) -> float:
        """Следующий уровень для TP (dynamic_level)"""
        channel = self.calculate_price_channel(df)
        va = self.calculate_value_area(df)

        if direction == "L":  # Long
            return max(channel["upper"], va["vah"])
        else:  # Short
            return min(channel["lower"], va["val"])