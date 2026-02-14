# src/data/resampler.py
"""
Ресэмплинг 1m свечей в higher таймфреймы (3m, 5m, 10m, 15m).
Используется как в live_loop, так и в бэктесте.

Особенности:
- Работает на Polars (очень быстро)
- Инкрементальный ресэмплинг (в live не пересчитывает всё заново)
- Кэширование в памяти (coin → tf → DataFrame)
- Возвращает dict[str, pl.DataFrame] — именно то, что ожидает feature_engine.process()
- Поддержка всех TF из ТЗ
- Логирование + обработка ошибок
"""

import polars as pl
from typing import Dict, Optional
from datetime import timedelta

from ..utils.logger import logger
from ..core.config import get_config


class Resampler:
    def __init__(self):
        self.config = get_config()
        self.timeframes = ["1m", "3m", "5m", "10m", "15m"]
        
        # Кэш для live-режима: coin → {tf: DataFrame}
        self.cache: Dict[str, Dict[str, pl.DataFrame]] = {}

        logger.info("Resampler инициализирован", timeframes=self.timeframes[1:])

    def resample(self, df_1m: pl.DataFrame, target_tf: str) -> pl.DataFrame:
        """Ресэмплит 1m → target_tf (3m, 5m, 10m, 15m)."""
        if target_tf == "1m":
            return df_1m

        interval_map = {
            "3m": "3m",
            "5m": "5m",
            "10m": "10m",
            "15m": "15m"
        }

        if target_tf not in interval_map:
            raise ValueError(f"Неподдерживаемый таймфрейм: {target_tf}")

        # Преобразуем timestamp в datetime для group_by_dynamic
        df = df_1m.with_columns(
            pl.col("timestamp").cast(pl.Datetime("ms")).alias("dt")
        )

        resampled = df.group_by_dynamic(
            "dt",
            every=interval_map[target_tf],
            closed="left"
        ).agg(
            open=pl.first("open"),
            high=pl.max("high"),
            low=pl.min("low"),
            close=pl.last("close"),
            volume=pl.sum("volume"),
            buy_volume=pl.sum("buy_volume")
        )

        # Возвращаем timestamp обратно в int (ms)
        resampled = resampled.with_columns(
            pl.col("dt").cast(pl.Int64).alias("timestamp")
        ).drop("dt").sort("timestamp")

        return resampled

    def get_all_timeframes(self, coin: str, df_1m: pl.DataFrame) -> Dict[str, pl.DataFrame]:
        """
        Возвращает все таймфреймы для монеты.
        Использует кэш + инкрементальный ресэмплинг (очень важно для live).
        """
        if coin not in self.cache:
            self.cache[coin] = {}

        # Обновляем 1m
        self.cache[coin]["1m"] = df_1m

        # Ресэмплим higher TF (инкрементально)
        for tf in self.timeframes[1:]:
            cached = self.cache[coin].get(tf)

            if cached is None or len(cached) == 0:
                # Полный ресэмплинг
                new_tf = self.resample(df_1m, tf)
                self.cache[coin][tf] = new_tf
            else:
                # Добавляем только новые свечи
                last_ts = cached["timestamp"].max()
                new_rows = df_1m.filter(pl.col("timestamp") > last_ts)

                if not new_rows.is_empty():
                    new_resampled = self.resample(new_rows, tf)
                    self.cache[coin][tf] = pl.concat([cached, new_resampled]).unique("timestamp").sort("timestamp")

        logger.debug("Ресэмплинг завершён", coin=coin, tfs=list(self.cache[coin].keys()))
        return self.cache[coin]

    def clear_cache(self, coin: Optional[str] = None):
        """Очищает кэш (полезно при перезапуске или смене монет)."""
        if coin:
            self.cache.pop(coin, None)
            logger.debug("Кэш очищен для монеты", coin=coin)
        else:
            self.cache.clear()
            logger.info("Весь кэш ресэмплинга очищен")

    def get_cache_size(self) -> int:
        """Возвращает количество свечей во всех кэшах (для мониторинга)."""
        total = 0
        for coin_data in self.cache.values():
            for df in coin_data.values():
                total += len(df)
        return total