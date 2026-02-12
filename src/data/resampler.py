# src/data/resampler.py
"""
Модуль ресэмплинга (пересчёта) данных с 1m на более высокие таймфреймы.

Функционал:
- Берёт 1m свечи из БД или API
- Пересчитывает OHLCV + volume + bid_volume + ask_volume на 3m, 5m, 10m, 15m
- Использует Polars (быстрее pandas, columnar)
- Учитывает конфиг (какие TF нужно ресэмплить)
- Сохраняет результат в БД (чтобы не пересчитывать каждый раз)

Когда использовать:
- Если high TF нет в API или данные устарели
- На телефоне — экономия API-запросов
- Для проверки целостности данных (1m → 5m должен совпадать с нативным 5m)

Зависимости:
- polars (быстрый DataFrame)
- src/data/storage.py (для чтения/записи)
"""

import logging
from typing import List, Dict, Optional
from datetime import datetime

import polars as pl

from src.core.config import load_config
from src.data.storage import Storage

logger = logging.getLogger(__name__)

# Маппинг: сколько 1m свечей в одном high TF
TIMEFRAME_MINUTES = {
    '1m': 1,
    '3m': 3,
    '5m': 5,
    '10m': 10,
    '15m': 15,
}

class Resampler:
    """Ресэмплинг 1m данных на higher timeframes"""

    def __init__(self, config: Dict):
        self.config = config
        self.storage = Storage(config)

    async def resample(self, symbol: str, target_tf: str, since: Optional[int] = None, limit: int = 10000):
        """
        Пересчитывает 1m свечи на target_tf (3m, 5m и т.д.)
        
        Args:
            symbol: "BTCUSDT"
            target_tf: "5m", "15m" и т.д.
            since: timestamp (ms) — с какого момента
            limit: сколько 1m свечей взять (для безопасности)

        Returns:
            List[Dict] — свечи в формате target_tf
        """
        if target_tf == '1m':
            logger.warning("Resample to 1m requested — returning raw data")
            return await self.storage.get_candles(symbol, '1m', since, limit)

        minutes = TIMEFRAME_MINUTES.get(target_tf)
        if not minutes:
            raise ValueError(f"Unsupported timeframe for resample: {target_tf}")

        # 1. Берём 1m свечи
        one_min_candles = await self.storage.get_candles(symbol, '1m', since, limit * minutes)
        if not one_min_candles:
            logger.warning("No 1m candles for %s", symbol)
            return []

        # 2. Преобразуем в Polars DataFrame
        df = pl.DataFrame(one_min_candles)
        df = df.with_columns(
            pl.col("timestamp").cast(pl.Int64).alias("ts_ms")
        ).sort("ts_ms")

        # 3. Группируем по target_tf (каждые N минут)
        df = df.with_columns(
            (pl.col("ts_ms") // (minutes * 60 * 1000)).alias("group")
        )

        # 4. Ресэмплинг OHLCV + volume + bid/ask
        resampled = df.group_by("group").agg(
            pl.col("ts_ms").min().alias("timestamp"),               # Время открытия новой свечи
            pl.col("open").first().alias("open"),                   # Первая open
            pl.col("high").max().alias("high"),                     # Максимум high
            pl.col("low").min().alias("low"),                       # Минимум low
            pl.col("close").last().alias("close"),                  # Последняя close
            pl.col("volume").sum().alias("volume"),                 # Сумма volume
            pl.col("bid_volume").sum().alias("bid_volume"),         # Сумма bid_volume
            pl.col("ask_volume").sum().alias("ask_volume")          # Сумма ask_volume
        ).sort("timestamp")

        # 5. Преобразуем обратно в список словарей
        candles = resampled.to_dicts()

        # 6. Сохраняем в БД (если нужно)
        if self.config["data"]["resample_higher_tf"]:
            await self.storage.save_ohlcv(symbol, target_tf, candles)

        logger.info("Resampled %d candles from 1m to %s for %s", len(candles), target_tf, symbol)
        return candles

    async def validate_resample(self, symbol: str, target_tf: str):
        """Проверка: ресэмпл 1m → target_tf совпадает с нативными данными Binance"""
        # Опциональная функция для отладки
        native = await self.client.fetch_ohlcv(symbol, target_tf, limit=100)
        resampled = await self.resample(symbol, target_tf, limit=100 * TIMEFRAME_MINUTES[target_tf])
        # Сравнение (можно добавить assert или лог)
        logger.debug("Validation resample for %s %s: native=%d, resampled=%d", symbol, target_tf, len(native), len(resampled))

# Пример использования (для тестов)
if __name__ == "__main__":
    import asyncio

    async def test():
        config = load_config()
        resampler = Resampler(config)
        candles = await resampler.resample("BTCUSDT", "5m", limit=1000)
        print("Resampled 5m candles:", len(candles))

    asyncio.run(test())