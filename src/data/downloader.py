# src/data/downloader.py
"""
Модуль скачивания данных с Binance Futures.

Функционал:
- Полная история (full history) от даты листинга монеты
- Инкрементальное обновление (live) — догоняем новые свечи
- Автоматическое добавление новых монет в "модуль монет"
- Учёт лимитов Binance (1500 свечей за запрос)
- Retry при ошибках + rate-limit защита
- Асинхронная работа (asyncio)

Используется в:
- scripts/download_full.py (полная скачка)
- live_loop.py (догон live-свечей)
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict

import ccxt.async_support as ccxt

from src.core.config import load_config
from src.data.binance_client import BinanceClient
from src.data.storage import Storage  # Абстракция БД

logger = logging.getLogger(__name__)

class Downloader:
    """Скачивание и обновление данных с Binance Futures"""

    def __init__(self, config: Dict):
        self.config = config
        self.client = BinanceClient(
            api_key=config["binance"]["api_key"],
            api_secret=config["binance"]["api_secret"],
            testnet=config["binance"]["testnet"]
        )
        self.storage = Storage(config)  # Для сохранения свечей и списка монет
        self.max_history = config["data"]["max_history_candles"]

    async def update_markets_list(self) -> List[str]:
        """Обновляет список всех USDT perpetual фьючерсов (раз в сутки)"""
        markets = await self.client.exchange.load_markets()
        usdt_perpetual = [
            symbol for symbol in markets
            if symbol.endswith('USDT') and markets[symbol]['type'] == 'swap'
        ]

        # Сохраняем в БД/конфиг
        current_coins = await self.storage.get_current_coins()
        new_coins = [s for s in usdt_perpetual if s not in current_coins]

        if new_coins:
            logger.info("Found %d new coins: %s", len(new_coins), new_coins[:5])
            await self.storage.add_coins(new_coins)

        # Удаляем удалённые монеты
        removed = [s for s in current_coins if s not in usdt_perpetual]
        if removed:
            logger.info("Removing %d delisted coins: %s", len(removed), removed[:5])
            await self.storage.remove_coins(removed)

        return usdt_perpetual

    async def download_full_history(self, symbol: str, timeframe: str = '1m') -> None:
        """Скачивает полную историю для монеты (если не скачана)"""
        # Проверяем, есть ли уже данные
        last_ts = await self.storage.get_last_timestamp(symbol, timeframe)
        if last_ts:
            logger.info("History for %s (%s) already exists, last: %s", symbol, timeframe, last_ts)
            return

        since = await self._get_listed_since(symbol)
        logger.info("Downloading full history for %s (%s) from %s", symbol, timeframe, datetime.fromtimestamp(since / 1000))

        all_ohlcv = []
        current_since = since
        while True:
            ohlcv = await self.client.fetch_ohlcv(symbol, timeframe, current_since, BINANCE_KLINES_LIMIT_PER_REQUEST)
            if not ohlcv:
                break

            all_ohlcv.extend(ohlcv)
            current_since = ohlcv[-1][0] + 1  # Следующая свеча

            # Ограничение по конфигу
            if len(all_ohlcv) >= self.max_history:
                logger.warning("Reached max_history_candles (%d) for %s", self.max_history, symbol)
                break

            await asyncio.sleep(0.5)  # Защита от rate-limit

        if all_ohlcv:
            await self.storage.save_ohlcv(symbol, timeframe, all_ohlcv)
            logger.info("Saved %d candles for %s (%s)", len(all_ohlcv), symbol, timeframe)

    async def _get_listed_since(self, symbol: str) -> int:
        """Дата листинга монеты (или fallback 2 года назад)"""
        markets = await self.client.exchange.load_markets()
        if symbol in markets and 'info' in markets[symbol]:
            listing_time = markets[symbol]['info'].get('listingTime')
            if listing_time:
                return int(listing_time)

        # Fallback — 2 года назад
        return int((datetime.now() - timedelta(days=730)).timestamp() * 1000)

    async def fetch_new_candles(self, symbol: str, timeframe: str) -> List[List[float]]:
        """Догоняет новые свечи с последнего timestamp"""
        last_ts = await self.storage.get_last_timestamp(symbol, timeframe)
        if not last_ts:
            logger.warning("No history for %s (%s). Need full download first.", symbol, timeframe)
            return []

        ohlcv = await self.client.fetch_ohlcv(symbol, timeframe, last_ts)
        if ohlcv:
            await self.storage.save_ohlcv(symbol, timeframe, ohlcv)
            logger.info("Added %d new candles for %s (%s)", len(ohlcv), symbol, timeframe)

        return ohlcv

    async def close(self):
        """Закрытие клиента"""
        await self.client.close()