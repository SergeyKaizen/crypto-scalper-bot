# src/trading/websocket_manager.py
"""
Динамическая подписка на WebSocket Binance.
Добавлено подробное логирование подписок и событий.
"""

import asyncio
from typing import Dict, Set

from binance import AsyncClient, BinanceSocketManager
from binance.enums import FuturesType

from ..utils.logger import logger
from ..core.config import get_config


class WebSocketManager:
    def __init__(self, live_loop):
        self.live_loop = live_loop
        self.config = get_config()
        self.client = None
        self.bsm = None
        self.task = None

        self.active_streams: Dict[str, Set[str]] = {}
        self.shadow_streams: Set[str] = set()

        logger.info("WebSocketManager создан")

    async def start(self):
        logger.info("Запуск WebSocketManager")
        self.client = await AsyncClient.create(
            api_key=self.config["binance"]["api_key"],
            api_secret=self.config["binance"]["secret_key"]
        )
        self.bsm = BinanceSocketManager(self.client)

        self.task = asyncio.create_task(self._run())

    async def _run(self):
        logger.info("WebSocket цикл запущен")
        while True:
            try:
                await self._rebalance()
                await asyncio.sleep(300)  # каждые 5 минут
            except Exception as e:
                logger.error("Ошибка в WebSocket цикле", error=str(e))
                await asyncio.sleep(60)

    async def _rebalance(self):
        logger.debug("Перебалансировка подписок",
                     active_count=len(self.active_streams),
                     shadow_count=len(self.shadow_streams))

        # Active монеты
        for coin, cfg in self.live_loop.traded_coins.items():
            tfs = {cfg.tf}
            if coin not in self.active_streams or self.active_streams[coin] != tfs:
                logger.info("Переподписка на активную монету",
                            coin=coin,
                            tfs=list(tfs))
                await self._subscribe(coin, tfs, is_active=True)

        # Shadow монеты — только 1m
        # Логика заполнения self.shadow_streams должна быть в live_loop
        # Здесь только пример
        pass

    async def _subscribe(self, symbol: str, tfs: Set[str], is_active: bool):
        streams = [f"{symbol.lower()}@kline_{tf}" for tf in tfs]
        if not streams:
            logger.warning("Пустой список TF для подписки", symbol=symbol)
            return

        logger.info("Подписка на стримы",
                    symbol=symbol,
                    streams=streams,
                    type="active" if is_active else "shadow")

        # Здесь должен быть реальный код подписки через bsm
        # Для примера — заглушка
        pass

    async def stop(self):
        logger.info("Остановка WebSocketManager")
        if self.task:
            self.task.cancel()
        if self.client:
            await self.client.close_connection()