# src/trading/websocket_manager.py
"""
Менеджер WebSocket-подписок на Binance Futures.

Ключевые требования и особенности:
- Подписка на 1m свечи для всех монет из whitelist (может быть 250+ монет)
- Разбиение на батчи по 30–50 монет (настраивается в конфиге), чтобы не превышать лимиты Binance (~200–300 подписок на соединение)
- Автоматическое переподключение при разрыве (экспоненциальная задержка: 1→2→4→8→16→60 сек)
- Обновление списка монет при изменении whitelist (раз в 5 минут или по событию)
- Если монета выпала из whitelist → отписка (через переподключение)
- Передача новых свечей в Resampler
- Логирование подключений, батчей, ошибок и количества подписок
- Heartbeat встроен в ccxt.pro — не влияет на лимиты REST API
"""

import asyncio
import time
from typing import List, Set
import ccxt.pro as ccxt

from src.core.config import load_config
from src.data.resampler import Resampler
from src.data.storage import Storage
from src.utils.logger import get_logger

logger = get_logger(__name__)

class WebSocketManager:
    def __init__(self, config: dict, resampler: Resampler):
        self.config = config
        self.resampler = resampler
        self.storage = Storage(config)

        self.exchange = ccxt.binance(config["exchange"])
        self.exchange.enableRateLimit = True

        # Батч-размер подписки (оптимально 30–50, лимит Binance ~200–300 на соединение)
        self.batch_size = config.get("websocket", {}).get("batch_size", 40)

        # Текущие активные подписки и whitelist
        self.active_subscriptions: Set[str] = set()
        self.current_whitelist: Set[str] = set(self.storage.get_whitelist())

        # Статус и задержка переподключения
        self.is_connected = False
        self.reconnect_delay = 1  # начальная задержка в секундах

    async def start(self):
        """Запуск менеджера подписок"""
        logger.info(f"WebSocketManager запущен | Батч-размер: {self.batch_size} | "
                    f"Монет в whitelist: {len(self.current_whitelist)}")

        while True:
            try:
                await self._maintain_subscriptions()
                await asyncio.sleep(60)  # проверка каждую минуту
            except Exception as e:
                logger.error(f"Критическая ошибка в WebSocketManager: {e}")
                await asyncio.sleep(10)

    async def _maintain_subscriptions(self):
        """Основной цикл поддержания актуальных подписок"""
        # Проверяем актуальный whitelist
        new_whitelist = set(self.storage.get_whitelist())

        if new_whitelist != self.current_whitelist:
            logger.info(f"Whitelist изменился: было {len(self.current_whitelist)}, стало {len(new_whitelist)}")
            await self._resubscribe(new_whitelist)
            self.current_whitelist = new_whitelist

        # Если не подключены — подключаемся
        if not self.is_connected:
            await self._connect_and_subscribe()

    async def _connect_and_subscribe(self):
        """Подключение и подписка на текущий whitelist"""
        symbols = list(self.current_whitelist)
        if not symbols:
            logger.warning("Whitelist пуст → подписка невозможна")
            self.is_connected = False
            return

        batches = [symbols[i:i + self.batch_size] for i in range(0, len(symbols), self.batch_size)]
        logger.info(f"Разбиваем подписку на {len(batches)} батчей по {self.batch_size} монет")

        self.is_connected = True
        self.reconnect_delay = 1  # сбрасываем задержку

        for batch in batches:
            asyncio.create_task(self._subscribe_batch(batch))

    async def _subscribe_batch(self, symbols: List[str]):
        """Подписка на один батч монет"""
        logger.info(f"Подписка на батч ({len(symbols)} монет): {', '.join(symbols[:5])}...")

        while True:
            try:
                async for ohlcv in self.exchange.watch_ohlcv(symbols, timeframe="1m"):
                    for candle in ohlcv:
                        symbol = candle["symbol"]
                        if symbol not in symbols:
                            continue
                        
                        # Передаём свечу в resampler
                        df_candle = pl.DataFrame({
                            "timestamp": [candle["timestamp"]],
                            "open": [candle["open"]],
                            "high": [candle["high"]],
                            "low": [candle["low"]],
                            "close": [candle["close"]],
                            "volume": [candle["volume"]]
                        })
                        
                        self.resampler.add_1m_candle(df_candle.to_dicts()[0])
                        logger.debug(f"Получена свеча {symbol} | close={candle['close']}")

            except ccxt.NetworkError as e:
                logger.warning(f"Сеть упала на батче {symbols[:3]}...: {e}")
                self.is_connected = False
                await asyncio.sleep(self.reconnect_delay)
                self.reconnect_delay = min(self.reconnect_delay * 2, 60)  # экспоненциальная задержка
            except ccxt.RateLimitExceeded:
                logger.warning("Rate limit WebSocket — ждём 30 сек")
                await asyncio.sleep(30)
            except Exception as e:
                logger.error(f"Ошибка подписки на батч {symbols[:3]}...: {e}")
                await asyncio.sleep(10)

    async def _resubscribe(self, new_whitelist: Set[str]):
        """Полная переподписка при изменении whitelist"""
        to_subscribe = new_whitelist - self.active_subscriptions
        to_unsubscribe = self.active_subscriptions - new_whitelist

        logger.info(f"Переподписка: +{len(to_subscribe)} | -{len(to_unsubscribe)}")

        # Закрываем текущее соединение и очищаем подписки
        await self.exchange.close()
        self.active_subscriptions.clear()
        self.is_connected = False

        # Запускаем новую подписку
        await self._connect_and_subscribe()

    def stop(self):
        """Остановка менеджера"""
        logger.info("WebSocketManager останавливается...")
        asyncio.create_task(self.exchange.close())