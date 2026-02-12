# src/data/binance_client.py
"""
Обёртка над ccxt для Binance Futures.
Обеспечивает:
- rate-limit контроль
- retry при ошибках (с экспоненциальной задержкой)
- асинхронную работу (asyncio)
- логирование всех запросов
- обработку ошибок Binance (Invalid API-key, Rate limit, etc.)

Используется во всех модулях, где нужны данные с биржи:
- downloader.py
- order_executor.py
- live_loop.py
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
import ccxt.async_support as ccxt
from ccxt.base.errors import RateLimitExceeded, AuthenticationError, NetworkError

from src.core.config import load_config

logger = logging.getLogger(__name__)

class BinanceClient:
    """Клиент для Binance Futures с защитой от банов и retry"""

    def __init__(self, api_key: str, api_secret: str, testnet: bool = False):
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet

        # Инициализация ccxt
        self.exchange = ccxt.binance({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future',  # Только perpetual futures USDT-M
                'adjustForTimeDifference': True,
                'recvWindow': 10000,  # 10 секунд — стандарт
            },
            'urls': {
                'api': {
                    'public': 'https://fapi.binance.com/fapi/v1',
                    'private': 'https://fapi.binance.com/fapi/v1',
                }
            } if not testnet else {
                'api': {
                    'public': 'https://testnet.binancefuture.com/fapi/v1',
                    'private': 'https://testnet.binancefuture.com/fapi/v1',
                }
            }
        })

        # Максимальное количество retry
        self.max_retries = 5
        self.base_delay = 1  # секунды

        logger.info("BinanceClient initialized (testnet: %s)", testnet)

    async def _retry_request(self, func, *args, **kwargs) -> Any:
        """Универсальный retry-механизм для всех запросов"""
        for attempt in range(self.max_retries):
            try:
                result = await func(*args, **kwargs)
                return result
            except RateLimitExceeded as e:
                delay = self.base_delay * (2 ** attempt)  # экспоненциальная задержка
                logger.warning("Rate limit exceeded. Waiting %s seconds... (attempt %d/%d)",
                               delay, attempt + 1, self.max_retries)
                await asyncio.sleep(delay)
            except NetworkError as e:
                delay = self.base_delay * (2 ** attempt)
                logger.warning("Network error: %s. Retrying in %s seconds...", e, delay)
                await asyncio.sleep(delay)
            except AuthenticationError as e:
                logger.error("Authentication error: %s. Check API keys!", e)
                raise
            except Exception as e:
                logger.error("Unexpected error in request: %s", e)
                raise

        raise Exception(f"Request failed after {self.max_retries} retries")

    async def fetch_ohlcv(self, symbol: str, timeframe: str, since: Optional[int] = None,
                          limit: int = 1500) -> List[List[float]]:
        """Получение свечей (OHLCV)"""
        return await self._retry_request(
            self.exchange.fetch_ohlcv,
            symbol,
            timeframe,
            since=since,
            limit=limit
        )

    async def fetch_ticker(self, symbol: str) -> Dict:
        """Текущая цена и данные тикера"""
        return await self._retry_request(self.exchange.fetch_ticker, symbol)

    async def fetch_balance(self) -> Dict:
        """Баланс фьючерсного счёта"""
        return await self._retry_request(self.exchange.fetch_balance, params={'type': 'future'})

    async def create_order(self, symbol: str, type: str, side: str, amount: float,
                           price: Optional[float] = None, params: Dict = {}) -> Dict:
        """Создание ордера (limit/market)"""
        return await self._retry_request(
            self.exchange.create_order,
            symbol, type, side, amount, price, params
        )

    async def cancel_order(self, id: str, symbol: str) -> Dict:
        """Отмена ордера"""
        return await self._retry_request(self.exchange.cancel_order, id, symbol)

    async def fetch_order(self, id: str, symbol: str) -> Dict:
        """Статус ордера"""
        return await self._retry_request(self.exchange.fetch_order, id, symbol)

    async def fetch_positions(self, symbol: Optional[str] = None) -> List[Dict]:
        """Все открытые позиции"""
        params = {'symbol': symbol} if symbol else {}
        return await self._retry_request(self.exchange.fetch_positions, params=params)

    async def close(self):
        """Закрытие соединения"""
        await self.exchange.close()
        logger.info("BinanceClient closed")

# Пример использования (для тестов)
if __name__ == "__main__":
    import asyncio

    async def test():
        config = load_config()
        client = BinanceClient(
            api_key=config["binance"]["api_key"],
            api_secret=config["binance"]["api_secret"],
            testnet=config["binance"]["testnet"]
        )

        try:
            ohlcv = await client.fetch_ohlcv("BTCUSDT", "1m", limit=10)
            print("Последняя свеча:", ohlcv[-1])
        finally:
            await client.close()

    asyncio.run(test())