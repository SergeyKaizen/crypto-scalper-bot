"""
src/trading/order_executor.py

=== Основной принцип работы файла ===

OrderExecutor отвечает за выставление и управление реальными ордерами на Binance Futures.

Ключевые задачи:
- place_order — выставление ордера (market/limit, long/short)
- close_position — закрытие позиции (market)
- cancel_all_open_orders — отмена всех открытых ордеров
- get_position_info — получение информации о позиции
- get_open_orders — получение списка открытых ордеров

=== Примечания ===
- Все методы используют BinanceClient
- Поддержка retry при rate-limit и сетевых ошибках
- Логирование всех действий и ошибок

FIX Фаза 14:
- Добавлена обработка rate-limit (429) и других ошибок Binance
- Exponential backoff + retry с максимальным количеством попыток
- Логирование всех попыток и ошибок
"""

import time
import logging
from binance.error import ClientError
from src.core.config import load_config
from src.data.binance_client import BinanceClient
from src.utils.logger import setup_logger

logger = setup_logger("order_executor", logging.INFO)

class OrderExecutor:
    def __init__(self):
        self.config = load_config()
        self.client = BinanceClient()

        # FIX Фаза 14: параметры retry (временные, финальные в конфиге Фаза 17)
        self.max_retries = 5
        self.retry_delay_base = 1  # секунды, будет умножаться на 2^n

    def _retry_on_rate_limit(func):
        """Декоратор для retry при rate-limit и других ошибках"""
        def wrapper(self, *args, **kwargs):
            retries = 0
            while retries < self.max_retries:
                try:
                    return func(self, *args, **kwargs)
                except ClientError as e:
                    if e.error_code == -1003 or "rate limit" in str(e).lower() or e.status_code == 429:
                        delay = self.retry_delay_base * (2 ** retries)
                        logger.warning(f"Rate-limit ошибка в {func.__name__}. Ждём {delay} сек, попытка {retries+1}/{self.max_retries}")
                        time.sleep(delay)
                        retries += 1
                    else:
                        logger.error(f"Необрабатываемая ошибка Binance в {func.__name__}: {e}")
                        raise
                except Exception as e:
                    logger.error(f"Общая ошибка в {func.__name__}: {e}")
                    raise
            raise Exception(f"Превышено количество попыток при rate-limit в {func.__name__}")
        return wrapper

    @_retry_on_rate_limit
    def place_order(self, symbol: str, side: str, type: str, quantity: float, price: float = None):
        """Выставление ордера"""
        params = {
            "symbol": symbol,
            "side": side.upper(),
            "type": type.upper(),
            "quantity": quantity,
        }
        if price is not None:
            params["price"] = price

        try:
            order = self.client.futures_create_order(**params)
            logger.info(f"Ордер выставлен: {order}")
            return order
        except Exception as e:
            logger.error(f"Ошибка place_order: {e}")
            raise

    @_retry_on_rate_limit
    def close_position(self, symbol: str, side: str, quantity: float):
        """Закрытие позиции market-ордером"""
        opposite_side = "SELL" if side.upper() == "BUY" else "BUY"
        params = {
            "symbol": symbol,
            "side": opposite_side,
            "type": "MARKET",
            "quantity": quantity,
            "reduceOnly": True,
        }

        try:
            order = self.client.futures_create_order(**params)
            logger.info(f"Позиция закрыта: {order}")
            return order
        except Exception as e:
            logger.error(f"Ошибка close_position: {e}")
            raise

    @_retry_on_rate_limit
    def cancel_all_open_orders(self, symbol: str = None):
        """Отмена всех открытых ордеров (или по символу)"""
        params = {}
        if symbol:
            params["symbol"] = symbol

        try:
            result = self.client.futures_cancel_all_open_orders(**params)
            logger.info(f"Все открытые ордера отменены: {result}")
            return result
        except Exception as e:
            logger.error(f"Ошибка cancel_all_open_orders: {e}")
            raise

    @_retry_on_rate_limit
    def get_position_info(self, symbol: str):
        """Получение информации о позиции"""
        try:
            positions = self.client.futures_position_information(symbol=symbol)
            return positions
        except Exception as e:
            logger.error(f"Ошибка get_position_info: {e}")
            raise

    @_retry_on_rate_limit
    def get_open_orders(self, symbol: str = None):
        """Получение списка открытых ордеров"""
        params = {}
        if symbol:
            params["symbol"] = symbol

        try:
            orders = self.client.futures_get_open_orders(**params)
            return orders
        except Exception as e:
            logger.error(f"Ошибка get_open_orders: {e}")
            raise