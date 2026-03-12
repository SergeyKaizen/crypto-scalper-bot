"""
src/trading/order_executor.py

=== Основной принцип работы файла ===

OrderExecutor отвечает за выставление и управление реальными ордерами на Binance Futures.
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
        self.max_retries = 5
        self.retry_delay_base = 1

    def _retry_on_rate_limit(func):
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
    def place_order(self, position: Dict):
        """Выставление ордера (принимает position dict)"""
        symbol = position['symbol']
        direction = position['direction']
        size = position['size']
        side = "BUY" if direction == 'L' else "SELL"

        # Округление размера под stepSize/minQty (подтверждено)
        size = self._round_to_step_size(symbol, size)

        params = {
            "symbol": symbol,
            "side": side,
            "type": "MARKET",
            "quantity": size,
        }

        try:
            order = self.client.futures_create_order(**params)
            logger.info(f"Ордер выставлен: {order}")
            return order.get('orderId')
        except Exception as e:
            logger.error(f"Ошибка place_order: {e}")
            raise

    def _round_to_step_size(self, symbol: str, size: float) -> float:
        # Получаем stepSize от Binance
        info = self.client.futures_exchange_info()
        for s in info['symbols']:
            if s['symbol'] == symbol:
                step_size = float(s['filters'][1]['stepSize'])
                min_qty = float(s['filters'][1]['minQty'])
                size = round(size / step_size) * step_size
                if size < min_qty:
                    return 0.0
                return size
        return size

    @_retry_on_rate_limit
    def close_position(self, position: Dict):
        """Закрытие позиции market-ордером"""
        symbol = position['symbol']
        direction = position['direction']
        size = position['size']
        opposite_side = "SELL" if direction == 'L' else "BUY"

        params = {
            "symbol": symbol,
            "side": opposite_side,
            "type": "MARKET",
            "quantity": size,
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