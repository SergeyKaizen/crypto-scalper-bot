"""
src/trading/order_executor.py

=== Основной принцип работы файла ===

Исполнитель ордеров на Binance Futures.

Ключевые особенности:
- place_order: market entry + установка TP (LIMIT reduceOnly) + SL (STOP_MARKET reduceOnly)
- Установка leverage перед открытием (по config)
- Полная обработка ошибок (InsufficientFunds, RateLimit, InvalidOrder и т.д.)
- cancel_order, get_order_status
- Логирование с quiet_streak, consensus_count и weight (если передан)
- Поддержка extra в position (quiet_streak, consensus_count, weight)

=== Главные функции ===
- place_order(position: dict) → order_id или None
- cancel_order(order_id: str)
- get_order_status(order_id: str)
- _set_leverage(symbol: str, leverage: int)

=== Примечания ===
- reduceOnly=True для TP/SL — позиция закрывается только частично/полностью
- Market entry — для скорости в интрадей
- Полностью соответствует ТЗ + последним изменениям (extra в log)
- Готов к интеграции в entry_manager и live_loop
- Логи через setup_logger
"""

import ccxt
import logging
from typing import Dict, Optional
import time  # FIX Фаза 1: отсутствовал → NameError в RateLimitExceeded

from src.core.config import load_config
from src.utils.logger import setup_logger

logger = setup_logger('order_executor', logging.INFO)

class OrderExecutor:
    def __init__(self):
        self.config = load_config()
        self.exchange = ccxt.binance({
            'apiKey': self.config['binance']['api_key'],
            'secret': self.config['binance']['api_secret'],
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future',
                'adjustForTimeDifference': True,
                'recvWindow': 10000,
            }
        })

    def place_order(self, position: Dict) -> Optional[str]:
        """
        Открытие позиции: market entry + TP LIMIT + SL STOP_MARKET
        """
        symbol = position['symbol']
        direction = position['direction']
        size = position['size']
        entry_price = position['entry_price']
        tp = position.get('tp')
        sl = position.get('sl')
        quiet_streak = position.get('quiet_streak', 0)
        consensus_count = position.get('consensus_count', 1)
        weight = position.get('weight', None)  # если передан из entry_manager

        try:
            # Установка leverage
            leverage = self.config['trading'].get('leverage', 20)
            self._set_leverage(symbol, leverage)

            # Market entry
            side = 'buy' if direction == 'long' else 'sell'
            order = self.exchange.create_market_order(
                symbol=symbol,
                side=side,
                amount=size
            )
            order_id = order['id']

            # TP LIMIT reduceOnly
            if tp:
                tp_side = 'sell' if direction == 'long' else 'buy'
                self.exchange.create_limit_order(
                    symbol=symbol,
                    side=tp_side,
                    amount=size,
                    price=tp,
                    params={'reduceOnly': True}
                )

            # SL STOP_MARKET reduceOnly
            if sl:
                sl_side = 'sell' if direction == 'long' else 'buy'
                self.exchange.create_order(
                    symbol=symbol,
                    type='STOP_MARKET',
                    side=sl_side,
                    amount=size,
                    params={
                        'stopPrice': sl,
                        'reduceOnly': True
                    }
                )

            # Лог с дополнительными параметрами
            log_msg = f"[ORDER] Открыта позиция {direction} {symbol} size={size:.4f} " \
                      f"entry={entry_price:.2f} TP={tp} SL={sl} " \
                      f"quiet_streak={quiet_streak} consensus={consensus_count}"
            if weight is not None:
                log_msg += f" weight={weight:.4f}"
            logger.info(log_msg + f" order_id={order_id}")

            return order_id

        except ccxt.InsufficientFunds as e:
            logger.error(f"Недостаточно средств для {symbol}: {e}")
        except ccxt.RateLimitExceeded as e:
            logger.error(f"Rate limit для {symbol}: {e}")
            time.sleep(10)
        except ccxt.InvalidOrder as e:
            logger.error(f"Неверный ордер для {symbol}: {e}")
        except Exception as e:
            logger.exception(f"Ошибка открытия позиции {symbol}: {e}")

        return None

    def cancel_order(self, order_id: str):
        """Отмена ордера"""
        try:
            self.exchange.cancel_order(order_id)
            logger.info(f"Ордер {order_id} отменён")
        except Exception as e:
            logger.error(f"Ошибка отмены ордера {order_id}: {e}")

    def get_order_status(self, order_id: str) -> Optional[str]:
        """Статус ордера"""
        try:
            order = self.exchange.fetch_order(order_id)
            return order['status']
        except Exception as e:
            logger.error(f"Ошибка получения статуса ордера {order_id}: {e}")
            return None

    def _set_leverage(self, symbol: str, leverage: int):
        """Установка плеча"""
        try:
            self.exchange.fapiPrivatePostLeverage({
                'symbol': symbol.replace('/', ''),
                'leverage': leverage
            })
            logger.debug(f"Установлено плечо {leverage}x для {symbol}")
        except Exception as e:
            logger.warning(f"Ошибка установки плеча для {symbol}: {e}")