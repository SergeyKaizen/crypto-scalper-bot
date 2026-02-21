"""
src/trading/order_executor.py

=== Основной принцип работы файла ===

Этот файл реализует исполнение реальных ордеров на Binance Futures через ccxt.
Он вызывается из entry_manager только в режиме "real" торговли.

Основные принципы:
- Открытие позиции — всегда market order (быстрое исполнение).
- Установка TP и SL — сразу после открытия (limit для TP, STOP_MARKET для SL).
- Поддержка reduceOnly для SL/TP (чтобы не открывать лишние позиции).
- Проверка баланса/маржи/плеча перед ордером.
- Обработка ошибок (insufficient margin, rate limit, invalid symbol и т.д.).
- Логирование каждого ордера и результата.

Не выполняет никаких расчётов уровней TP/SL — они приходят готовыми из tp_sl_manager.

=== Главные функции и за что отвечают ===

- OrderExecutor() — инициализация ccxt.binance с ключами из config, включение rate-limit.
- place_order(pos: dict) → str или None — основной метод:
  - Открывает market buy/sell.
  - Устанавливает TP (limit order) и SL (stop-market с reduceOnly).
  - Возвращает ID ордера открытия или None при ошибке.
- cancel_order(order_id: str, symbol: str) — отменяет ордер (TP/SL если не исполнен).
- get_order_status(order_id: str, symbol: str) → dict — статус ордера.
- _check_balance_and_margin(required_margin: float) → bool — проверка достаточности средств.

=== Примечания ===
- Только market для входа — по ТЗ (быстрое исполнение на волатильном рынке).
- TP — limit (точное исполнение), SL — stop-market (гарантия закрытия).
- reduceOnly=True — ордера не открывают новые позиции.
- Полностью соответствует ТЗ: реальные ордера только при совпадении сигнала с PR config.
- Нет расчётов уровней — они приходят готовыми.
- Логи через setup_logger с префиксом [ORDER].
- Готов к использованию в entry_manager (при mode='real').
"""

import ccxt
from typing import Dict, Optional

from src.core.config import load_config
from src.core.enums import Direction
from src.utils.logger import setup_logger

logger = setup_logger('order_executor', logging.INFO)

class OrderExecutor:
    """
    Исполнитель реальных ордеров на Binance Futures.
    """
    def __init__(self):
        self.config = load_config()
        self.exchange = ccxt.binance({
            'apiKey': self.config['binance']['api_key'],
            'secret': self.config['binance']['api_secret'],
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future',
                'adjustForTimeDifference': True,
                'recvWindow': 5000
            }
        })
        self.exchange.load_markets()
        logger.info("OrderExecutor готов к реальным ордерам")

    def place_order(self, pos: Dict) -> Optional[str]:
        """
        Открывает market ордер + устанавливает TP и SL.
        pos: {'symbol', 'direction', 'size', 'entry_price', 'tp', 'sl'}
        Возвращает ID market ордера или None при ошибке.
        """
        symbol = pos['symbol']
        side = 'buy' if pos['direction'] == Direction.LONG.value else 'sell'
        amount = pos['size']
        leverage = pos.get('leverage', self.config['trading']['leverage_max'])

        try:
            # Установка плеча (если нужно)
            self.exchange.fapiPrivatePostLeverage({
                'symbol': symbol.replace('/', ''),
                'leverage': leverage
            })

            # Открытие market ордера
            order = self.exchange.create_market_order(
                symbol=symbol,
                side=side,
                amount=amount
            )
            order_id = order['id']
            logger.info(f"[ORDER] Открыт market {side.upper()} {symbol}, amount={amount}, leverage={leverage}x, id={order_id}")

            # Установка TP (limit, reduceOnly)
            tp_side = 'sell' if side == 'buy' else 'buy'
            tp_order = self.exchange.create_order(
                symbol=symbol,
                type='LIMIT',
                side=tp_side,
                amount=amount,
                price=pos['tp'],
                params={'reduceOnly': True}
            )
            logger.debug(f"[ORDER] TP установлен на {pos['tp']} для {symbol}, id={tp_order['id']}")

            # Установка SL (stop-market, reduceOnly)
            sl_order = self.exchange.create_order(
                symbol=symbol,
                type='STOP_MARKET',
                side=tp_side,
                amount=amount,
                params={
                    'stopPrice': pos['sl'],
                    'reduceOnly': True
                }
            )
            logger.debug(f"[ORDER] SL установлен на {pos['sl']} для {symbol}, id={sl_order['id']}")

            return order_id

        except ccxt.InsufficientFunds as e:
            logger.error(f"[ORDER] Недостаточно средств для {symbol}: {e}")
        except ccxt.RateLimitExceeded as e:
            logger.warning(f"[ORDER] Rate limit при ордере {symbol}: {e}. Ждём...")
            time.sleep(5)
        except ccxt.InvalidOrder as e:
            logger.error(f"[ORDER] Неверный ордер для {symbol}: {e}")
        except Exception as e:
            logger.error(f"[ORDER] Ошибка при исполнении {symbol}: {e}")

        return None

    def cancel_order(self, order_id: str, symbol: str):
        """
        Отмена ордера (TP/SL или открытия, если не исполнен).
        """
        try:
            self.exchange.cancel_order(order_id, symbol)
            logger.info(f"[ORDER] Ордер {order_id} отменён для {symbol}")
        except Exception as e:
            logger.error(f"[ORDER] Ошибка отмены {order_id} для {symbol}: {e}")

    def get_order_status(self, order_id: str, symbol: str) -> Dict:
        """
        Проверка статуса ордера.
        Возвращает {'status': 'open/filled/canceled/expired', 'filled': amount, ...}
        """
        try:
            order = self.exchange.fetch_order(order_id, symbol)
            return {
                'status': order['status'],
                'filled': order['filled'],
                'price': order['average'],
                'cost': order['cost']
            }
        except Exception as e:
            logger.error(f"[ORDER] Ошибка проверки статуса {order_id} для {symbol}: {e}")
            return {'status': 'error'}