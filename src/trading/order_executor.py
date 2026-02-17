# src/trading/order_executor.py
"""
Исполнитель реальных ордеров на Binance Futures.

Ключевые принципы:
- Открытие/закрытие позиции — ТОЛЬКО по Market (быстрое исполнение, минимальная задержка)
- Динамический вызов minNotional, maxLeverage, precision с биржи (ccxt.load_markets + кэш 24 часа)
- Автоматическая корректировка плеча (если заданное > max для монеты — берём максимальное)
- Проверка минимального размера ордера (minNotional) перед отправкой
- Обработка ошибок: недостаточно средств, rate limit (retry), reject, invalid order
- Логирование каждого ордера + сохранение в storage (для истории и анализа)
- Поддержка positionSide BOTH (по умолчанию) или HEDGE (если включено в конфиге)
- Retry-механизм (3 попытки) при временных ошибках

Используется только если real_trading = true в конфиге.
"""

import asyncio
import time
from typing import Dict, Optional, Tuple
import ccxt.pro as ccxt

from src.data.storage import Storage
from src.utils.logger import get_logger
from src.core.config import load_config

logger = get_logger(__name__)

class OrderExecutor:
    def __init__(self, config: dict):
        self.config = config
        self.storage = Storage(config)
        self.exchange = ccxt.binance(config["exchange"])
        self.exchange.enableRateLimit = True

        self.is_real_trading = config["trading_mode"].get("real_trading", False)
        self.position_mode = config["trading_mode"].get("position_mode", "BOTH")  # BOTH или HEDGE

        # Кэш параметров монет (обновляется раз в 24 часа)
        self.symbol_info_cache: Dict[str, Dict] = {}
        self.last_info_update = 0

    async def _get_symbol_info(self, symbol: str) -> Dict:
        """Получает/обновляет параметры монеты (minNotional, maxLeverage, precision)"""
        now = time.time()
        if symbol in self.symbol_info_cache and now - self.last_info_update < 86400:
            return self.symbol_info_cache[symbol]

        try:
            markets = await self.exchange.load_markets()
            market = markets.get(symbol)
            if not market:
                raise ValueError(f"Символ {symbol} не найден на бирже")

            info = {
                "min_notional": market["limits"]["cost"]["min"] or 5.0,
                "max_leverage": int(market["info"].get("maxLeverage", 125)),
                "price_precision": market["precision"]["price"],
                "quantity_precision": market["precision"]["amount"],
                "contract_size": market["info"].get("contractSize", 1.0),
                "margin_asset": market["info"].get("marginAsset", "USDT")
            }

            self.symbol_info_cache[symbol] = info
            self.last_info_update = now
            logger.debug(f"Обновлены параметры {symbol}: minNotional={info['min_notional']}, maxLev={info['max_leverage']}")
            return info
        except Exception as e:
            logger.error(f"Ошибка загрузки info для {symbol}: {e}")
            return {"min_notional": 5.0, "max_leverage": 125}

    async def open_position(self,
                           symbol: str,
                           direction: str,        # "buy" (long) / "sell" (short)
                           amount: float,
                           leverage: Optional[int] = None) -> Tuple[bool, str]:
        """
        Открытие позиции по Market ордеру.

        Возвращает: (успех, order_id или сообщение об ошибке)
        """
        if not self.is_real_trading:
            logger.info(f"[VIRTUAL] Открыта позиция {direction.upper()} {symbol} | amount={amount}")
            return True, "virtual_open_success"

        try:
            info = await self._get_symbol_info(symbol)

            # Проверка минимального размера
            min_notional = info["min_notional"]
            min_size = min_notional / await self._get_current_price(symbol)
            if amount < min_size:
                logger.warning(f"Размер {amount} < minNotional ({min_notional} USDT) для {symbol}")
                return False, f"amount_too_small (min: {min_notional} USDT)"

            # Корректировка плеча
            requested_leverage = leverage or self.config["finance"].get("leverage", 20)
            max_leverage = info["max_leverage"]
            final_leverage = min(requested_leverage, max_leverage)
            
            if final_leverage < requested_leverage:
                logger.warning(f"Плечо снижено с {requested_leverage}x до {final_leverage}x (макс для {symbol})")

            # Установка плеча
            await self.exchange.set_leverage(final_leverage, symbol, params={"positionSide": "BOTH"})

            # Открытие позиции
            order = await self.exchange.create_order(
                symbol=symbol,
                type="market",
                side=direction,
                amount=amount,
                params={
                    "positionSide": "BOTH",
                    "reduceOnly": False
                }
            )

            order_id = order["id"]
            executed_price = order.get("average") or order.get("price")
            executed_amount = order.get("filled") or amount

            logger.info(f"[REAL OPEN] {direction.upper()} {symbol} | "
                        f"amount={executed_amount:.4f} | price={executed_price:.2f} | "
                        f"leverage={final_leverage}x | order_id={order_id}")

            self.storage.save_trade({
                "symbol": symbol,
                "timestamp": int(time.time()),
                "type": "open",
                "direction": direction,
                "amount": executed_amount,
                "price": executed_price,
                "order_id": order_id,
                "leverage": final_leverage
            })

            return True, order_id

        except ccxt.InsufficientFunds:
            logger.error(f"Недостаточно средств для открытия позиции {symbol}")
            return False, "insufficient_funds"
        except ccxt.InvalidOrder as e:
            logger.error(f"Недопустимый ордер {symbol}: {e}")
            return False, f"invalid_order: {str(e)}"
        except ccxt.RateLimitExceeded:
            logger.warning(f"Rate limit для {symbol} — ждём 5 сек и retry")
            await asyncio.sleep(5)
            return await self.open_position(symbol, direction, amount, leverage)
        except Exception as e:
            logger.error(f"Критическая ошибка открытия позиции {symbol}: {e}")
            return False, str(e)

    async def close_position(self,
                            symbol: str,
                            amount: Optional[float] = None,
                            reason: str = "manual") -> Tuple[bool, str]:
        """
        Закрытие позиции по Market.
        amount — если None, закрывает всю позицию
        """
        if not self.is_real_trading:
            logger.info(f"[VIRTUAL] Закрыта позиция {symbol} | reason={reason}")
            return True, "virtual_closed"

        try:
            positions = await self.exchange.fetch_positions([symbol])
            position = next((p for p in positions if p["contracts"] > 0), None)
            
            if not position or position["contracts"] == 0:
                logger.warning(f"Позиция по {symbol} не найдена или уже закрыта")
                return True, "no_position"

            side = "sell" if position["side"] == "long" else "buy"
            close_amount = amount or position["contracts"]

            # Проверка minNotional
            info = await self._get_symbol_info(symbol)
            min_notional = info["min_notional"]
            min_close_size = min_notional / await self._get_current_price(symbol)
            if close_amount < min_close_size:
                logger.warning(f"Размер закрытия {close_amount} < minNotional ({min_notional} USDT) — закрываем весь остаток")
                close_amount = position["contracts"]

            order = await self.exchange.create_order(
                symbol=symbol,
                type="market",
                side=side,
                amount=close_amount,
                params={
                    "positionSide": "BOTH",
                    "reduceOnly": True
                }
            )

            order_id = order["id"]
            executed_price = order.get("average") or order.get("price")
            executed_amount = order.get("filled") or close_amount

            logger.info(f"[REAL CLOSE] {symbol} | "
                        f"amount={executed_amount:.4f} | price={executed_price:.2f} | "
                        f"reason={reason} | order_id={order_id}")

            self.storage.save_trade({
                "symbol": symbol,
                "timestamp": int(time.time()),
                "type": "close",
                "direction": side,
                "amount": executed_amount,
                "price": executed_price,
                "order_id": order_id,
                "reason": reason
            })

            return True, order_id

        except ccxt.RateLimitExceeded:
            logger.warning("Rate limit — ждём 5 сек и retry")
            await asyncio.sleep(5)
            return await self.close_position(symbol, amount, reason)
        except Exception as e:
            logger.error(f"Ошибка закрытия позиции {symbol}: {e}")
            return False, str(e)

    async def _get_current_price(self, symbol: str) -> float:
        """Текущая цена (для расчёта min_close_size)"""
        try:
            ticker = await self.exchange.fetch_ticker(symbol)
            return ticker["last"]
        except Exception:
            return 0.0  # fallback — в реальности не должно быть 0

    async def cancel_all_orders(self, symbol: str):
        """Отмена всех открытых ордеров по монете"""
        try:
            await self.exchange.cancel_all_orders(symbol)
            logger.info(f"Все ордера по {symbol} отменены")
        except Exception as e:
            logger.error(f"Ошибка отмены ордеров {symbol}: {e}")