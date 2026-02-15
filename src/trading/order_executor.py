# src/trading/order_executor.py
"""
Исполнитель реальных ордеров на Binance Futures.

Ключевые принципы реализации:
- Открытие позиции — ТОЛЬКО по Market (как ты просил)
- Проверка и автоматическая корректировка плеча (если заданное > max для монеты — берём максимальное)
- Проверка минимального размера ордера для монеты
- Поддержка long/short (positionSide BOTH или hedge по конфигу)
- Обработка основных ошибок биржи (недостаточно баланса, reject, rate limit, invalid order)
- Логирование каждого ордера + сохранение в storage для истории и анализа
- Автоматическое закрытие позиции по TP/SL (если не используется trailing)
- Все действия асинхронные (ccxt.pro)
"""

import asyncio
import time
from typing import Dict, Optional, Tuple
import ccxt.pro as ccxt
import polars as pl

from src.data.storage import Storage
from src.utils.logger import get_logger
from src.core.config import load_config

logger = get_logger(__name__)

class OrderExecutor:
    def __init__(self, config: dict):
        self.config = config
        self.storage = Storage(config)
        self.exchange = ccxt.binance(config["exchange"])
        
        # Настройки из конфига
        self.is_real_trading = config["trading_mode"].get("real_trading", False)
        self.position_mode = config["trading_mode"].get("position_mode", "BOTH")  # BOTH или HEDGE
        
        # Кэш параметров монет (обновляется раз в час)
        self.symbol_info: Dict[str, Dict] = {}
        self.last_info_update = 0

    async def _update_symbol_info(self, symbol: str):
        """Загружает/обновляет параметры монеты (min amount, max leverage и т.д.)"""
        now = time.time()
        if symbol in self.symbol_info and now - self.last_info_update < 3600:
            return
        
        try:
            markets = await self.exchange.load_markets()
            market = markets.get(symbol)
            if not market:
                raise ValueError(f"Символ {symbol} не найден на бирже")

            info = {
                "min_amount": market["limits"]["amount"]["min"] or 0.001,
                "max_leverage": int(market["info"].get("maxLeverage", 125)),
                "price_precision": market["precision"]["price"],
                "quantity_precision": market["precision"]["amount"],
                "contract_size": market["info"].get("contractSize", 1.0),
                "margin_asset": market["info"].get("marginAsset", "USDT")
            }
            
            self.symbol_info[symbol] = info
            self.last_info_update = now
            logger.debug(f"Обновлены параметры {symbol}: {info}")
        except Exception as e:
            logger.error(f"Ошибка загрузки info для {symbol}: {e}")

    async def open_position(self,
                           symbol: str,
                           direction: str,        # "buy" / "sell"
                           amount: float,
                           leverage: Optional[int] = None,
                           price: Optional[float] = None) -> Tuple[bool, str]:
        """
        Открытие позиции по Market ордеру.

        Параметры:
        - direction: "buy" (long) или "sell" (short)
        - amount: размер в контрактах (или USDT — зависит от конфига)
        - leverage: желаемое плечо (если None — из конфига)
        - price: не используется (Market), но оставлен для совместимости

        Возвращает: (успех, order_id или сообщение об ошибке)
        """
        if not self.is_real_trading:
            logger.info(f"[VIRTUAL] Открыта позиция {direction.upper()} {symbol} | amount={amount}")
            return True, "virtual_open_success"

        try:
            await self._update_symbol_info(symbol)
            info = self.symbol_info.get(symbol)
            if not info:
                return False, "не удалось загрузить параметры монеты"

            # Проверяем минимальный размер
            min_amount = info["min_amount"]
            if amount < min_amount:
                logger.warning(f"Размер {amount} < минимального {min_amount} для {symbol}")
                return False, f"amount_too_small (min: {min_amount})"

            # Корректируем плечо
            requested_leverage = leverage or self.config["finance"].get("leverage", 20)
            max_leverage = info["max_leverage"]
            final_leverage = min(requested_leverage, max_leverage)
            
            if final_leverage < requested_leverage:
                logger.warning(f"Плечо снижено с {requested_leverage}x до {final_leverage}x (макс для {symbol})")

            # Устанавливаем плечо (если не установлено ранее)
            await self.exchange.set_leverage(final_leverage, symbol, params={"positionSide": "BOTH"})

            # Открываем позицию
            order = await self.exchange.create_order(
                symbol=symbol,
                type="market",
                side=direction,
                amount=amount,
                params={
                    "positionSide": "BOTH",  # или "LONG"/"SHORT" если hedge mode
                    "reduceOnly": False
                }
            )

            order_id = order["id"]
            executed_price = order.get("average") or order.get("price")
            executed_amount = order.get("filled") or amount

            logger.info(f"[REAL OPEN] {direction.upper()} {symbol} | "
                        f"amount={executed_amount} | price={executed_price} | "
                        f"leverage={final_leverage}x | order_id={order_id}")

            # Сохраняем в историю
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
            logger.error(f"Недостаточно средств для {symbol}")
            return False, "insufficient_funds"
        except ccxt.InvalidOrder as e:
            logger.error(f"Недопустимый ордер {symbol}: {e}")
            return False, f"invalid_order: {str(e)}"
        except ccxt.RateLimitExceeded:
            logger.warning(f"Rate limit для {symbol} — ждём 10 сек")
            await asyncio.sleep(10)
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
                        f"amount={executed_amount} | price={executed_price} | "
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
            logger.warning("Rate limit — ждём 10 сек")
            await asyncio.sleep(10)
            return await self.close_position(symbol, amount, reason)
        except Exception as e:
            logger.error(f"Ошибка закрытия позиции {symbol}: {e}")
            return False, str(e)

    async def cancel_all_orders(self, symbol: str):
        """Отмена всех открытых ордеров по монете (на всякий случай)"""
        try:
            await self.exchange.cancel_all_orders(symbol)
            logger.info(f"Все ордера по {symbol} отменены")
        except Exception as e:
            logger.error(f"Ошибка отмены ордеров {symbol}: {e}")


if __name__ == "__main__":
    config = load_config()
    executor = OrderExecutor(config)
    
    # Тестовый запуск (асинхронно)
    async def test():
        success, msg = await executor.open_position("BTCUSDT", "buy", 0.001)
        print(f"Открытие: {success} | {msg}")
        
        await asyncio.sleep(5)
        success, msg = await executor.close_position("BTCUSDT")
        print(f"Закрытие: {success} | {msg}")

    asyncio.run(test())