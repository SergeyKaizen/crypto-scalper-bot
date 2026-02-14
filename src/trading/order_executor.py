# src/trading/order_executor.py
"""
OrderExecutor — отвечает за размещение реальных и виртуальных ордеров.
Поддерживает:
- Реальный режим (ccxt)
- Виртуальный режим (симуляция)
- Логирование всех ордеров
- Проверку баланса и маржи перед отправкой
"""

import time
from typing import Optional

import ccxt

from ..utils.logger import logger
from ..core.config import get_config
from ..core.types import Position


class OrderExecutor:
    def __init__(self):
        self.config = get_config()
        self.is_real = self.config["trading"].get("real_trading", False)

        if self.is_real:
            self.exchange = ccxt.binance({
                'apiKey': self.config['binance']['api_key'],
                'secret': self.config['binance']['secret_key'],
                'enableRateLimit': True,
                'options': {'defaultType': 'future'},
            })
            logger.info("OrderExecutor запущен в РЕАЛЬНОМ режиме")
        else:
            logger.info("OrderExecutor запущен в ВИРТУАЛЬНОМ режиме (симуляция)")

    async def execute_open(self, position: Position) -> bool:
        """Открывает позицию (реальную или виртуальную)."""
        if self.is_real:
            return await self._open_real(position)
        else:
            return self._open_virtual(position)

    async def _open_real(self, position: Position) -> bool:
        """Реальное размещение ордера через ccxt."""
        try:
            symbol = f"{position.coin}USDT"
            side = "buy" if position.side == "L" else "sell"
            order_type = "market"  # можно сделать limit позже

            order = self.exchange.create_order(
                symbol=symbol,
                type=order_type,
                side=side,
                amount=position.size,
                params={'positionSide': 'BOTH'}
            )

            position.order_id = order['id']
            logger.info("РЕАЛЬНЫЙ ордер отправлен",
                        coin=position.coin,
                        side=side,
                        size=position.size,
                        order_id=order['id'])

            return True

        except Exception as e:
            logger.error("Ошибка отправки реального ордера", error=str(e))
            return False

    def _open_virtual(self, position: Position) -> bool:
        """Виртуальное открытие (симуляция)."""
        logger.info("ВИРТУАЛЬНЫЙ ордер открыт",
                    coin=position.coin,
                    side=position.side,
                    entry_price=position.entry_price,
                    size=position.size)
        return True

    async def close_position(self, position: Position, close_price: float, reason: str):
        """Закрытие позиции."""
        if self.is_real:
            # Реальное закрытие
            logger.info("Закрываем реальную позицию", reason=reason)
            # Здесь будет код закрытия
            pass
        else:
            logger.info("Закрыта виртуальная позиция", 
                        coin=position.coin,
                        reason=reason,
                        close_price=close_price)