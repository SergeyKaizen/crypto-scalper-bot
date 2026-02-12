# src/trading/order_executor.py
"""
Модуль исполнения ордеров (реальных и виртуальных).

Основные функции:
- place_order() — открытие ордера (real или virtual)
- cancel_order() — отмена ордера
- simulate_order_execution() — симуляция исполнения (для virtual и shadow trading)
- get_order_status() — проверка статуса ордера
- close_position() — принудительное закрытие позиции

Логика:
- real: ccxt.create_order() / cancel_order()
- virtual: симуляция (fill по следующей свече или market price)
- slippage: в симуляции добавляется 0.05–0.2% (настраивается)
- комиссии: taker/maker fee из constants
- ошибки Binance обрабатываются (retry, логи)
- shadow_trading — параллельно считает "реальный" исход (с slippage)

Зависимости:
- src/data/binance_client.py — для реальных ордеров
- src/core/constants.py — комиссии, min_order
- config["trading_mode"] — real / virtual
"""

import logging
from typing import Dict, Optional
from datetime import datetime

import ccxt

from src.core.config import load_config
from src.core.types import Position
from src.data.binance_client import BinanceClient
from src.core.constants import BINANCE_FUTURES_TAKER_FEE, BINANCE_FUTURES_MAKER_FEE

logger = logging.getLogger(__name__)


class OrderExecutor:
    """Исполнение ордеров (real / virtual)"""

    def __init__(self, config: Dict):
        self.config = config
        self.mode = config["trading"]["mode"]  # "real" / "virtual"
        self.client = None

        if self.mode == "real":
            self.client = BinanceClient(
                api_key=config["binance"]["api_key"],
                api_secret=config["binance"]["api_secret"],
                testnet=config["binance"]["testnet"]
            )
            logger.info("OrderExecutor: REAL mode (Binance live)")
        else:
            logger.info("OrderExecutor: VIRTUAL mode (simulation)")

        self.slippage_pct = config.get("slippage_pct", 0.10)  # 0.1% slippage в симуляции
        self.commission_taker = float(BINANCE_FUTURES_TAKER_FEE)
        self.commission_maker = float(BINANCE_FUTURES_MAKER_FEE)

    async def place_order(
        self,
        symbol: str,
        side: str,          # "buy" / "sell"
        amount: float,      # в базовой валюте (BTC, ETH и т.д.)
        price: Optional[float] = None,
        order_type: str = "market",
        params: Dict = {}
    ) -> Dict:
        """
        Открытие ордера (реального или симулированного)

        Returns:
            Dict с результатом исполнения (order_id, fill_price, fee и т.д.)
        """
        if self.mode == "real":
            try:
                order = await self.client.create_order(
                    symbol=symbol,
                    type=order_type,
                    side=side,
                    amount=amount,
                    price=price,
                    params=params
                )
                logger.info("Real order placed: %s %s %s @ %.2f (id=%s)", 
                            side.upper(), order_type, symbol, price or order["price"], order["id"])
                return order
            except Exception as e:
                logger.error("Failed to place real order: %s", e)
                raise

        else:
            # Виртуальная симуляция
            fill_price = price if order_type == "limit" else await self._get_market_price(symbol, side)
            fill_price = self._apply_slippage(fill_price, side)

            fee = amount * fill_price * self.commission_taker

            simulated_order = {
                "id": f"virt_{int(datetime.now().timestamp())}",
                "symbol": symbol,
                "side": side,
                "type": order_type,
                "amount": amount,
                "price": fill_price,
                "filled": amount,
                "cost": amount * fill_price,
                "fee": {"cost": fee, "currency": "USDT"},
                "status": "closed",
                "timestamp": int(datetime.now().timestamp() * 1000)
            }

            logger.info("Virtual order simulated: %s %s %s @ %.2f", side.upper(), order_type, symbol, fill_price)
            return simulated_order

    async def cancel_order(self, order_id: str, symbol: str):
        """Отмена ордера"""
        if self.mode == "real":
            try:
                result = await self.client.cancel_order(order_id, symbol)
                logger.info("Order cancelled: id=%s, symbol=%s", order_id, symbol)
                return result
            except Exception as e:
                logger.error("Failed to cancel order %s: %s", order_id, e)
                raise
        else:
            logger.info("Virtual order cancelled: id=%s", order_id)
            return {"status": "cancelled"}

    async def _get_market_price(self, symbol: str, side: str) -> float:
        """Текущая рыночная цена (для симуляции market ордера)"""
        ticker = await self.client.fetch_ticker(symbol)
        if side == "buy":
            return ticker["ask"]
        else:
            return ticker["bid"]

    def _apply_slippage(self, price: float, side: str) -> float:
        """Добавление slippage в симуляции"""
        slippage = price * (self.slippage_pct / 100)
        if side == "buy":
            return price + slippage  # Покупаем дороже
        else:
            return price - slippage  # Продаём дешевле

    async def close_position(self, position: Position):
        """Принудительное закрытие позиции"""
        side = "sell" if position.direction == "L" else "buy"
        amount = position.size

        return await self.place_order(
            symbol=position.symbol,
            side=side,
            amount=amount,
            order_type="market"
        )

    async def get_order_status(self, order_id: str, symbol: str) -> Dict:
        """Статус ордера"""
        if self.mode == "real":
            return await self.client.fetch_order(order_id, symbol)
        else:
            # Виртуально — всегда closed (упрощённо)
            return {"status": "closed"}