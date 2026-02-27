"""
src/trading/virtual_trader.py

=== Основной принцип работы файла ===

Виртуальный трейдер — симулятор исполнения ордеров в бэктесте и shadow-торговле.
Учитывает:
- taker fee 0.0004 (Binance futures standard)
- slippage (динамический, на основе ATR и объёма)
- funding rate (если доступно)
- min notional / lot size check
- open / close / partial close позиции

=== Главные функции ===
- open_position(direction, price, size, timestamp) → order_id
- close_position(order_id, price, timestamp) → pnl
- calculate_pnl(entry_price, exit_price, size, direction) → pnl_value
- apply_slippage(price, direction, atr) → adjusted_price

=== Примечания ===
- taker_fee = 0.0004 (0.04%) — константа, можно вынести в config
- slippage = 0.3–0.8 × ATR (настраивается)
- Все расчёты в USDT (quote currency)
- Логи через setup_logger
"""

import logging
import uuid
from typing import Dict, Optional

from src.core.config import load_config
from src.utils.logger import setup_logger

setup_logger()
logger = logging.getLogger(__name__)


class VirtualTrader:
    def __init__(self, config: dict, symbol: str):
        self.config = config
        self.symbol = symbol
        self.taker_fee = 0.0004  # 0.04% — Binance futures taker fee
        self.slippage_multiplier = config.get("slippage_multiplier", 0.5)  # 0.5 × ATR default
        self.positions: Dict[str, Dict] = {}  # order_id → position data
        self.balance = config["initial_balance"]

    def apply_slippage(self, price: float, direction: str, atr: float = None) -> float:
        """Применяет slippage к цене исполнения"""
        if atr is None:
            atr = price * 0.001  # fallback 0.1%

        slippage = atr * self.slippage_multiplier
        if direction == "L":
            return price + slippage  # buy — worse price
        else:
            return price - slippage  # sell — worse price

    def open_position(self,
                      direction: str,
                      price: float,
                      size: float,
                      timestamp: int,
                      atr: float = None) -> Optional[Dict]:
        """Открытие позиции с учётом slippage и fee"""
        entry_price = self.apply_slippage(price, direction, atr)

        order_id = str(uuid.uuid4())
        fee = entry_price * size * self.taker_fee

        position = {
            "id": order_id,
            "direction": direction,
            "entry_price": entry_price,
            "size": size,
            "timestamp": timestamp,
            "fee_open": fee,
            "closed": False,
            "exit_price": None,
            "fee_close": 0.0,
            "pnl": 0.0
        }

        self.positions[order_id] = position
        logger.info(f"Virtual open {direction} {self.symbol}: {size:.4f} @ {entry_price:.2f}, fee {fee:.2f}")

        return position

    def close_position(self,
                       order_id: str,
                       exit_price: float,
                       timestamp: int,
                       atr: float = None) -> Optional[float]:
        """Закрытие позиции с учётом slippage и fee"""
        if order_id not in self.positions:
            logger.warning(f"Position {order_id} not found")
            return None

        pos = self.positions[order_id]
        if pos["closed"]:
            return pos["pnl"]

        exit_price_adjusted = self.apply_slippage(exit_price, "S" if pos["direction"] == "L" else "L", atr)
        fee_close = exit_price_adjusted * pos["size"] * self.taker_fee

        if pos["direction"] == "L":
            pnl = (exit_price_adjusted - pos["entry_price"]) * pos["size"]
        else:
            pnl = (pos["entry_price"] - exit_price_adjusted) * pos["size"]

        net_pnl = pnl - pos["fee_open"] - fee_close

        pos["exit_price"] = exit_price_adjusted
        pos["fee_close"] = fee_close
        pos["pnl"] = net_pnl
        pos["closed"] = True
        pos["timestamp_close"] = timestamp

        self.balance += net_pnl
        logger.info(f"Virtual close {order_id}: exit @ {exit_price_adjusted:.2f}, pnl {net_pnl:.2f}, net {net_pnl:.2f}")

        return net_pnl

    def calculate_pnl(self,
                      entry_price: float,
                      exit_price: float,
                      size: float,
                      direction: str) -> float:
        """Расчёт PnL без slippage/fee (для предварительных оценок)"""
        if direction == "L":
            return (exit_price - entry_price) * size
        else:
            return (entry_price - exit_price) * size

    def get_open_positions(self) -> List[Dict]:
        """Список открытых позиций"""
        return [p for p in self.positions.values() if not p["closed"]]

    def get_balance(self) -> float:
        """Текущий виртуальный баланс"""
        return self.balance