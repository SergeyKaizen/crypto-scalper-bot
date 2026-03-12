"""
src/trading/virtual_trader.py

=== Основной принцип работы файла ===

Виртуальный трейдер — симулятор исполнения ордеров в бэктесте и shadow-торговле.
"""

import logging
import uuid
import time
from typing import Dict, Optional, List

from src.core.config import load_config
from src.utils.logger import setup_logger

setup_logger()
logger = logging.getLogger(__name__)

class VirtualTrader:
    def __init__(self, config=None, symbol=None):
        self.config = config or load_config()
        self.symbol = symbol or "VIRTUAL"
        self.taker_fee = 0.0004
        self.slippage_multiplier = self.config.get("slippage_multiplier", 0.5)
        self.positions: Dict[str, Dict] = {}
        self.balance = self.config.get("initial_deposit", 10000.0)

    def apply_slippage(self, price: float, direction: str, atr: float = None) -> float:
        if atr is None:
            atr = price * 0.001
        slippage = atr * self.slippage_multiplier
        if direction == 'L':
            return price + slippage
        else:
            return price - slippage

    def open_position(self, symbol: str, anomaly_type: str, prob: float, tp_sl: Dict):
        """Открытие виртуальной позиции"""
        direction = 'L' if prob > 0.5 else 'S'
        price = tp_sl.get('entry_price', 0.0) or 0.0
        atr = tp_sl.get('atr', price * 0.001)
        size = self.config.get("base_position_size", 0.001)

        entry_price = self.apply_slippage(price, direction, atr)
        order_id = str(uuid.uuid4())
        fee = entry_price * size * self.taker_fee

        position = {
            "id": order_id,
            "direction": direction,
            "entry_price": entry_price,
            "size": size,
            "timestamp": int(time.time() * 1000),
            "fee_open": fee,
            "closed": False,
            "exit_price": None,
            "fee_close": 0.0,
            "pnl": 0.0
        }

        self.positions[order_id] = position
        logger.debug(f"Virtual open {direction} {symbol}: size={size:.4f}")

    def close_position(self, order_id: str, exit_price: float = None, timestamp: int = None, atr: float = None) -> Optional[float]:
        if order_id not in self.positions:
            return None
        pos = self.positions[order_id]
        if pos["closed"]:
            return pos["pnl"]

        exit_price = exit_price or pos["entry_price"] * 1.01
        exit_price_adjusted = self.apply_slippage(exit_price, "S" if pos["direction"] == 'L' else "L", atr)
        fee_close = exit_price_adjusted * pos["size"] * self.taker_fee

        if pos["direction"] == 'L':
            pnl = (exit_price_adjusted - pos["entry_price"]) * pos["size"]
        else:
            pnl = (pos["entry_price"] - exit_price_adjusted) * pos["size"]

        net_pnl = pnl - pos["fee_open"] - fee_close
        pos["exit_price"] = exit_price_adjusted
        pos["fee_close"] = fee_close
        pos["pnl"] = net_pnl
        pos["closed"] = True
        pos["timestamp_close"] = timestamp or int(time.time() * 1000)

        self.balance += net_pnl
        logger.debug(f"Virtual close {order_id}: pnl {net_pnl:.2f}")
        return net_pnl

    def calculate_pnl(self, entry_price: float, exit_price: float, size: float, direction: str) -> float:
        if direction == 'L':
            return (exit_price - entry_price) * size
        else:
            return (entry_price - exit_price) * size

    def get_open_positions(self) -> List[Dict]:
        return [p for p in self.positions.values() if not p["closed"]]

    def get_balance(self) -> float:
        return self.balance