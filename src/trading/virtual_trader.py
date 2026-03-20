"""
src/trading/virtual_trader.py

=== Основной принцип работы файла ===

Виртуальный трейдер — симулятор исполнения ордеров в бэктесте и shadow-торговле.

После аудита (Phase 4):
- УДАЛЁН собственный dict позиций (self.positions)
- Теперь ТОЛЬКО PositionManager является single source of truth
- VirtualTrader — тонкий слой: slippage + fees + расчёт pnl
- open/close теперь делегируют хранение в PositionManager
"""

import logging
import uuid
import time
from typing import Dict, Optional

from src.core.config import load_config
from src.utils.logger import setup_logger

setup_logger()
logger = logging.getLogger(__name__)

class VirtualTrader:
    def __init__(self):
        self.config = load_config()
        self.taker_fee = 0.0004
        self.slippage_multiplier = self.config.get("slippage_multiplier", 0.5)

        # Теперь НЕ храним позиции сами — только PositionManager
        self.position_manager = None  # будет установлен из PositionManager

    def set_position_manager(self, manager):
        """Инжекция PositionManager (вызывается в PositionManager.__init__)"""
        self.position_manager = manager

    def apply_slippage(self, price: float, direction: str, atr: float = None) -> float:
        if atr is None:
            atr = price * 0.001
        slippage = atr * self.slippage_multiplier
        if direction == 'L':
            return price + slippage
        else:
            return price - slippage

    def open_position(self, pos_data: Dict):
        """
        Симуляция открытия (возвращает adjusted entry_price + order_id)
        Реальное хранение — в PositionManager
        """
        symbol = pos_data['symbol']
        direction = pos_data['direction']
        price = pos_data.get('entry_price', 0.0)
        atr = pos_data.get('atr', price * 0.001)
        size = pos_data.get('size', 0.001)

        entry_price = self.apply_slippage(price, direction, atr)
        order_id = str(uuid.uuid4())
        fee = entry_price * size * self.taker_fee

        logger.debug(f"Virtual open {direction} {symbol}: size={size:.4f}, entry={entry_price:.4f}")

        # Возвращаем данные для PositionManager
        return {
            "order_id": order_id,
            "entry_price": entry_price,
            "fee_open": fee
        }

    def close_position(self, pos_id: str, exit_price: float = None, atr: float = None) -> Optional[float]:
        """
        Симуляция закрытия — возвращает net_pnl
        Реальное обновление позиции — в PositionManager
        """
        # Получаем данные позиции из PositionManager
        if not self.position_manager:
            logger.error("PositionManager not set in VirtualTrader")
            return None

        pos_info = self.position_manager.positions.get(pos_id)
        if not pos_info or pos_info['state'] != "OPEN":
            return None

        pos_data = pos_info['data']
        direction = pos_data['direction']
        entry_price = pos_data['entry_price']
        size = pos_data['size']

        exit_price = exit_price or pos_data.get('tp') or pos_data.get('sl') or entry_price * 1.01
        exit_price_adjusted = self.apply_slippage(exit_price, "S" if direction == 'L' else "L", atr)

        fee_close = exit_price_adjusted * size * self.taker_fee

        if direction == 'L':
            pnl = (exit_price_adjusted - entry_price) * size
        else:
            pnl = (entry_price - exit_price_adjusted) * size

        net_pnl = pnl - pos_data.get('fee_open', 0) - fee_close

        logger.debug(f"Virtual close {pos_id}: pnl {net_pnl:.2f}")
        return net_pnl

    def calculate_pnl(self, entry_price: float, exit_price: float, size: float, direction: str) -> float:
        """Чистый расчёт pnl (без комиссий)"""
        if direction == 'L':
            return (exit_price - entry_price) * size
        else:
            return (entry_price - exit_price) * size