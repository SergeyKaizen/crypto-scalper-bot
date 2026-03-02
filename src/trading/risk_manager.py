"""
src/trading/risk_manager.py

=== Основной принцип работы файла ===

RiskManager — модуль управления рисками.

Отвечает за:
- расчёт размера позиции по risk_pct и расстоянию до SL
- проверку дневного лимита убытка
- проверку максимального количества открытых позиций
- контроль leverage (max 50x)
- обновление депозита после закрытия позиции
"""

import logging
from src.core.config import load_config
from src.utils.logger import setup_logger

logger = setup_logger("risk_manager", logging.INFO)

class RiskManager:
    def __init__(self, config=None):
        self.config = config or load_config()
        self.deposit = self.config.get("initial_deposit", 10000.0)
        self.risk_pct = self.config["trading"].get("risk_pct", 0.01)
        self.daily_loss_limit = self.config["trading"].get("daily_loss_limit", 0.05)
        self.max_open_positions = self.config["trading"].get("max_open_positions", 3)
        self.max_leverage = self.config["trading"].get("max_leverage", 50)
        self.current_daily_loss = 0.0
        self.open_positions_count = 0

    def calculate_position_size(self, symbol: str, entry_price: float, sl_price: float) -> float:
        if entry_price <= 0 or sl_price <= 0:
            logger.warning(f"Некорректные цены для {symbol}: entry={entry_price}, sl={sl_price}")
            return 0.0

        risk_amount = self.deposit * self.risk_pct
        price_diff = abs(entry_price - sl_price)

        if price_diff == 0:
            logger.warning(f"SL = entry для {symbol} — размер позиции = 0")
            return 0.0

        size = risk_amount / price_diff

        min_size = self.config["trading"].get("min_position_size", 0.001)
        size = max(min_size, size)

        max_size_by_leverage = (self.deposit * self.max_leverage) / entry_price
        size = min(size, max_size_by_leverage)

        logger.debug(f"Размер позиции для {symbol}: {size:.6f} (risk {self.risk_pct*100}%, расстояние до SL {price_diff:.2f})")

        return size

    def can_open_new_position(self) -> bool:
        # FIX Фаза 7: реальная проверка daily_loss_limit
        if self.current_daily_loss <= -self.deposit * self.daily_loss_limit:
            logger.warning(f"Достигнут дневной лимит убытка — новые позиции запрещены")
            return False

        if self.open_positions_count >= self.max_open_positions:
            logger.info(f"Достигнут лимит открытых позиций: {self.open_positions_count}/{self.max_open_positions}")
            return False
        return True

    def update_after_open(self):
        self.open_positions_count += 1

    def update_after_close(self):
        self.open_positions_count = max(0, self.open_positions_count - 1)

    def update_deposit(self, pnl: float):
        self.deposit += pnl
        self.current_daily_loss += pnl if pnl < 0 else 0

    def reset_daily_loss(self):
        self.current_daily_loss = 0.0
        logger.info("Дневной лимит убытка сброшен")