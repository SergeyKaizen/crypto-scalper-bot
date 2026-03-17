"""
src/trading/risk_manager.py

=== Основной принцип работы файла ===

RiskManager — модуль управления рисками.
Теперь поддерживает два режима расчёта риска: fixed и auto (по ТЗ).
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

        # Новый режим расчёта риска (утверждённый)
        self.risk_mode = self.config["trading"].get("risk_mode", "auto")

    def calculate_position_size(self, symbol: str, entry_price: float, tp_price: float, sl_price: float) -> float:
        """
        Расчёт размера позиции с поддержкой двух режимов:
        - "fixed" — фиксированный % от депозита (risk_pct)
        - "auto" — риск = (длина TP / длина SL) по формуле ТЗ
        """
        if entry_price <= 0 or sl_price <= 0 or tp_price <= 0:
            logger.warning(f"Некорректные цены для {symbol}")
            return 0.0

        sl_pct = abs(entry_price - sl_price) / entry_price
        if sl_pct == 0:
            logger.warning(f"SL = entry для {symbol} — размер позиции = 0")
            return 0.0

        tp_pct = abs(tp_price - entry_price) / entry_price

        if self.risk_mode == "auto":
            # Авто-режим по ТЗ: риск = RR = tp_pct / sl_pct
            risk_usdt = self.deposit * (tp_pct / sl_pct)
        else:
            # Fixed-режим
            risk_usdt = self.deposit * self.risk_pct

        position_value_usdt = risk_usdt / sl_pct
        size_coins = position_value_usdt / entry_price

        # Ограничение по максимальному плечу
        size_coins = min(size_coins, (self.deposit * self.max_leverage) / entry_price)

        logger.debug(f"[{self.risk_mode.upper()}] Размер позиции для {symbol}: {size_coins:.6f} монет "
                     f"(риск {risk_usdt:.2f} USDT, SL {sl_pct*100:.2f}%)")

        return size_coins

    def can_open_new_position(self) -> bool:
        if self.current_daily_loss <= -self.deposit * self.daily_loss_limit:
            logger.warning(f"Достигнут дневной лимит убытка — новые позиции запрещены")
            return False

        if self.open_positions_count >= self.max_open_positions:
            logger.info(f"Достигнут лимит открытых позиций: {self.open_positions_count}/{self.max_open_positions}")
            return False
        return True

    def update_deposit(self, pnl: float):
        self.deposit += pnl
        self.current_daily_loss += pnl if pnl < 0 else 0

    def reset_daily_loss(self):
        self.current_daily_loss = 0.0
        logger.info("Дневной лимит убытка сброшен")