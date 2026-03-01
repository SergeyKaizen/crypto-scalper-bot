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

Изменения Этапа 2 (пункт 4):
- Убрано любое умножение размера позиции на коэффициент режима (regime_factor, regime_strength и т.п.)
- Размер позиции теперь считается строго по risk_pct без дополнительных множителей
- Это снижает риск переоценки позиции в сильном режиме и делает расчёт более предсказуемым

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
        """
        Расчёт размера позиции.

        Изменения по пункту 4:
        - Убрано любое умножение на regime_factor / regime_strength
        - Чистый расчёт: size = (deposit * risk_pct) / |entry - sl|
        """
        if entry_price <= 0 or sl_price <= 0:
            logger.warning(f"Некорректные цены для {symbol}: entry={entry_price}, sl={sl_price}")
            return 0.0

        risk_amount = self.deposit * self.risk_pct
        price_diff = abs(entry_price - sl_price)

        if price_diff == 0:
            logger.warning(f"SL = entry для {symbol} — размер позиции = 0")
            return 0.0

        size = risk_amount / price_diff

        # Учёт минимального размера и шага лота (если есть в конфиге)
        min_size = self.config["trading"].get("min_position_size", 0.001)
        size = max(min_size, size)

        # Учёт максимального leverage (не превышаем)
        max_size_by_leverage = (self.deposit * self.max_leverage) / entry_price
        size = min(size, max_size_by_leverage)

        logger.debug(f"Размер позиции для {symbol}: {size:.6f} (risk {self.risk_pct*100}%, расстояние до SL {price_diff:.2f})")

        return size

    def can_open_new_position(self) -> bool:
        """Можно ли открыть новую позицию (по лимиту открытых)"""
        # FIX Фаза 2: теперь daily_loss_limit реально блокирует новые позиции
        if self.current_daily_loss <= -self.deposit * self.daily_loss_limit:
            logger.warning(f"Достигнут дневной лимит убытка — новые позиции запрещены: {self.current_daily_loss:.2f} / {self.deposit * self.daily_loss_limit:.2f}")
            return False

        if self.open_positions_count >= self.max_open_positions:
            logger.info(f"Достигнут лимит открытых позиций: {self.open_positions_count}/{self.max_open_positions}")
            return False
        return True

    def update_after_open(self):
        """Обновление счётчиков после открытия позиции"""
        self.open_positions_count += 1

    def update_after_close(self):
        """Обновление после закрытия позиции"""
        self.open_positions_count = max(0, self.open_positions_count - 1)

    def update_deposit(self, pnl: float):
        """Обновление депозита после закрытия позиции"""
        self.deposit += pnl
        self.current_daily_loss += pnl if pnl < 0 else 0

        if self.current_daily_loss <= -self.deposit * self.daily_loss_limit:
            logger.warning(f"Достигнут дневной лимит убытка: {self.current_daily_loss:.2f} / {self.deposit * self.daily_loss_limit:.2f}")
            # Здесь можно добавить глобальный флаг остановки торговли

    def reset_daily_loss(self):
        """Сброс дневного убытка (вызывается в 00:00 UTC)"""
        self.current_daily_loss = 0.0
        logger.info("Дневной лимит убытка сброшен")

    # ... остальные методы (если есть) остаются без изменений ...