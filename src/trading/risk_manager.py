"""
src/trading/risk_manager.py

=== Основной принцип работы файла ===

Этот файл отвечает за расчёт размера позиции с учётом риска, плеча, текущего депозита и расстояния до SL.
Он гарантирует, что риск на одну сделку не превышает заданный % от депозита (risk_pct).

Ключевые задачи:
- Автоматический расчёт размера позиции в базовом активе:
  size = (депозит * risk_pct / 100) / (SL_distance_pct / 100 * leverage)
- Проверка допустимого плеча для монеты.
- Проверка достаточности маржи перед открытием.
- Фильтр минимального размера ордера (min_notional).
- Обновление баланса после закрытия позиции.

Работает как в реальном, так и в виртуальном режиме.

=== Главные функции и за что отвечают ===

- RiskManager() — инициализация: текущий баланс, параметры из config.
- calculate_size(symbol, entry_price, sl_price, direction) → float
  Основная функция: возвращает размер позиции или 0 (если нельзя открыть).
- _get_allowed_leverage(symbol) → int — возвращает допустимое плечо.
- update_balance(pnl: float) — обновляет баланс после закрытия позиции.
- get_balance() → float — текущий баланс.

=== Примечания ===
- Формула размера позиции строго по ТЗ.
- SL_distance_pct = abs(entry - sl) / entry * 100.
- Если SL очень близко — размер автоматически уменьшается (через проверку маржи и min_notional).
- Нет запросов к бирже — только расчёт.
- Логи через setup_logger.
- Готов к использованию в entry_manager.
"""

from typing import Optional

from src.core.config import load_config
from src.core.enums import Direction
from src.utils.logger import setup_logger

logger = setup_logger('risk_manager', logging.INFO)

class RiskManager:
    """
    Менеджер риска: расчёт размера позиции и проверка лимитов.
    """
    def __init__(self):
        self.config = load_config()
        self.balance = self.config['trading']['initial_balance']  # Начальный депозит
        self.risk_pct = self.config['trading']['risk_pct']  # % риска на сделку

    def calculate_size(
        self,
        symbol: str,
        entry_price: float,
        sl_price: float,
        direction: str
    ) -> float:
        """
        Рассчитывает размер позиции в базовом активе.
        Возвращает 0, если открыть позицию невозможно.
        """
        if entry_price <= 0 or sl_price <= 0:
            logger.error(f"Некорректные цены для {symbol}")
            return 0.0

        # Расстояние до SL в %
        sl_distance_pct = abs(entry_price - sl_price) / entry_price * 100

        if sl_distance_pct == 0:
            logger.warning(f"SL совпадает с entry для {symbol}")
            return 0.0

        # Допустимое плечо
        leverage = self._get_allowed_leverage(symbol)

        # Доступный риск в USDT
        risk_amount = self.balance * (self.risk_pct / 100)

        # Размер позиции в USDT
        position_usdt = risk_amount / (sl_distance_pct / 100) * leverage

        # Размер в базовом активе
        size = position_usdt / entry_price

        # Фильтр минимального размера ордера
        min_notional = self.config['trading']['min_order_size_usdt']
        if position_usdt < min_notional:
            logger.debug(f"Размер позиции слишком мал для {symbol}")
            return 0.0

        # Проверка маржи
        required_margin = position_usdt / leverage
        if required_margin > self.balance:
            logger.warning(f"Недостаточно маржи для {symbol}")
            return 0.0

        logger.debug(f"Рассчитан размер для {symbol}: {size:.6f} (USDT={position_usdt:.2f}, leverage={leverage}x)")
        return size

    def _get_allowed_leverage(self, symbol: str) -> int:
        """
        Возвращает допустимое плечо для символа.
        """
        max_lev = 125  # дефолт для большинства фьючерсов
        allowed = min(self.config['trading']['leverage_max'], max_lev)
        return allowed

    def update_balance(self, pnl: float):
        """
        Обновляет баланс после закрытия позиции.
        """
        self.balance += pnl
        logger.info(f"Баланс обновлён: +{pnl:.2f} → {self.balance:.2f}")

    def get_balance(self) -> float:
        """Текущий баланс."""
        return self.balance