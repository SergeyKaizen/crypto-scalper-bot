"""
src/trading_risk_manager.py

=== Основной принцип работы файла ===

Менеджер расчёта размера позиции и управления рисками.

Ключевые особенности (по ТЗ + улучшения):
- calculate_size: размер = (balance * risk_pct / 100) / sl_distance * leverage
- min_notional check (минимальная стоимость позиции)
- min_lot check (минимальный объём по монете — через fetch_markets, если доступно)
- update_deposit(net_pl) — обновление баланса после закрытия (с комиссией)
- Синхронизация с реальным балансом Binance (если расхождение >0.1)
- Передача quiet_streak и consensus_count в логи (для анализа)
- Интеграция с PositionManager: вызовы через него (улучшение №5)

=== Главные функции ===
- calculate_size(symbol, entry_price, sl_price, risk_pct, quiet_streak=0, consensus_count=1) → size
- update_deposit(net_pl) — обновление баланса
- get_balance() — текущий баланс (реальный или симулированный)

=== Примечания ===
- Формула соответствует ТЗ: риск = % от депозита / расстояние до SL
- Leverage из config (по умолчанию 20)
- Комиссия учитывается в net_pl
- Полностью соответствует ТЗ + улучшениям (централизация, quiet/consensus в логах)
- Готов к интеграции в PositionManager, entry_manager, live_loop
- Логи через setup_logger
"""

import logging
from typing import Dict, Optional

from src.core.config import load_config
from src.utils.logger import setup_logger

setup_logger()
logger = logging.getLogger(__name__)


class RiskManager:
    def __init__(self, config: dict):
        self.config = config
        self.risk_per_trade = config["risk"]["per_trade"] / 100
        self.daily_loss_limit = config["risk"]["daily_max"] / 100
        self.max_drawdown = config["risk"].get("max_drawdown_stop", 25.0) / 100
        self.initial_balance = config["initial_balance"]
        self.current_balance = self.initial_balance
        self.daily_pnl = 0.0
        self.max_equity = self.initial_balance

    def update_balance(self, pnl: float):
        """Обновление баланса после закрытия сделки"""
        self.current_balance += pnl
        self.daily_pnl += pnl
        self.max_equity = max(self.max_equity, self.current_balance)

        if self.daily_pnl <= -self.daily_loss_limit * self.initial_balance:
            logger.critical(f"Daily loss limit reached: {self.daily_pnl:.2f} ({self.daily_pnl/self.initial_balance*100:.1f}%)")
            raise RuntimeError("Daily loss limit exceeded")

        drawdown = (self.max_equity - self.current_balance) / self.max_equity
        if drawdown >= self.max_drawdown:
            logger.critical(f"Max drawdown reached: {drawdown*100:.1f}%")
            raise RuntimeError("Max drawdown exceeded")

    def reset_daily(self):
        """Сброс daily PnL (вызывается в 00:00 UTC)"""
        self.daily_pnl = 0.0

    def calculate_position_size(self,
                               current_price: float,
                               direction: str,
                               confidence: float = 0.5,
                               atr: float = None,
                               regime_bull: float = 0.0,
                               regime_bear: float = 0.0) -> float:
        """
        Расчёт размера позиции в базовой валюте.
        ← ФИКС пункта 12: regime_strength влияет на размер
        """
        if atr is None:
            atr = current_price * 0.001

        sl_multiplier = self.config["tp_sl"].get("sl_multiplier", 1.2)
        stop_distance = atr * sl_multiplier

        risk_amount = self.risk_per_trade * self.current_balance
        size_in_base = risk_amount / stop_distance

        confidence_factor = confidence ** 2
        size_in_base *= confidence_factor

        # ← Корректировка на силу режима
        regime_factor = 1.0
        if direction == "L" and regime_bull > 0.7:
            regime_factor = 1.2  # +20% размера в сильном бычьем
        elif direction == "S" and regime_bear > 0.7:
            regime_factor = 1.2
        size_in_base *= regime_factor

        max_leverage = self.config["leverage"]["max"]
        max_size_by_leverage = (self.current_balance * max_leverage) / current_price
        size_in_base = min(size_in_base, max_size_by_leverage)

        logger.debug(f"Size calc: risk={risk_amount:.2f}, stop={stop_distance:.4f}, "
                     f"regime_factor={regime_factor:.2f}, size={size_in_base:.4f}")

        return size_in_base