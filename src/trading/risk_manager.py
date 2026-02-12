# src/trading/risk_manager.py
"""
Модуль управления риском и расчёта размера позиции.

Основные функции:
- calculate_position_size() — основной расчёт размера позиции
- get_risk_pct() — определение % риска на сделку (по вероятности или default)
- check_margin_available() — проверка, хватает ли маржи
- adjust_size_for_min_order() — подгонка под минимальный ордер Binance

Логика:
- Риск считается от базы (initial / current)
- manual_prob_risk_levels — если включено, риск зависит от probability
- Формула размера:
  risk_usdt = risk_pct * base_balance
  position_usdt = risk_usdt / (sl_distance_pct / 100)
  size = position_usdt / entry_price * leverage
- Hard-limit: risk_pct ≤ 5.0%
- Плечо — max_leverage монеты (из Binance fetch_markets)
- Комиссия учитывается при PNL (но не здесь)

Зависимости:
- config["risk"]
- BinanceClient — для fetch_markets (плечо, min order)
- constants — MAX_RISK_PCT_HARD_LIMIT, MIN_ORDER_SIZES
"""

import logging
from typing import Dict, Optional
from decimal import Decimal

from src.core.config import load_config
from src.core.constants import MAX_RISK_PCT_HARD_LIMIT, BINANCE_FUTURES_MIN_ORDER_VALUE_USDT
from src.data.binance_client import BinanceClient

logger = logging.getLogger(__name__)


class RiskManager:
    """Управление риском и расчёт размера позиции"""

    def __init__(self, config: Dict):
        self.config = config
        self.risk_base = config["risk"]["risk_base"]
        self.default_risk_pct = config["risk"]["default_risk_pct"]
        self.manual_levels = config["risk"].get("manual_prob_risk_levels", {})
        self.max_risk_hard = MAX_RISK_PCT_HARD_LIMIT

        # Кэш плеча и мин.ордера по монетам
        self.leverage_cache = {}
        self.min_order_cache = {}

    def get_risk_pct(self, probability: float) -> float:
        """Определяет % риска на сделку по вероятности модели"""
        if not self.manual_levels:
            return self.default_risk_pct

        # Ищем ближайший уровень сверху
        sorted_levels = sorted(self.manual_levels.items(), reverse=True)
        for prob_level, risk_pct in sorted_levels:
            if probability >= prob_level:
                return risk_pct

        # Если ниже самого низкого — default
        return self.default_risk_pct

    def calculate_position_size(
        self,
        entry_price: float,
        signal: Dict,
        df: pl.DataFrame,  # Для расчёта SL distance
        current_balance: float,
        leverage: int = 10,
        symbol: str = "BTCUSDT"
    ) -> float:
        """
        Расчёт размера позиции в контрактах/монетах

        Формула:
            risk_usdt = risk_pct * base_balance
            sl_distance_pct = |entry - sl| / entry * 100
            position_usdt = risk_usdt / (sl_distance_pct / 100)
            size = position_usdt / entry_price * leverage

        Args:
            entry_price — цена входа
            signal — содержит probability и direction
            df — последние свечи (для SL расчёта)
            current_balance — текущий баланс (если risk_base=current)
            leverage — текущее плечо
            symbol — для min_order_size

        Returns:
            size — размер позиции (в монетах)
        """
        # 1. Определяем % риска
        prob = signal.get("probability", 0.5)
        risk_pct = self.get_risk_pct(prob)

        # Hard-limit
        risk_pct = min(risk_pct, self.max_risk_hard)

        # 2. База баланса
        if self.risk_base == "initial":
            base_balance = self.config.get("initial_balance", current_balance)
        else:
            base_balance = current_balance

        risk_usdt = base_balance * (risk_pct / 100)

        # 3. Расстояние до SL (пример — используем tp_sl_manager в реальном коде)
        # Здесь упрощённо — берём 0.5% (реально — от tp_sl_manager.calculate_sl_distance)
        sl_distance_pct = 0.5  # Placeholder — заменить реальным расчётом

        if sl_distance_pct <= 0:
            logger.warning("SL distance = 0 → позиция не открывается")
            return 0.0

        position_usdt = risk_usdt / (sl_distance_pct / 100)

        # 4. Размер с учётом плеча
        size = position_usdt / entry_price * leverage

        # 5. Проверка минимального ордера
        min_order = self.get_min_order_size(symbol)
        if size * entry_price < min_order:
            logger.debug("Size too small for %s: %.4f < min %.2f USDT", symbol, size * entry_price, min_order)
            return 0.0

        logger.debug("Calculated size for %s: %.4f (risk=%.2f%%, leverage=%d)", 
                     symbol, size, risk_pct, leverage)

        return size

    def get_min_order_size(self, symbol: str) -> float:
        """Минимальный размер ордера для монеты"""
        if symbol in self.min_order_cache:
            return self.min_order_cache[symbol]

        # В реальном коде — fetch_markets()
        min_size = MIN_ORDER_SIZES.get(symbol, BINANCE_FUTURES_MIN_ORDER_VALUE_USDT / 60000)  # Пример для BTC
        self.min_order_cache[symbol] = min_size
        return min_size

    def check_margin_available(self, required_margin: float, available_margin: float) -> bool:
        """Проверка, хватает ли маржи"""
        if required_margin > available_margin * 0.9:  # 90% — запас
            logger.warning("Недостаточно маржи: required=%.2f, available=%.2f", required_margin, available_margin)
            return False
        return True