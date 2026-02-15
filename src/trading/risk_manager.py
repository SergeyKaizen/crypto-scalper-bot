# src/trading/risk_manager.py
"""
Менеджер риска — расчёт размера позиции, проверка лимитов и корректировка рисков.

Ключевые требования из ТЗ:
- Риск на сделку — авто-расчёт на основе риск-профита (TP/SL) или ручной ввод (0.5%, 1.1%, 2% и т.д.)
- Авто-режим: риск = 1 / (TP/SL ratio) — баланс риска и профита
- Плечо — проверка максимального допустимого для монеты (если задано 50, а монета позволяет 10 — берём 10)
- Корреляционный фильтр — опционально (по умолчанию выключен)
- Max positions — ограничение общего количества открытых позиций
- Режимы агрессивности (conservative / balanced / aggressive / custom) — разные риски и min_prob
- Авто-переключение режимов — отключено по умолчанию (можно включить)

Логика:
- calculate_size() — основной метод, возвращает размер позиции в USDT
- allow_new_position() — проверяет, можно ли открывать новую сделку (лимит позиций, корреляция и т.д.)
- get_current_risk() — возвращает текущий % риска на сделку (зависит от режима)
"""

import time
from typing import Dict, Optional
import numpy as np

from src.data.storage import Storage
from src.utils.logger import get_logger
from src.core.config import load_config

logger = get_logger(__name__)

class RiskManager:
    def __init__(self, config: dict):
        self.config = config
        self.storage = Storage(config)

        # Основные настройки риска
        self.risk_mode = config["trading_mode"]["risk_mode"]  # conservative / balanced / aggressive / custom
        self.manual_risk_pct = config["risk"].get("manual_risk_pct", None)  # если задан — переопределяет авто

        # Риски по режимам (в % от депозита на сделку)
        self.risk_levels = {
            "conservative": 0.3,   # низкий риск — ниже прибыль
            "balanced":     1.0,   # средний
            "aggressive":   2.0    # высокий риск — высокая прибыль
        }

        self.current_risk_pct = self._get_risk_pct()

        # Лимиты
        self.max_positions = config["risk"].get("max_positions", 10)
        self.correlation_filter_enabled = config["risk"].get("correlation_filter", {}).get("enabled", False)
        self.correlation_threshold = config["risk"].get("correlation_filter", {}).get("threshold", 0.75)

        # Кэш открытых позиций (обновляется из live_loop)
        self.open_positions_count = 0

    def _get_risk_pct(self) -> float:
        """Возвращает текущий % риска на сделку в зависимости от режима"""
        if self.manual_risk_pct is not None:
            return self.manual_risk_pct
        
        return self.risk_levels.get(self.risk_mode, 1.0)

    def calculate_size(self,
                      balance: float,
                      entry_price: float,
                      direction: str,
                      tp_sl_ratio: float = 1.0,  # TP/SL расстояние в % (из tp_sl_manager)
                      stop_loss_pct: Optional[float] = None) -> float:
        """
        Основной метод расчёта размера позиции.

        Логика авто-режима:
        - Риск % = 1 / tp_sl_ratio (баланс риска и профита)
        - Если задан manual_risk_pct — используется он
        - Если задан stop_loss_pct — риск считается от SL расстояния

        Возвращает размер позиции в USDT (или контрактах — зависит от конфига)
        """
        risk_pct = stop_loss_pct or self.current_risk_pct
        
        # Авто-режим: риск пропорционален обратному отношению TP/SL
        if self.manual_risk_pct is None:
            risk_pct = min(2.0, max(0.3, 1.0 / tp_sl_ratio))  # ограничиваем 0.3–2.0%

        # Размер риска в деньгах
        risk_amount = balance * (risk_pct / 100)

        # Расстояние до SL в % (если не задано — берём из tp_sl_ratio)
        sl_distance_pct = stop_loss_pct or (1.0 / tp_sl_ratio)

        # Размер позиции в USDT
        position_size_usdt = risk_amount / (sl_distance_pct / 100)

        # Учёт плеча (если торговля с плечом)
        leverage = self.config["finance"].get("leverage", 1)
        position_size_usdt *= leverage

        logger.debug(f"Рассчитан размер позиции для {direction}: {position_size_usdt:.2f} USDT "
                     f"(риск {risk_pct:.2f}%, SL distance {sl_distance_pct:.2f}%)")

        return position_size_usdt

    def allow_new_position(self, symbol: str, open_positions: Dict[str, Dict]) -> bool:
        """
        Проверяет, можно ли открывать новую позицию.

        Условия:
        - Не превышен max_positions
        - Корреляционный фильтр (если включён)
        - Монета в whitelist
        """
        if len(open_positions) >= self.max_positions:
            logger.debug(f"Достигнут лимит позиций ({self.max_positions})")
            return False

        if symbol not in self.storage.get_whitelist():
            logger.debug(f"{symbol} не в whitelist → новая позиция запрещена")
            return False

        if self.correlation_filter_enabled:
            # Здесь можно добавить реальную проверку корреляции (например rolling 4h)
            # Пока заглушка — всегда разрешено
            pass

        return True

    def update_open_positions_count(self, count: int):
        """Обновляет текущее количество открытых позиций (вызывается из live_loop)"""
        self.open_positions_count = count

    def get_current_risk(self) -> float:
        """Возвращает текущий % риска на сделку"""
        return self.current_risk_pct

    def switch_mode(self, new_mode: str):
        """Переключение режима агрессивности (если авто-включено)"""
        if new_mode not in self.risk_levels:
            logger.warning(f"Неизвестный режим риска: {new_mode}")
            return
        
        self.risk_mode = new_mode
        self.current_risk_pct = self._get_risk_pct()
        logger.info(f"Режим риска переключён на {new_mode} (риск на сделку: {self.current_risk_pct}%)")


if __name__ == "__main__":
    config = load_config()
    rm = RiskManager(config)
    
    # Тест расчёта размера
    size = rm.calculate_size(balance=10000, entry_price=60000, direction="L", tp_sl_ratio=1.5)
    print(f"Размер позиции: {size:.2f} USDT")
    
    print(f"Текущий риск %: {rm.get_current_risk()}")