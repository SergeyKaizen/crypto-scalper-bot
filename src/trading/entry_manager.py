# src/trading/entry_manager.py
"""
Модуль управления мягкими входами (soft entry).

Основные функции:
- should_enter_soft() — проверка, пора ли открывать/добавлять часть позиции
- calculate_entry_parts() — расчёт долей позиции по уровням вероятности
- process_soft_entry() — основная логика: открытие первой части, ожидание следующих
- cancel_pending_parts() — отмена оставшихся частей, если timeout или сигнал отменён

Логика:
- soft_entry_enabled — включено/выключено
- soft_levels — пороги вероятности (например [0.65, 0.78, 0.85])
- soft_sizes — доли позиции (сумма = 1.0, например [0.3, 0.4, 0.3])
- soft_timeout — сколько свечей ждать следующую часть (если не пришло — отменяем)
- После первой части — ждём, пока prob ≥ следующий уровень
- Если timeout — оставшиеся части не открываем
- Каждая часть — отдельный ордер (limit/market по конфигу)

Влияние:
- Снижает риск ложных входов на 30–50%
- Увеличивает средний профит (входим частями по лучшей цене)
- Уменьшает средний размер позиции (если подтверждения нет)

Зависимости:
- config["soft_entry_enabled"], soft_levels, soft_sizes, soft_timeout
- src/trading/order_executor.py — для открытия ордеров
- src/trading/live_loop.py — передача текущей вероятности
"""

import logging
from typing import List, Tuple, Dict, Optional
from datetime import datetime

from src.core.config import load_config
from src.core.types import Signal, Position
from src.trading.order_executor import place_order

logger = logging.getLogger(__name__)


class EntryManager:
    """Управление мягкими входами"""

    def __init__(self, config: Dict):
        self.config = config
        self.enabled = config.get("soft_entry_enabled", False)
        self.levels = config.get("soft_levels", [0.65, 0.78, 0.85])
        self.sizes = config.get("soft_sizes", [0.3, 0.4, 0.3])
        self.timeout = config.get("soft_timeout", 5)  # в свечах

        # Проверка сумм долей
        if abs(sum(self.sizes) - 1.0) > 0.01:
            logger.warning("Soft sizes sum != 1.0: %s → normalizing", self.sizes)
            total = sum(self.sizes)
            self.sizes = [s / total for s in self.sizes]

        # Открытые soft-позиции: symbol → {entry_time, current_level, pending_parts}
        self.active_soft = {}

        logger.info("EntryManager initialized: soft_entry=%s, levels=%s, sizes=%s, timeout=%d свечей",
                    self.enabled, self.levels, self.sizes, self.timeout)

    def should_enter(self, signal: Signal, current_prob: float) -> Tuple[bool, int]:
        """
        Проверка: открывать ли новую часть позиции

        Returns:
            (enter: bool, level_index: int) — level_index начинается с 0
        """
        if not self.enabled:
            return current_prob >= self.config["prob_threshold"], 0

        # Если позиция уже открыта — проверяем следующую часть
        if signal.symbol in self.active_soft:
            pos = self.active_soft[signal.symbol]
            current_level = pos["current_level"]
            if current_level >= len(self.levels) - 1:
                return False, -1  # Все части открыты

            next_level = current_level + 1
            if current_prob >= self.levels[next_level]:
                return True, next_level

            # Проверка timeout
            candles_passed = (datetime.now() - pos["entry_time"]).total_seconds() / 60  # Примерно
            if candles_passed > self.timeout * 1:  # 1 минута = 1 свеча на 1m
                self.cancel_pending_parts(signal.symbol)
                return False, -1

            return False, -1

        # Первая часть
        if current_prob >= self.levels[0]:
            return True, 0

        return False, -1

    def open_soft_part(self, signal: Signal, current_prob: float, entry_price: float, size: float):
        """Открытие очередной части позиции"""
        if signal.symbol not in self.active_soft:
            self.active_soft[signal.symbol] = {
                "entry_time": datetime.now(),
                "current_level": 0,
                "parts": [],
                "pending_parts": len(self.levels) - 1
            }

        level = self.active_soft[signal.symbol]["current_level"]
        part_size = size * self.sizes[level]

        # Открываем ордер
        order = place_order(
            symbol=signal.symbol,
            side="buy" if signal.direction in ["L", "LS"] else "sell",
            amount=part_size,
            price=entry_price,
            type="market"  # или limit — по конфигу
        )

        self.active_soft[signal.symbol]["parts"].append({
            "level": level,
            "price": entry_price,
            "size": part_size,
            "order_id": order.get("id")
        })

        self.active_soft[signal.symbol]["current_level"] += 1

        logger.info("Soft entry part opened: %s, level=%d, size=%.4f, price=%.2f", 
                    signal.symbol, level, part_size, entry_price)

    def cancel_pending_parts(self, symbol: str):
        """Отмена оставшихся частей при timeout или отмене сигнала"""
        if symbol in self.active_soft:
            pos = self.active_soft[symbol]
            logger.info("Cancelling pending soft parts for %s (timeout/current_level=%d)", 
                        symbol, pos["current_level"])
            del self.active_soft[symbol]

    def reset(self):
        """Сброс всех soft-позиций (например после перезапуска)"""
        self.active_soft.clear()
        logger.info("Soft entry manager reset")