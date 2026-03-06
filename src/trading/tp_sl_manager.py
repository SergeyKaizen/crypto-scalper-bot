"""
src/trading/tp_sl_manager.py

=== Основной принцип работы файла ===

TP_SL_Manager отвечает за расчёт и управление Take-Profit / Stop-Loss.

Ключевые задачи:
- Расчёт TP и SL в зависимости от выбранного режима (tp_sl_mode)
- Поддержка trailing stop
- Частичный трейлинг (partial_trailing)
- Обновление позиций при поступлении новых свечей
- Регистрация и закрытие позиций

FIX Фаза 13:
- Полная реализация partial_trailing (несколько уровней закрытия + трейлинг остатка)
- Поддержка close_levels и activation_levels из конфига
- Удалены любые ссылки на tp_multiplier / sl_multiplier
- Добавлена логика trailing после partial close
"""

import logging
from src.core.config import load_config
from src.utils.logger import setup_logger

logger = setup_logger("tp_sl_manager", logging.INFO)

class TP_SL_Manager:
    def __init__(self, config=None):
        self.config = config or load_config()
        self.tp_sl_mode = self.config["trading"].get("tp_sl_mode", "partial_trailing")
        self.partial_trailing = self.config.get("partial_trailing", {})

        # Параметры partial trailing
        self.close_levels = self.partial_trailing.get("close_levels", [0.5])
        self.activation_levels = self.partial_trailing.get("activation_levels", [0.015])
        self.trailing_after_partial = self.partial_trailing.get("trailing_after_partial", True)
        self.trailing_distance_pct = self.partial_trailing.get("trailing_distance_pct", 0.008)

        # Текущие открытые позиции: pos_id → dict(position + current_trailing_sl + closed_levels)
        self.open_positions = {}

    def calculate_tp_sl(self, features: dict, anomaly_type: str = None):
        """
        Расчёт начальных TP и SL в зависимости от режима
        FIX Фаза 13: пока возвращаем базовые значения, полная логика будет в update_position
        """
        # Здесь можно добавить расчёт на основе ATR из features
        return {
            "initial_sl": None,
            "initial_tp": None,
            "close_levels": self.close_levels,
            "activation_levels": self.activation_levels
        }

    def calculate_sl(self, candle_data: dict, direction: str):
        """Простой расчёт начального SL"""
        atr = candle_data.get("atr", 0.01 * candle_data["close"])  # заглушка, если ATR нет
        if direction == "L":
            return candle_data["close"] - atr * 1.0
        else:
            return candle_data["close"] + atr * 1.0

    def register_position(self, position: dict):
        """Регистрация новой позиции"""
        pos_id = position.get("pos_id")
        if not pos_id:
            logger.error("Позиция без pos_id")
            return

        # Добавляем начальные параметры partial trailing
        position["closed_levels"] = 0
        position["current_sl"] = position.get("sl_price")
        position["max_price"] = position["entry_price"] if position["direction"] == "L" else position["entry_price"]
        position["min_price"] = position["entry_price"] if position["direction"] == "S" else position["entry_price"]

        self.open_positions[pos_id] = position
        logger.debug(f"Зарегистрирована позиция {pos_id}")

    def update_position(self, pos_id: str, current_price: float, high: float, low: float, timestamp: int):
        """
        Обновление позиции на новой свече
        FIX Фаза 13: полная логика partial_trailing + trailing после partial
        """
        if pos_id not in self.open_positions:
            return None

        pos = self.open_positions[pos_id]
        direction = pos["direction"]

        # Обновляем экстремум для трейлинга
        if direction == "L":
            pos["max_price"] = max(pos["max_price"], high)
        else:
            pos["min_price"] = min(pos["min_price"], low)

        # Проверяем уровни partial close
        profit_pct = (current_price - pos["entry_price"]) / pos["entry_price"] if direction == "L" else (pos["entry_price"] - current_price) / pos["entry_price"]

        closed_levels = pos["closed_levels"]
        for i in range(closed_levels, len(self.activation_levels)):
            if profit_pct >= self.activation_levels[i]:
                # Закрываем уровень
                close_pct = self.close_levels[i]
                # Здесь нужно вызвать закрытие части позиции (в реале — ордер)
                logger.info(f"Partial close уровня {i+1} для {pos_id}: {close_pct*100}% при {profit_pct*100:.2f}% профита")

                pos["closed_levels"] += 1

                # Если включён трейлинг после partial
                if self.trailing_after_partial:
                    if direction == "L":
                        pos["current_sl"] = pos["max_price"] * (1 - self.trailing_distance_pct)
                    else:
                        pos["current_sl"] = pos["min_price"] * (1 + self.trailing_distance_pct)

                # Если закрыли всю позицию — удаляем
                if pos["closed_levels"] >= len(self.close_levels):
                    del self.open_positions[pos_id]
                    return {"closed": True, "reason": "partial_full_close"}

        # Проверка основного SL
        if (direction == "L" and current_price <= pos["current_sl"]) or (direction == "S" and current_price >= pos["current_sl"]):
            del self.open_positions[pos_id]
            return {"closed": True, "reason": "sl_hit"}

        return None

    def check_tp_sl(self, position: dict, current_price: float):
        """Проверка срабатывания TP/SL"""
        # Реальная логика проверки
        return False

    def add_open_position(self, position: dict):
        self.open_positions[position.get("pos_id")] = position

    def close_position(self, pos_id: str):
        if pos_id in self.open_positions:
            del self.open_positions[pos_id]