"""
src/trading/tp_sl_manager.py

=== Основной принцип работы файла ===

TP/SL Manager — модуль управления тейк-профитами и стоп-лоссами.
Теперь поддерживает trailing_mode (manual/auto) и partial_trailing с объёмами фиксации.
"""

import logging
from typing import Dict, Optional

from src.core.config import load_config
from src.core.enums import AnomalyType, Direction
from src.utils.logger import setup_logger

logger = setup_logger('tp_sl_manager', logging.INFO)

class TP_SL_Manager:
    def __init__(self):
        self.config = load_config()
        self.trailing_mode = self.config["trading"].get("trailing_mode", "manual")
        self.trailing_activation_pct = self.config["trading"].get("trailing_activation_pct", 0.015)
        self.trailing_distance_pct = self.config["trading"].get("trailing_distance_pct", 0.008)
        self.partial_tp_levels = self.config["trading"].get("partial_tp_levels", [0.5, 0.75])
        self.partial_tp_volumes = self.config["trading"].get("partial_tp_volumes", [50, 25])
        self.open_positions = {}

    def calculate_tp_sl(self, features: Dict, anomaly_type: str) -> Dict[str, Optional[float]]:
        entry_price = features.get('close', 0.0)
        atr = features.get('atr', 0.0)

        if entry_price <= 0:
            return {'tp': None, 'sl': None}

        tp_multiplier = self.config["trading"].get("tp_multiplier", 1.0)
        sl_multiplier = self.config["trading"].get("sl_multiplier", 1.0)

        if anomaly_type == AnomalyType.C.value:
            tp_distance = atr * tp_multiplier * 1.2
            sl_distance = atr * sl_multiplier * 0.8
        elif anomaly_type == AnomalyType.V.value:
            tp_distance = atr * tp_multiplier * 0.8
            sl_distance = atr * sl_multiplier * 1.2
        else:
            tp_distance = atr * tp_multiplier
            sl_distance = atr * sl_multiplier

        direction = features.get('direction', 'L')
        if direction == 'L':
            tp = entry_price + tp_distance
            sl = entry_price - sl_distance
        else:
            tp = entry_price - tp_distance
            sl = entry_price + sl_distance

        return {'tp': round(tp, 4), 'sl': round(sl, 4)}

    def calculate_sl(self, candle_data: Dict, direction: str) -> float:
        close = candle_data.get('close', 0.0)
        atr = candle_data.get('atr', 0.0)
        sl_multiplier = self.config["trading"].get("sl_multiplier", 1.0)
        sl_distance = atr * sl_multiplier

        if direction == 'L':
            return round(close - sl_distance, 4)
        else:
            return round(close + sl_distance, 4)

    def add_open_position(self, position: Dict):
        pos_id = position.get('pos_id')
        if not pos_id:
            return

        position['trailing_active'] = False
        position['trailing_stop_price'] = position.get('sl')
        position['remaining_size'] = position.get('size', 0.0)   # для partial_tp

        self.open_positions[pos_id] = position

    def check_tp_sl(self, position: Dict, candle_data: Dict) -> bool:
        """Проверяет TP/SL + partial_trailing + trailing_mode"""
        pos_id = position.get('pos_id')
        if pos_id not in self.open_positions:
            return False

        direction = position.get('direction')
        current_price = candle_data.get('close', 0.0)
        high = candle_data.get('high', current_price)
        low = candle_data.get('low', current_price)

        tp = position.get('tp')
        sl = position.get('sl') if not position.get('trailing_active') else position.get('trailing_stop_price')

        hit_tp = (direction == 'L' and high >= tp) or (direction == 'S' and low <= tp)
        hit_sl = (direction == 'L' and low <= sl) or (direction == 'S' and high >= sl)

        if hit_tp and hit_sl:
            if direction == 'L':
                hit_tp = (high - tp) < (sl - low)
            else:
                hit_tp = (tp - low) < (high - sl)

        if hit_tp or hit_sl:
            del self.open_positions[pos_id]
            return True

        # Partial trailing + trailing_mode
        self._handle_partial_trailing(position, current_price)
        self.update_trailing_stop(position, current_price)
        return False

    def _handle_partial_trailing(self, position: Dict, current_price: float):
        if self.config["trading"].get("tp_sl_mode") != "partial_trailing":
            return

        direction = position.get('direction')
        entry = position.get('entry_price')
        remaining = position.get('remaining_size', position.get('size', 0.0))

        # === НОВЫЙ АВТО РЕЖИМ ПО ТЗ ===
        if self.config["trading"].get("partial_trailing_mode", "auto") == "auto":
            # Первый уровень = TP
            tp_price = position.get('tp')
            if (direction == 'L' and current_price >= tp_price) or (direction == 'S' and current_price <= tp_price):
                close_volume = remaining * 0.5
                logger.info(f"Частичная фиксация 50% на уровне TP")
                position['remaining_size'] = remaining - close_volume

            # Второй уровень = половина TP
            half_tp = (position.get('tp') + position.get('sl')) / 2
            if (direction == 'L' and current_price >= half_tp) or (direction == 'S' and current_price <= half_tp):
                close_volume = remaining * 0.3
                logger.info(f"Частичная фиксация 30% на уровне TP/2")
                position['remaining_size'] = remaining - close_volume

            # Остаток 20% — trailing
            return

        # Ручной режим (оставляем как было)
        for i, level in enumerate(self.partial_tp_levels):
            tp_price = entry * (1 + level) if direction == 'L' else entry * (1 - level)
            if (direction == 'L' and current_price >= tp_price) or (direction == 'S' and current_price <= tp_price):
                close_volume = remaining * (self.partial_tp_volumes[i] / 100.0)
                logger.info(f"Частичная фиксация {self.partial_tp_volumes[i]}% на уровне {level}")
                position['remaining_size'] = remaining - close_volume

    def update_trailing_stop(self, position: Dict, current_price: float):
        if 'trailing_active' not in position:
            return

        direction = position.get('direction')
        entry_price = position.get('entry_price', 0.0)

        # trailing_mode: auto = % от TP
        if self.trailing_mode == "auto":
            tp_distance = abs(position.get('tp', 0) - entry_price)
            activation_pct = tp_distance / entry_price
        else:
            activation_pct = self.trailing_activation_pct

        if not position['trailing_active']:
            profit_pct = (current_price - entry_price) / entry_price if direction == 'L' else (entry_price - current_price) / entry_price
            if profit_pct >= activation_pct:
                position['trailing_active'] = True

        if position['trailing_active']:
            if direction == 'L':
                new_trailing_sl = current_price * (1 - self.trailing_distance_pct)
                if new_trailing_sl > position.get('trailing_stop_price', 0):
                    position['trailing_stop_price'] = round(new_trailing_sl, 4)
            else:
                new_trailing_sl = current_price * (1 + self.trailing_distance_pct)
                if new_trailing_sl < position.get('trailing_stop_price', 0):
                    position['trailing_stop_price'] = round(new_trailing_sl, 4)