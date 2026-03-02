"""
src/trading/tp_sl_manager.py

=== Основной принцип работы файла ===

TP/SL Manager — модуль управления тейк-профитами и стоп-лоссами.
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
        self.trailing_activation_pct = self.config['trading'].get('trailing_activation_pct', 0.015)
        self.trailing_distance_pct = self.config['trading'].get('trailing_distance_pct', 0.008)
        self.open_positions = {}

    def calculate_tp_sl(self, features: Dict, anomaly_type: str) -> Dict[str, Optional[float]]:
        entry_price = features.get('close', 0.0)
        atr = features.get('atr', 0.0)

        if entry_price <= 0:
            return {'tp': None, 'sl': None}

        tp_multiplier = self.config['trading'].get('tp_multiplier', 2.0)
        sl_multiplier = self.config['trading'].get('sl_multiplier', 1.0)

        if anomaly_type == AnomalyType.C.value:
            tp_distance = atr * tp_multiplier * 1.2
            sl_distance = atr * sl_multiplier * 0.8
        elif anomaly_type == AnomalyType.V.value:
            tp_distance = atr * tp_multiplier * 0.8
            sl_distance = atr * sl_multiplier * 1.2
        else:
            tp_distance = atr * tp_multiplier
            sl_distance = atr * sl_multiplier

        tp = entry_price + tp_distance if Direction.LONG.value in features.get('direction', '') else entry_price - tp_distance
        sl = entry_price - sl_distance if Direction.LONG.value in features.get('direction', '') else entry_price + sl_distance

        return {'tp': round(tp, 4), 'sl': round(sl, 4)}

    def calculate_sl(self, candle_data: Dict, direction: str) -> float:
        close = candle_data.get('close', 0.0)
        atr = candle_data.get('atr', 0.0)

        if close <= 0:
            return 0.0

        sl_multiplier = self.config['trading'].get('sl_multiplier', 1.0)
        sl_distance = atr * sl_multiplier

        if direction == 'long':
            return round(close - sl_distance, 4)
        else:
            return round(close + sl_distance, 4)

    def add_open_position(self, position: Dict):
        pos_id = position.get('pos_id')
        if not pos_id:
            return

        position['trailing_active'] = False
        position['trailing_stop_price'] = position.get('sl')

        self.open_positions[pos_id] = position

    def check_tp_sl(self, position: Dict, current_price: float) -> bool:
        pos_id = position.get('pos_id')
        if pos_id not in self.open_positions:
            return False

        tp = position.get('tp')
        sl = position.get('sl') if not position.get('trailing_active') else position.get('trailing_stop_price')

        direction = position.get('direction')

        hit_tp = (direction == 'long' and current_price >= tp) or (direction == 'short' and current_price <= tp)
        hit_sl = (direction == 'long' and current_price <= sl) or (direction == 'short' and current_price >= sl)

        if hit_tp or hit_sl:
            del self.open_positions[pos_id]
            return True

        self.update_trailing_stop(position, current_price)
        return False

    def update_trailing_stop(self, position: Dict, current_price: float):
        if 'trailing_active' not in position or 'trailing_stop_price' not in position:
            return

        direction = position.get('direction')
        entry_price = position.get('entry_price', 0.0)

        if not position['trailing_active']:
            profit_pct = (current_price - entry_price) / entry_price if direction == 'long' else (entry_price - current_price) / entry_price
            if profit_pct >= self.trailing_activation_pct:
                position['trailing_active'] = True

        if position['trailing_active']:
            if direction == 'long':
                new_trailing_sl = current_price * (1 - self.trailing_distance_pct)
                if new_trailing_sl > position['trailing_stop_price']:
                    position['trailing_stop_price'] = round(new_trailing_sl, 4)
            else:
                new_trailing_sl = current_price * (1 + self.trailing_distance_pct)
                if new_trailing_sl < position['trailing_stop_price']:
                    position['trailing_stop_price'] = round(new_trailing_sl, 4)