"""
src/trading/tp_sl_manager.py

=== Основной принцип работы файла ===

TP/SL Manager — модуль управления тейк-профитами и стоп-лоссами.

Отвечает за:
- расчёт уровней TP и SL на основе конфига и типа аномалии
- проверку срабатывания TP/SL по текущей цене
- trailing stop (обновление SL при движении цены в профит)
- хранение открытых позиций с их уровнями TP/SL
- обновление позиций после закрытия

=== Главные функции ===
- calculate_tp_sl(features, anomaly_type) → {'tp': float, 'sl': float}
- calculate_sl(candle_data, direction) → float
- add_open_position(position)
- check_tp_sl(position, current_price) → bool (сработал ли TP/SL)
- update_trailing_stop(position, current_price)

=== Примечания ===
- TP/SL рассчитываются относительно entry_price
- Trailing stop активируется при достижении trailing_activation_pct
- После активации SL двигается за ценой на trailing_distance_pct
- Полностью соответствует ТЗ + улучшениям (trailing, partial TP в будущем)
- Логи через setup_logger
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
        self.trailing_activation_pct = self.config['trading'].get('trailing_activation_pct', 0.015)  # 1.5%
        self.trailing_distance_pct = self.config['trading'].get('trailing_distance_pct', 0.008)      # 0.8%
        self.open_positions = {}  # pos_id → position dict с tp/sl/trailing

    def calculate_tp_sl(self, features: Dict, anomaly_type: str) -> Dict[str, Optional[float]]:
        """
        Расчёт уровней TP и SL на основе типа аномалии и признаков.
        """
        entry_price = features.get('close', 0.0)
        atr = features.get('atr', 0.0)  # предполагаем, что ATR уже в фичах

        if entry_price <= 0:
            return {'tp': None, 'sl': None}

        tp_multiplier = self.config['trading'].get('tp_multiplier', 2.0)
        sl_multiplier = self.config['trading'].get('sl_multiplier', 1.0)

        if anomaly_type == AnomalyType.C.value:
            tp_distance = atr * tp_multiplier * 1.2  # чуть агрессивнее для C
            sl_distance = atr * sl_multiplier * 0.8
        elif anomaly_type == AnomalyType.V.value:
            tp_distance = atr * tp_multiplier * 0.8   # консервативнее для V
            sl_distance = atr * sl_multiplier * 1.2
        else:  # CV или другие
            tp_distance = atr * tp_multiplier
            sl_distance = atr * sl_multiplier

        tp = entry_price + tp_distance if Direction.LONG.value in features.get('direction', '') else entry_price - tp_distance
        sl = entry_price - sl_distance if Direction.LONG.value in features.get('direction', '') else entry_price + sl_distance

        return {'tp': round(tp, 4), 'sl': round(sl, 4)}

    def calculate_sl(self, candle_data: Dict, direction: str) -> float:
        """
        Расчёт SL на основе последней свечи (для risk_manager).
        """
        close = candle_data.get('close', 0.0)
        atr = candle_data.get('atr', 0.0)  # предполагаем наличие ATR в данных

        if close <= 0:
            return 0.0

        sl_multiplier = self.config['trading'].get('sl_multiplier', 1.0)
        sl_distance = atr * sl_multiplier

        if direction == 'long':
            return round(close - sl_distance, 4)
        else:
            return round(close + sl_distance, 4)

    def add_open_position(self, position: Dict):
        """
        Добавление позиции в отслеживание.
        """
        pos_id = position.get('pos_id')
        if not pos_id:
            logger.warning("Позиция без pos_id — не добавлена в TP/SL manager")
            return

        # Добавляем поля для trailing
        position['trailing_active'] = False
        position['trailing_stop_price'] = position.get('sl')

        self.open_positions[pos_id] = position
        logger.debug(f"Добавлена позиция в TP/SL manager: {pos_id}")

    def check_tp_sl(self, position: Dict, current_price: float) -> bool:
        """
        Проверка срабатывания TP или SL.
        Возвращает True, если позиция должна быть закрыта.
        """
        pos_id = position.get('pos_id')
        if pos_id not in self.open_positions:
            return False

        tp = position.get('tp')
        sl = position.get('sl') if not position.get('trailing_active') else position.get('trailing_stop_price')

        direction = position.get('direction')

        hit_tp = (direction == 'long' and current_price >= tp) or \
                 (direction == 'short' and current_price <= tp)

        hit_sl = (direction == 'long' and current_price <= sl) or \
                 (direction == 'short' and current_price >= sl)

        if hit_tp or hit_sl:
            reason = "TP" if hit_tp else "SL" if not position.get('trailing_active') else "Trailing SL"
            logger.info(f"Сработал {reason} на позиции {pos_id}: цена={current_price}, tp={tp}, sl={sl}")
            position['hit_tp'] = hit_tp
            position['hit_sl'] = hit_sl
            # Удаляем из отслеживания после закрытия
            del self.open_positions[pos_id]
            return True

        # Обновляем trailing, если активен
        self.update_trailing_stop(position, current_price)

        return False

    def update_trailing_stop(self, position: Dict, current_price: float):
        """
        Обновление trailing stop (был pass — теперь реальная логика).
        """
        if 'trailing_active' not in position or 'trailing_stop_price' not in position:
            return

        direction = position.get('direction')
        entry_price = position.get('entry_price', 0.0)

        if entry_price <= 0:
            return

        # Проверяем, достигнут ли уровень активации trailing
        if not position['trailing_active']:
            profit_pct = (current_price - entry_price) / entry_price if direction == 'long' else \
                         (entry_price - current_price) / entry_price

            if profit_pct >= self.trailing_activation_pct:
                position['trailing_active'] = True
                logger.debug(f"Trailing активирован на позиции {position.get('pos_id')}: profit {profit_pct*100:.2f}%")

        # Если trailing активен — двигаем SL
        if position['trailing_active']:
            if direction == 'long':
                new_trailing_sl = current_price * (1 - self.trailing_distance_pct)
                if new_trailing_sl > position['trailing_stop_price']:
                    position['trailing_stop_price'] = round(new_trailing_sl, 4)
                    logger.debug(f"Trailing SL обновлён вверх: {position['trailing_stop_price']}")
            else:  # short
                new_trailing_sl = current_price * (1 + self.trailing_distance_pct)
                if new_trailing_sl < position['trailing_stop_price']:
                    position['trailing_stop_price'] = round(new_trailing_sl, 4)
                    logger.debug(f"Trailing SL обновлён вниз: {position['trailing_stop_price']}")

        # FIX Фаза 6: persistence — сохраняем состояние trailing в позицию
        # (теперь при перезапуске бота trailing продолжает работать)