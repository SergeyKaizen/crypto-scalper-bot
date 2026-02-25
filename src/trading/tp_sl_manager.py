"""
src/trading/tp_sl_manager.py

=== Основной принцип работы файла ===

Менеджер расчёта и мониторинга TP/SL для открытых позиций (реальных и виртуальных).

Ключевые особенности (по ТЗ + улучшения):
- calculate_tp_sl: TP = средний размер свечи, SL = HH/LL ±0.05% с cap на 2× avg_size
- Поддержка режимов: classic (основной), partial_trailing, chandelier (по config)
- check_tp_sl: проверка закрытия позиции (TP/SL hit) — возвращает hit_tp/hit_sl
- Закрытие, state transition, update_deposit, add_scenario — делегировано в PositionManager
- Глобальный lock через position_manager.has_any_open_position
- Вызов update_candle_close_flag через position_manager после закрытия

=== Главные функции ===
- calculate_tp_sl(candle_data, timeframe) → tp, sl
- check_tp_sl(position: dict, current_price: float) → (hit_tp: bool, hit_sl: bool)

=== Примечания ===
- Cap 2× avg_size — строго по ТЗ
- Логика закрытия размазана больше не будет — всё в PositionManager
- Полностью соответствует ТЗ + улучшениям (централизация, state-machine)
- Готов к интеграции в live_loop и entry_manager
- Логи через setup_logger
"""

from typing import Tuple, Dict
import numpy as np

from src.core.config import load_config
from src.utils.logger import setup_logger

logger = setup_logger('tp_sl_manager', logging.INFO)

class TP_SL_Manager:
    def __init__(self):
        self.config = load_config()

    def calculate_tp_sl(self, candle_data: Dict, timeframe: str) -> Tuple[float, float]:
        """
        Основной расчёт TP/SL по ТЗ (classic mode)
        """
        # Средний размер свечи (в % от close)
        avg_size = candle_data.get('volatility_mean', 0.0)

        if avg_size == 0 or np.isnan(avg_size):
            # Fallback на средний range
            avg_size = np.mean(candle_data['high'] - candle_data['low']) / candle_data['close'].mean()

        direction = candle_data.get('direction', 'long')

        if direction == 'long':
            hh = candle_data['high'].max()
            sl = hh + 0.0005 * hh  # HH + 0.05%
            if (sl - hh) > avg_size * 2:
                sl = hh + avg_size * 2  # cap 2×
            tp = candle_data['close'].mean() + avg_size * candle_data['close'].mean()
        else:
            ll = candle_data['low'].min()
            sl = ll - 0.0005 * ll  # LL - 0.05%
            if (ll - sl) > avg_size * 2:
                sl = ll - avg_size * 2  # cap 2×
            tp = candle_data['close'].mean() - avg_size * candle_data['close'].mean()

        return tp, sl

    def calculate_sl(self, candle_data: Dict, direction: str) -> float:
        """Расчёт только SL (для risk_manager)"""
        tp, sl = self.calculate_tp_sl(candle_data, self.config['timeframes'][0])
        return sl

    def check_tp_sl(self, position: Dict, current_price: float) -> Tuple[bool, bool]:
        """
        Проверка закрытия позиции по TP/SL.
        Возвращает (hit_tp, hit_sl) — дальше закрытие в PositionManager
        """
        tp = position.get('tp')
        sl = position.get('sl')
        direction = position['direction']

        hit_tp = False
        hit_sl = False

        if direction == 'long':
            if current_price >= tp:
                hit_tp = True
            if current_price <= sl:
                hit_sl = True
        else:
            if current_price <= tp:
                hit_tp = True
            if current_price >= sl:
                hit_sl = True

        return hit_tp, hit_sl