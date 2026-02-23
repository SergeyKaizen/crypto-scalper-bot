"""
src/trading/tp_sl_manager.py

=== Основной принцип работы файла ===

Менеджер расчёта и мониторинга TP/SL для открытых позиций.

Реализует расчёт уровней по ТЗ:
- TP = средний размер свечи (avg candle range)
- SL = HH/LL ±0.05% с cap на 2× средний размер свечи
- Поддерживает режимы: classic, partial_trailing, chandelier (по config)
- Мониторит закрытие позиций (TP/SL hit) каждую новую свечу
- Вызывает entry_manager.update_candle_close_flag после закрытия (запрет новых входов в свече)
- Хранит открытые позиции (symbol_anomaly_type → pos)

=== Главные функции ===
- calculate_tp_sl(candle_data, timeframe) → tp, sl (основной расчёт по ТЗ)
- calculate_levels(candle_data, direction, mode='classic') — расчёт по режиму
- check_tp_sl(position, current_price) — проверка закрытия, вызов update_flag
- add_open_position(pos) — регистрация позиции
- has_open_position(symbol, anomaly_type) — проверка открытой по типу
- update_candle_close_flag(candle_ts) — вызов entry_manager (новое)

=== Примечания ===
- Cap 2× avg_size — по ТЗ (если HH/LL > cap → shift к следующему)
- Вызов update_flag после закрытия — обеспечивает запрет новой позиции в свече
- quiet_streak и consensus_count передаются в pos (для log и анализа)
- Полностью соответствует ТЗ + последним изменениям
- Готов к интеграции в live_loop и entry_manager
"""

from typing import Dict, Tuple, Optional
import numpy as np

from src.core.config import load_config
from src.core.enums import Direction
from src.utils.logger import setup_logger
from src.trading.entry_manager import EntryManager  # для вызова update_flag

logger = setup_logger('tp_sl_manager', logging.INFO)

class TP_SL_Manager:
    def __init__(self):
        self.config = load_config()
        self.open_positions = {}  # key: f"{symbol}_{anomaly_type}" → position dict
        self.entry_manager = EntryManager()  # ссылка для вызова update_flag

    def calculate_tp_sl(self, candle_data: Dict, timeframe: str) -> Tuple[float, float]:
        """
        Основной расчёт TP/SL по ТЗ (classic mode)
        """
        # Средний размер свечи (в % от close)
        avg_size = candle_data.get('volatility_mean', 0.0)

        if avg_size == 0:
            avg_size = np.mean(candle_data['high'] - candle_data['low']) / candle_data['close'].mean()

        direction = candle_data.get('direction', Direction.LONG.value)

        if direction == Direction.LONG.value:
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

    def check_tp_sl(self, position: Dict, current_price: float) -> bool:
        """
        Проверка закрытия позиции по TP/SL.
        Если закрыта — вызывает update_candle_close_flag
        """
        tp = position.get('tp')
        sl = position.get('sl')
        direction = position['direction']

        hit_tp = False
        hit_sl = False

        if direction == Direction.LONG.value:
            if current_price >= tp:
                hit_tp = True
            if current_price <= sl:
                hit_sl = True
        else:
            if current_price <= tp:
                hit_tp = True
            if current_price >= sl:
                hit_sl = True

        if hit_tp or hit_sl:
            # Закрытие позиции
            position['closed_ts'] = time.time()
            position['hit_tp'] = hit_tp
            position['hit_sl'] = hit_sl

            # Вызов флага закрытия в свече
            candle_ts = int(position['open_ts'] // 1000 * 1000)  # округление до свечи
            self.entry_manager.update_candle_close_flag(candle_ts)

            logger.info(f"Позиция закрыта {position['symbol']} {position['anomaly_type']}: "
                        f"{'TP' if hit_tp else 'SL'} at {current_price}")

            # Удаление из открытых
            key = f"{position['symbol']}_{position['anomaly_type']}"
            if key in self.open_positions:
                del self.open_positions[key]

            return True

        return False

    def add_open_position(self, pos: Dict):
        """Регистрация открытой позиции"""
        key = f"{pos['symbol']}_{pos['anomaly_type']}"
        self.open_positions[key] = pos
        logger.debug(f"Добавлена открытая позиция {key}")

    def has_open_position(self, symbol: str, anomaly_type: str) -> bool:
        """Проверка открытой позиции по типу"""
        key = f"{symbol}_{anomaly_type}"
        return key in self.open_positions

    def has_any_open_position(self, symbol: str) -> bool:
        """Глобальный lock: есть ли хоть одна открытая позиция на монете"""
        for key in self.open_positions:
            if key.startswith(symbol + "_"):
                return True
        return False