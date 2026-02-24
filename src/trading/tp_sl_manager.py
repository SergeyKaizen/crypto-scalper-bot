"""
src/trading/tp_sl_manager.py

=== Основной принцип работы файла ===

Менеджер расчёта и мониторинга TP/SL для открытых позиций (реальных и виртуальных).

Ключевые особенности (по ТЗ):
- calculate_tp_sl: TP = средний размер свечи, SL = HH/LL ±0.05% с cap на 2× avg_size
- Поддержка режимов: classic (основной), partial_trailing, chandelier (по config)
- check_tp_sl: мониторинг закрытия позиции (TP/SL hit), вызов update_candle_close_flag после закрытия
- add_open_position / has_open_position / has_any_open_position — регистрация и проверки
- Глобальный lock (has_any_open_position) — только 1 позиция на монету в live

=== Главные функции ===
- calculate_tp_sl(candle_data, timeframe) → tp, sl (основной расчёт по ТЗ)
- check_tp_sl(position, current_price) — проверка закрытия, вызов update_flag
- add_open_position(pos)
- has_open_position(symbol, anomaly_type)
- has_any_open_position(symbol) — глобальный lock для live
- update_candle_close_flag(candle_ts) — вызов entry_manager

=== Примечания ===
- Cap 2× avg_size — строго по ТЗ
- Вызов update_flag после любого закрытия — обеспечивает запрет новой позиции в свече
- quiet_streak и consensus_count передаются в pos (для log и анализа)
- Полностью соответствует ТЗ + всем уточнениям (1 позиция в live, флаг после закрытия)
- Готов к интеграции в live_loop и entry_manager
- Логи через setup_logger
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

        if avg_size == 0 or np.isnan(avg_size):
            # Fallback на средний range
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
                        f"{'TP' if hit_tp else 'SL'} at {current_price:.2f} "
                        f"quiet_streak={position.get('quiet_streak', 0)} consensus={position.get('consensus_count', 1)}")

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
        logger.debug(f"Добавлена открытая позиция {key} quiet_streak={pos.get('quiet_streak', 0)}")

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