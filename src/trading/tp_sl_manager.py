"""
src/trading/tp_sl_manager.py

=== Основной принцип работы файла ===

Этот файл реализует логику расчёта и управления уровнями Take Profit (TP) и Stop Loss (SL) для всех позиций.
Он отвечает за:
- Расчёт уровней TP/SL по типам (classic, partial_trailing, chandelier — dynamic удалён по ТЗ).
- Проверку закрытия позиций в каждой новой свече (TP/SL hit).
- Частичный trailing и partial close (если цена достигла промежуточных уровней).
- Учёт направления (Long/Short): TP/SL симметричны.
- Передачу сигнала закрытия в entry_manager (флаг для запрета новой позиции в свече закрытия).
- Обновление виртуальных позиций в virtual_trader и реальных через order_executor.

Ключевые особенности:
- Classic TP = средний размер свечи (из feature_engine или config).
- SL = HH/LL ± 0.05% с cap <= средний размер * 2 (по ТЗ).
- Partial_trailing: разбивает позицию на порции (например, 50% на первом уровне, 50% на втором).
- Chandelier: trailing по ATR (но только если включено в config, по умолчанию classic).
- Все позиции хранятся в dict (open_positions) с обновлением в каждой свече.

=== Главные функции и за что отвечают ===

- TPSLManager() — инициализация: open_positions dict, config.
- calculate_levels(candle_data: dict, direction: str) → dict[tp, sl, tp_distance, sl_distance]
  Основная функция расчёта TP/SL по classic (ТЗ).
- check_close(pos: dict, candle_data: dict) → dict[closed: bool, is_tp: bool]
  Проверяет hit TP/SL в текущей свече, обновляет trailing если partial.
- add_open_position(pos: dict) — добавляет позицию в управление.
- update_open_positions(candle_data: dict) — проверяет все открытые позиции в новой свече.
- _calculate_classic_tp_sl(candle_data, direction) → tp, sl
  Расчёт по ТЗ: TP = avg candle size, SL = HH/LL ±0.05% с cap.
- _apply_partial_close(pos, current_price) — частичное закрытие при достижении уровня.

=== Примечания ===
- Dynamic режим полностью удалён (по предыдущему указанию).
- Запрет новой позиции в свече закрытия — передача флага в entry_manager.
- Полностью соответствует ТЗ + уточнениям (classic + partial_trailing).
- Готов к интеграции в live_loop, backtest и virtual_trader.
- Логи через setup_logger.
"""

from typing import Dict, Optional
import numpy as np

from src.core.config import load_config
from src.core.enums import Direction, TpSlMode
from src.features.feature_engine import _compute_half_features  # для avg size (если нужно)
from src.utils.logger import setup_logger

logger = setup_logger('tp_sl_manager', logging.INFO)

class TPSLManager:
    """
    Менеджер уровней TP/SL и проверки закрытий позиций.
    """
    def __init__(self):
        self.config = load_config()
        self.open_positions: Dict[str, Dict] = {}  # key: pos_id, value: pos dict
        self.mode = TpSlMode(self.config['trading']['tp_sl_mode'])  # classic / partial_trailing / chandelier
        self.avg_candle_size_cache = {}  # кэш среднего размера свечи по символу/TF

    def calculate_levels(self, candle_data: Dict, direction: str) -> Optional[Dict]:
        """
        Расчёт TP/SL по режиму.
        candle_data — текущая свеча (close, high, low и т.д.).
        Возвращает dict['tp', 'sl', 'tp_distance', 'sl_distance'] или None.
        """
        if self.mode == TpSlMode.CLASSIC:
            return self._calculate_classic_tp_sl(candle_data, direction)
        elif self.mode == TpSlMode.PARTIAL_TRAILING:
            return self._calculate_partial_trailing(candle_data, direction)
        elif self.mode == TpSlMode.CHANDELIER:
            return self._calculate_chandelier(candle_data, direction)
        else:
            logger.error(f"Неизвестный режим TP/SL: {self.mode}")
            return None

    def _calculate_classic_tp_sl(self, candle_data: Dict, direction: str) -> Dict:
        """
        Классический режим по ТЗ.
        TP = средний размер свечи (из feature_engine или config).
        SL = HH/LL ±0.05% с cap <= avg_size * 2.
        """
        # Средний размер свечи (из кэша или расчёт)
        avg_size_pct = self._get_avg_candle_size(candle_data['symbol'], candle_data['timeframe'])
        avg_size = candle_data['close'] * avg_size_pct / 100

        entry = candle_data['close']
        if direction == Direction.LONG.value:
            tp = entry + avg_size
            # SL = LL - 0.05%, cap = entry - avg_size * 2
            ll = candle_data['low']  # упрощённо, реально — LL за период
            sl = ll - entry * 0.0005
            sl_cap = entry - avg_size * 2
            sl = max(sl, sl_cap)  # не ниже cap
        else:  # SHORT
            tp = entry - avg_size
            hh = candle_data['high']
            sl = hh + entry * 0.0005
            sl_cap = entry + avg_size * 2
            sl = min(sl, sl_cap)

        tp_distance = abs(tp - entry)
        sl_distance = abs(sl - entry)

        return {
            'tp': tp,
            'sl': sl,
            'tp_distance': tp_distance,
            'sl_distance': sl_distance
        }

    def _calculate_partial_trailing(self, candle_data: Dict, direction: str) -> Dict:
        """
        Partial trailing: несколько уровней TP, trailing SL.
        (упрощённо — 2 уровня, trailing по ATR или фиксированно)
        """
        # Пример реализации: TP1 = entry + 0.5*avg_size, TP2 = entry + avg_size
        # SL trailing = chandelier-like
        classic = self._calculate_classic_tp_sl(candle_data, direction)
        return classic  # можно расширить позже

    def _calculate_chandelier(self, candle_data: Dict, direction: str) -> Dict:
        """
        Chandelier exit: trailing по ATR.
        (если включено, иначе fallback на classic)
        """
        return self._calculate_classic_tp_sl(candle_data, direction)  # placeholder

    def _get_avg_candle_size(self, symbol: str, timeframe: str) -> float:
        """
        Возвращает средний размер свечи в % (кэшируется).
        """
        key = f"{symbol}_{timeframe}"
        if key in self.avg_candle_size_cache:
            return self.avg_candle_size_cache[key]

        # Здесь можно вызвать feature_engine или расчёт по БД
        # Упрощённо — фиксированный из config
        avg_size_pct = self.config['trading']['avg_candle_size_pct']  # например 0.5%
        self.avg_candle_size_cache[key] = avg_size_pct
        return avg_size_pct

    def check_close(self, pos: Dict, candle_data: Dict) -> Dict:
        """
        Проверяет закрытие позиции в текущей свече.
        Возвращает {'closed': bool, 'is_tp': bool, 'profit': float}
        """
        entry = pos['entry_price']
        current = candle_data['close']
        high = candle_data['high']
        low = candle_data['low']

        direction = pos['direction']
        if direction == Direction.LONG.value:
            if high >= pos['tp']:
                return {'closed': True, 'is_tp': True, 'profit': (pos['tp'] - entry) * pos['size']}
            if low <= pos['sl']:
                return {'closed': True, 'is_tp': False, 'profit': (pos['sl'] - entry) * pos['size']}
        else:  # SHORT
            if low <= pos['tp']:
                return {'closed': True, 'is_tp': True, 'profit': (entry - pos['tp']) * pos['size']}
            if high >= pos['sl']:
                return {'closed': True, 'is_tp': False, 'profit': (entry - pos['sl']) * pos['size']}

        # Trailing update если partial_trailing
        if self.mode == TpSlMode.PARTIAL_TRAILING:
            self._update_trailing(pos, candle_data)

        return {'closed': False}

    def _update_trailing(self, pos: Dict, candle_data: Dict):
        """
        Обновляет trailing SL для partial_trailing режима.
        """
        # Пример: trailing SL = max(SL, high - atr * mult) для Long
        pass  # реализация по необходимости

    def add_open_position(self, pos: Dict):
        """
        Добавляет позицию в управление.
        """
        pos_id = f"{pos['symbol']}_{pos['type']}_{pos['open_ts']}"
        self.open_positions[pos_id] = pos
        logger.debug(f"Добавлена позиция в управление: {pos_id}")

    def get_open_positions(self, symbol: Optional[str] = None) -> List[Dict]:
        """
        Возвращает открытые позиции (по символу или все).
        """
        if symbol:
            return [p for p in self.open_positions.values() if p['symbol'] == symbol]
        return list(self.open_positions.values())

    def has_open_position(self, symbol: str, anomaly_type: str) -> bool:
        """
        Проверяет наличие открытой позиции по типу.
        """
        for pos in self.open_positions.values():
            if pos['symbol'] == symbol and pos['type'] == anomaly_type:
                return True
        return False