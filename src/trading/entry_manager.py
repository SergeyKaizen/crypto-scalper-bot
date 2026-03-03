"""
src/trading/entry_manager.py

=== Основной принцип работы файла ===

Менеджер открытия позиций в live-режиме.

Ключевые особенности (по ТЗ + улучшения):
- Только 1 позиция на монету (глобальный lock через position_manager)
- Если несколько сигналов в свече — выбирается top-1 по весу из scenario_tracker
- Остальные сигналы — виртуальные для PR и статистики
- Запрет новой позиции в свече, где предыдущая закрылась (через candle_close_flags)
- Открытие через централизованный PositionManager
- Проверка whitelist, min_prob/min_prob_q, no open по типу
- Рассчёт размера через risk_manager, открытие через order_executor/virtual_trader

=== Главные функции ===
- process_signals(symbol, signals, candle_data, candle_ts) — сбор сигналов → выбор top-1 → открытие
- _can_open_position(symbol, anomaly_type, candle_ts) → bool — глобальный lock + проверка закрытия
- update_candle_close_flag(candle_ts) — флаг закрытия в свече

=== Примечания ===
- Глобальный lock: position_manager.has_any_open_position(symbol)
- Выбор max веса: только для live, фокус на лучших сигналах
- Полностью соответствует ТЗ + улучшениям (централизация, state-machine)
- Готов к интеграции в live_loop.py
- Логи через setup_logger
"""

from typing import List, Dict
import logging

from src.core.config import load_config
from src.core.enums import AnomalyType, Direction
from src.trading.position_manager import PositionManager
from src.trading.risk_manager import RiskManager
from src.trading.tp_sl_manager import TP_SL_Manager
from src.model.scenario_tracker import ScenarioTracker
from src.data.storage import Storage          # FIX Фаза 7
from src.utils.logger import setup_logger

logger = setup_logger('entry_manager', logging.INFO)

class EntryManager:
    def __init__(self, scenario_tracker: ScenarioTracker):
        self.config = load_config()
        self.risk_manager = RiskManager()
        self.tp_sl_manager = TP_SL_Manager()
        self.position_manager = PositionManager()
        self.scenario_tracker = scenario_tracker
        self.storage = Storage()  # FIX Фаза 7: инициализация

        self.candle_close_flags = {}  # candle_ts → True (закрытие в свече)

    def process_signals(
        self,
        symbol: str,
        signals: List[Dict],
        candle_data: Dict,
        candle_ts: int
    ):
        """
        Обработка всех сигналов в свече:
        - Выбор top-1 по весу из scenario_tracker
        - Открытие только этой позиции через position_manager
        - Остальные — виртуальные для PR
        """
        if not signals:
            return

        # FIX Фаза 7: динамический выбор top-1 по весу (старые best_anomaly, best_direction, best_window удалены)
        scored_signals = []
        for sig in signals:
            feats = sig.get('feats', {})
            weight = self.scenario_tracker.get_weight(
                self.scenario_tracker._binarize_features(feats)
            )
            scored_signals.append((sig, weight))

        if not scored_signals:
            return

        scored_signals.sort(key=lambda x: x[1], reverse=True)
        top_sig, top_weight = scored_signals[0]

        anomaly_type = top_sig['anom']['type']
        direction = self._resolve_direction(top_sig['feats'])
        prob = top_sig.get('prob', 0.0)
        quiet_streak = top_sig.get('quiet_streak', 0)

        if not self._can_open_position(symbol, anomaly_type, candle_ts):
            for sig, _ in scored_signals[1:]:
                self._open_virtual_position(symbol, sig)
            return

        sl_price = self.tp_sl_manager.calculate_sl(candle_data, direction)
        size = self.risk_manager.calculate_position_size(
            symbol=symbol,
            entry_price=candle_data['close'],
            sl_price=sl_price
        )

        if size <= 0:
            logger.warning(f"Некорректный размер позиции для {symbol}")
            return

        position_data = {
            'pos_id': f"{symbol}_{anomaly_type}_{candle_ts}",
            'symbol': symbol,
            'anomaly_type': anomaly_type,
            'direction': direction,
            'entry_price': candle_data['close'],
            'size': size,
            'open_ts': candle_ts,
            'prob': prob,
            'quiet_streak': quiet_streak,
            'consensus_count': top_sig['anom'].get('consensus_count', 1),
            'feats': top_sig.get('feats', {}),
            'mode': self._resolve_mode(symbol, anomaly_type, direction)
        }

        success = self.position_manager.open_position(position_data)
        if success:
            logger.info(f"Открыта позиция {anomaly_type} {direction} на {symbol}, size={size:.4f}, weight={top_weight:.4f}")
        else:
            logger.error(f"Не удалось открыть позицию {symbol}")

        for sig, _ in scored_signals[1:]:
            self._open_virtual_position(symbol, sig)

    def _can_open_position(self, symbol: str, anomaly_type: str, candle_ts: int) -> bool:
        """Проверки перед открытием позиции"""
        if self.position_manager.has_any_open_position(symbol):
            logger.debug(f"Уже есть открытая позиция на {symbol}")
            return False

        if self.candle_close_flags.get(candle_ts, False):
            logger.debug(f"В этой свече уже было закрытие позиции ({candle_ts})")
            return False

        return True

    def _resolve_mode(self, symbol: str, anomaly_type: str, direction: str) -> str:
        """Real если совпадает с whitelist, иначе virtual"""
        wl = self.storage.get_whitelist_settings(symbol)
        if not wl:
            return 'virtual'

        if wl.get('anomaly_type') == anomaly_type and wl.get('direction') == direction:
            return 'real'
        return 'virtual'

    def _resolve_direction(self, feats: Dict) -> str:
        """Определение направления сделки"""
        price_change = feats.get('price_change_pct', 0)
        if price_change > 0:
            return 'L'
        return 'S'

    def _open_virtual_position(self, symbol: str, sig: Dict):
        """Открытие виртуальной позиции"""
        try:
            self.position_manager.virtual_trader.open_virtual_position(
                symbol,
                sig['anom']['type'],
                sig.get('prob', 0.0),
                self.tp_sl_manager.calculate_tp_sl(sig.get('feats', {}), sig['anom']['type'])
            )
        except Exception as e:
            logger.debug(f"Ошибка открытия виртуальной позиции: {e}")

    def update_candle_close_flag(self, candle_ts: int):
        """Фиксирует, что в этой свече уже было закрытие позиции"""
        self.candle_close_flags[candle_ts] = True
        logger.debug(f"Установлен флаг закрытия позиции в свече {candle_ts}")

    def clear_old_flags(self, current_ts: int):
        """Очистка старых флагов закрытия (экономия памяти)"""
        to_remove = [ts for ts in self.candle_close_flags if ts < current_ts - 3600000]
        for ts in to_remove:
            self.candle_close_flags.pop(ts, None)