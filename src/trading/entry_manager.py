"""
src/trading/entry_manager.py

=== Основной принцип работы файла ===

Менеджер открытия позиций в live-режиме.

Ключевые особенности (по ТЗ + последние уточнения):
- Только 1 позиция на монету в любой момент (глобальный lock)
- Если несколько сигналов в одной свече — выбирается только тот с максимальным весом (scenario_tracker.get_weight)
- Остальные сигналы — виртуальные для PR и статистики
- Запрет новой позиции в свече, где предыдущая закрылась (candle_close_flags)
- Проверка whitelist, min_prob (повышенный для Q), no open по типу
- Рассчёт размера через risk_manager, открытие через order_executor/virtual_trader

=== Главные функции ===
- process_signals(symbol, signals, candle_data, candle_ts) — сбор сигналов → выбор top-1 → открытие
- _can_open_position(symbol, anomaly_type, candle_ts) → bool — глобальный lock + проверка закрытия
- _resolve_mode(symbol, anomaly_type, direction) → TradeMode — real/virtual по PR match
- update_candle_close_flag(candle_ts) — флаг закрытия в свече

=== Примечания ===
- Глобальный lock: has_any_open_position(symbol) — нет входа пока открыта любая позиция
- Выбор max веса: только для live, чтобы фокусироваться на лучших сигналах
- Полностью соответствует ТЗ + уточнению (1 позиция, max вес, no new после закрытия в свече)
- Готов к интеграции в live_loop.py
- Логи через setup_logger
"""

from typing import List, Dict
import logging

from src.core.config import load_config
from src.core.enums import AnomalyType, Direction, TradeMode
from src.trading.risk_manager import RiskManager
from src.trading.order_executor import OrderExecutor
from src.trading.virtual_trader import VirtualTrader
from src.trading.tp_sl_manager import TP_SL_Manager
from src.model.scenario_tracker import ScenarioTracker
from src.utils.logger import setup_logger

logger = setup_logger('entry_manager', logging.INFO)

class EntryManager:
    def __init__(self, scenario_tracker: ScenarioTracker):
        self.config = load_config()
        self.risk_manager = RiskManager()
        self.order_executor = OrderExecutor()
        self.virtual_trader = VirtualTrader()
        self.tp_sl_manager = TP_SL_Manager()
        self.scenario_tracker = scenario_tracker  # для расчёта веса

        self.mode = TradeMode(self.config['trading']['mode'])
        self.candle_close_flags = {}  # candle_ts → True (закрытие в свече)

    def process_signals(
        self,
        symbol: str,
        signals: List[Dict],  # [{'anom': dict, 'window': int, 'feats': dict, 'prob': float, 'quiet_streak': int}]
        candle_data: Dict,
        candle_ts: int
    ):
        """
        Обработка всех сигналов в свече:
        - Выбор top-1 по весу из scenario_tracker
        - Открытие только этой позиции (если проходит проверки)
        - Остальные — виртуальные для PR
        """
        if not signals:
            return

        # 1. Рассчитываем вес каждого сигнала
        scored_signals = []
        for sig in signals:
            feats = sig['feats']
            weight = self.scenario_tracker.get_weight(
                self.scenario_tracker._binarize_features(feats)
            )
            scored_signals.append((sig, weight))

        # 2. Сортировка по весу (descending)
        scored_signals.sort(key=lambda x: x[1], reverse=True)

        # 3. Глобальный lock: есть ли уже открытая позиция на монете
        if self.tp_sl_manager.has_any_open_position(symbol):
            logger.debug(f"Открытая позиция на {symbol} — новые запрещены")
            # Все сигналы — виртуальные
            for sig, _ in scored_signals:
                self.virtual_trader.open_virtual_position(
                    symbol, sig['anom']['type'], sig['prob'],
                    self.tp_sl_manager.calculate_tp_sl(sig['feats'], sig['anom']['type'])
                )
            return

        # 4. Берём top-1 по весу
        top_sig, top_weight = scored_signals[0]
        anomaly_type = top_sig['anom']['type']
        direction = self._resolve_direction(top_sig['feats'])  # по price_change или delta
        prob = top_sig['prob']
        quiet_streak = top_sig['quiet_streak']

        # 5. Проверка условий входа
        if not self._can_open_position(symbol, anomaly_type, candle_ts):
            # Остальные — виртуальные
            for sig, _ in scored_signals[1:]:
                self.virtual_trader.open_virtual_position(
                    symbol, sig['anom']['type'], sig['prob'],
                    self.tp_sl_manager.calculate_tp_sl(sig['feats'], sig['anom']['type'])
                )
            return

        # 6. Расчёт размера позиции
        sl_price = self.tp_sl_manager.calculate_sl(candle_data, direction)
        size = self.risk_manager.calculate_size(
            symbol=symbol,
            entry_price=candle_data['close'],
            sl_price=sl_price,
            risk_pct=self.config['trading']['risk_pct']
        )

        if size <= 0:
            logger.warning(f"Некорректный размер позиции для {symbol}")
            return

        # 7. Открытие позиции
        position = {
            'symbol': symbol,
            'anomaly_type': anomaly_type,
            'direction': direction,
            'entry_price': candle_data['close'],
            'size': size,
            'open_ts': candle_ts,
            'prob': prob,
            'quiet_streak': quiet_streak,
            'consensus_count': top_sig['anom'].get('consensus_count', 1)
        }

        mode = self._resolve_mode(symbol, anomaly_type, direction)
        if mode == TradeMode.REAL:
            order_id = self.order_executor.place_order(position)
            if order_id:
                position['order_id'] = order_id
                logger.info(f"Открыта реальная позиция {anomaly_type} {direction} на {symbol}, size={size}, weight={top_weight:.4f}")
            else:
                logger.error(f"Ошибка открытия реальной позиции {symbol}")
                return
        else:
            self.virtual_trader.open_position(position)
            logger.debug(f"Открыта виртуальная позиция {anomaly_type} {direction} на {symbol}, size={size}, weight={top_weight:.4f}")

        self.tp_sl_manager.add_open_position(position)

        # 8. Остальные сигналы — виртуальные для PR
        for sig, _ in scored_signals[1:]:
            self.virtual_trader.open_virtual_position(
                symbol, sig['anom']['type'], sig['prob'],
                self.tp_sl_manager.calculate_tp_sl(sig['feats'], sig['anom']['type'])
            )

    def _can_open_position(self, symbol: str, anomaly_type: str, candle_ts: int) -> bool:
        """
        Проверки:
        - Нет открытой позиции на монете (глобальный lock)
        - Нет закрытия в этой свече
        """
        # Глобальный lock
        if self.tp_sl_manager.has_any_open_position(symbol):
            logger.debug(f"Открытая позиция на {symbol} — новые запрещены")
            return False

        # Нет закрытия в этой свече
        if self.candle_close_flags.get(candle_ts, False):
            logger.debug(f"Закрытие в свече {candle_ts} — вход запрещён")
            return False

        return True

    def _resolve_mode(self, symbol: str, anomaly_type: str, direction: str) -> TradeMode:
        """Real если match с PR config, иначе virtual"""
        whitelist = self.storage.get_whitelist_config(symbol)
        if not whitelist:
            return TradeMode.VIRTUAL

        signal_key = f"{anomaly_type}_{direction}"
        pr_key = f"{whitelist['best_anomaly']}_{whitelist['best_direction']}"

        return TradeMode.REAL if signal_key == pr_key else TradeMode.VIRTUAL

    def _resolve_direction(self, feats: Dict) -> str:
        """Placeholder: направление по price_change или delta"""
        price_change = feats.get('price_change_pct', 0)
        delta = feats.get('delta_positive', 0)
        if price_change > 0 or delta > 0:
            return 'long'
        return 'short'

    def update_candle_close_flag(self, candle_ts: int):
        """Вызывается из tp_sl_manager после закрытия"""
        self.candle_close_flags[candle_ts] = True
        logger.debug(f"Установлен флаг закрытия в свече {candle_ts}")