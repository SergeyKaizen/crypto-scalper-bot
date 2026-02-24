"""
src/trading/virtual_trader.py

=== Основной принцип работы файла ===

Виртуальный трейдер для симуляции позиций и расчёта PR без реальных ордеров.

Ключевые особенности:
- open_position: регистрация виртуальной позиции
- update_positions: мониторинг и закрытие по TP/SL (через tp_sl_manager)
- _close_position: расчёт net_pnl с комиссией, вызов scenario_tracker.add_scenario
- get_balance, get_pnl — статистика виртуальной торговли
- Поддержка extra в position (quiet_streak, consensus_count) для анализа

=== Главные функции ===
- open_position(pos: dict)
- update_positions(candle_data: dict)
- _close_position(pos: dict, hit_tp: bool)
- get_balance() → float
- get_pnl() → float

=== Примечания ===
- Комиссия учитывается в net_pnl (entry + exit)
- Вызов scenario_tracker.add_scenario после закрытия (для статистики)
- Полностью соответствует ТЗ + последним изменениям (extra в position)
- Готов к интеграции в live_loop и entry_manager
- Логи через setup_logger
"""

from typing import Dict
import logging
import time

from src.core.config import load_config
from src.trading.tp_sl_manager import TP_SL_Manager
from src.model.scenario_tracker import ScenarioTracker
from src.utils.logger import setup_logger

logger = setup_logger('virtual_trader', logging.INFO)

class VirtualTrader:
    def __init__(self):
        self.config = load_config()
        self.tp_sl_manager = TP_SL_Manager()
        self.scenario_tracker = ScenarioTracker()
        self.positions = {}  # pos_id → position
        self.balance = self.config.get('deposit', 10000.0)  # начальный депозит
        self.commission_rate = self.config['trading']['commission']

    def open_position(self, pos: Dict):
        """
        Открытие виртуальной позиции
        """
        pos_id = f"virtual_{pos['symbol']}_{pos['anomaly_type']}_{int(pos['open_ts'])}"
        pos['pos_id'] = pos_id
        pos['entry_commission'] = pos['size'] * pos['entry_price'] * self.commission_rate

        self.positions[pos_id] = pos
        logger.debug(f"[VIRTUAL] Открыта позиция {pos['symbol']} {pos['anomaly_type']} "
                     f"size={pos['size']:.4f} entry={pos['entry_price']:.2f} "
                     f"quiet_streak={pos.get('quiet_streak', 0)} consensus={pos.get('consensus_count', 1)}")

    def update_positions(self, candle_data: Dict):
        """
        Обновление всех виртуальных позиций по новой свече
        """
        current_price = candle_data['close']
        for pos_id, pos in list(self.positions.items()):
            if self.tp_sl_manager.check_tp_sl(pos, current_price):
                hit_tp = pos.get('hit_tp', False)
                self._close_position(pos, hit_tp)

    def _close_position(self, pos: Dict, hit_tp: bool):
        """
        Закрытие виртуальной позиции, расчёт net_pnl, вызов scenario_tracker
        """
        exit_price = pos['tp'] if hit_tp else pos['sl']
        gross_pnl = (exit_price - pos['entry_price']) * pos['size'] if pos['direction'] == 'long' else \
                    (pos['entry_price'] - exit_price) * pos['size']

        exit_commission = pos['size'] * exit_price * self.commission_rate
        net_pnl = gross_pnl - pos['entry_commission'] - exit_commission

        self.balance += net_pnl

        # Вызов scenario_tracker для статистики
        outcome = 1 if hit_tp else 0
        self.scenario_tracker.add_scenario(pos, outcome)

        logger.info(f"[VIRTUAL] Закрыта позиция {pos['symbol']} {pos['anomaly_type']} "
                    f"{'TP' if hit_tp else 'SL'} at {exit_price:.2f} net_pnl={net_pnl:.2f} "
                    f"balance={self.balance:.2f} quiet_streak={pos.get('quiet_streak', 0)}")

        del self.positions[pos['pos_id']]

    def get_balance(self) -> float:
        return self.balance

    def get_pnl(self) -> float:
        return self.balance - self.config.get('deposit', 10000.0)