"""
src/trading/position_manager.py

=== Основной принцип работы файла ===

Централизованный менеджер позиций.

Ключевые особенности:
- State-machine: PositionState (OPEN, PENDING_CLOSE, CLOSED)
- Открытие: через entry_manager → position_manager.open_position
- Мониторинг и закрытие: через tp_sl_manager → position_manager.check_and_close
- Проверки: глобальный lock (1 позиция на монету)
- Интеграция: risk_manager (новая формула sizing), order_executor/virtual_trader
"""

from enum import Enum
from typing import Dict, Optional
import time
import logging

from src.core.config import load_config
from src.trading.risk_manager import RiskManager
from src.trading.order_executor import OrderExecutor
from src.trading.virtual_trader import VirtualTrader
from src.trading.tp_sl_manager import TP_SL_Manager
from src.model.scenario_tracker import ScenarioTracker
from src.utils.logger import setup_logger

logger = setup_logger('position_manager', logging.INFO)

class PositionState(Enum):
    OPEN = "OPEN"
    PENDING_CLOSE = "PENDING_CLOSE"
    CLOSED = "CLOSED"

class PositionManager:
    def __init__(self):
        self.config = load_config()
        self.risk_manager = RiskManager()
        self.order_executor = OrderExecutor()
        self.virtual_trader = VirtualTrader()
        self.tp_sl_manager = TP_SL_Manager()
        self.scenario_tracker = ScenarioTracker()

        self.positions = {}  # pos_id → {'data': dict, 'state': PositionState, 'open_time': float}

    def open_position(self, pos_data: Dict):
        """
        Открытие позиции (вызывается из entry_manager)
        """
        pos_id = pos_data['pos_id']
        symbol = pos_data['symbol']
        direction = pos_data['direction']
        size = pos_data['size']
        entry_price = pos_data['entry_price']
        tp = pos_data.get('tp')
        sl = pos_data.get('sl')

        if self.has_any_open_position(symbol):
            logger.warning(f"Глобальный lock: уже есть открытая позиция на {symbol}")
            return False

        if size <= 0:
            if tp and sl:
                size = self.risk_manager.calculate_position_size(symbol, entry_price, tp, sl)
            else:
                logger.error(f"Нет TP/SL для расчёта размера {symbol}")
                return False

        if size <= 0:
            logger.error(f"Некорректный размер позиции для {symbol}")
            return False

        if pos_data.get('mode') == 'real':
            order_id = self.order_executor.place_order(pos_data)
            if not order_id:
                return False
            pos_data['order_id'] = order_id
        else:
            self.virtual_trader.open_position(pos_data)

        self.positions[pos_id] = {
            'data': pos_data,
            'state': PositionState.OPEN,
            'open_time': time.time()
        }

        self.tp_sl_manager.add_open_position(pos_data)

        logger.info(f"Позиция открыта: {pos_id}, {direction} {symbol}, size={size:.4f}, state=OPEN")
        return True

    def check_and_close(self, current_price: float, current_time: float):
        """
        Проверка и закрытие всех открытых позиций
        """
        for pos_id, pos_info in list(self.positions.items()):
            if pos_info['state'] != PositionState.OPEN:
                continue

            pos_data = pos_info['data']
            symbol = pos_data['symbol']
            tp = pos_data.get('tp')
            sl = pos_data.get('sl')
            direction = pos_data['direction']

            hit_tp = (current_price >= tp) if direction == 'L' else (current_price <= tp)
            hit_sl = (current_price <= sl) if direction == 'L' else (current_price >= sl)

            if hit_tp or hit_sl:
                if pos_data.get('mode') == 'real':
                    self.order_executor.close_position(pos_id)
                else:
                    self.virtual_trader.close_position(pos_id)

                net_pnl = self._calculate_net_pnl(pos_data, current_price, hit_tp)
                self.risk_manager.update_deposit(net_pnl)

                outcome = 1 if hit_tp else 0
                self.scenario_tracker.add_scenario(pos_data['feats'], outcome, net_pnl)

                pos_info['state'] = PositionState.CLOSED
                pos_info['close_time'] = current_time
                pos_info['net_pnl'] = net_pnl

                logger.info(f"Позиция закрыта: {pos_id}, {'TP' if hit_tp else 'SL'}, net_pnl={net_pnl:.2f}")

    def has_any_open_position(self, symbol: str) -> bool:
        """Глобальный lock: есть ли открытая позиция на монете"""
        for pos_info in self.positions.values():
            if pos_info['state'] == PositionState.OPEN and pos_info['data']['symbol'] == symbol:
                return True
        return False

    def get_position_state(self, pos_id: str) -> PositionState:
        return self.positions.get(pos_id, {}).get('state', PositionState.CLOSED)

    def _calculate_net_pnl(self, pos_data: Dict, exit_price: float, hit_tp: bool) -> float:
        """Расчёт net_pnl (комиссия учтена)"""
        entry_price = pos_data['entry_price']
        size = pos_data['size']
        direction = pos_data['direction']
        commission_rate = self.config['trading']['commission']

        gross_pnl = (exit_price - entry_price) * size if direction == 'L' else (entry_price - exit_price) * size
        entry_comm = size * entry_price * commission_rate
        exit_comm = size * exit_price * commission_rate
        net_pnl = gross_pnl - entry_comm - exit_comm

        return net_pnl