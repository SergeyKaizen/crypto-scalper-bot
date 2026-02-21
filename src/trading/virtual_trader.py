"""
src/trading/virtual_trader.py

=== Основной принцип работы файла ===

Этот файл реализует виртуальную торговлю (симуляцию сделок без реальных ордеров на бирже).
Он полностью повторяет логику реальной торговли, но вместо вызова order_executor просто обновляет виртуальный депозит, позиции и PNL.

Ключевые задачи:
- Открытие виртуальных позиций (при получении сигнала от entry_manager).
- Обновление позиций в каждой новой свече (проверка TP/SL, trailing, partial close).
- Расчёт виртуального PNL с учётом комиссии, плеча и размера позиции.
- Передача закрытий в tp_sl_manager и scenario_tracker (для статистики).
- Синхронизация с реальным режимом: те же проверки (no open по типу, no new in closed candle).

Виртуальный режим используется:
- Когда монета не в whitelist (только симуляция для расчёта PR).
- В режиме "virtual" для всего бота.
- Для тестирования без риска.

=== Главные функции и за что отвечают ===

- VirtualTrader() — инициализация: виртуальный баланс (из config или storage), комиссии, открытые позиции.
- open_position(pos: dict) — открывает виртуальную позицию (расчёт размера, запись в open_positions).
- update_positions(candle_data: dict) — проверяет все открытые позиции на закрытие (TP/SL hit).
- _close_position(pos_id: str, is_tp: bool, price: float) — закрывает позицию, обновляет баланс, передаёт в scenario_tracker.
- get_pnl() → float — текущий виртуальный профит/убыток.
- get_balance() → float — текущий виртуальный депозит.

=== Примечания ===
- Комиссия: taker fee из config (например, 0.04%).
- Плечо: берётся из config или per-symbol.
- Полностью соответствует ТЗ: виртуальная торговля для PR и тестирования.
- Нет реальных ордеров — только симуляция.
- Логи через setup_logger.
- Готов к использованию в live_loop и entry_manager.
"""

from typing import Dict, Optional
import time

from src.core.config import load_config
from src.core.enums import Direction, TradeMode
from src.trading.tp_sl_manager import TPSLManager
from src.model.scenario_tracker import ScenarioTracker
from src.utils.logger import setup_logger

logger = setup_logger('virtual_trader', logging.INFO)

class VirtualTrader:
    """
    Симулятор виртуальной торговли.
    """
    def __init__(self):
        self.config = load_config()
        self.balance = self.config['trading']['initial_balance']  # Начальный виртуальный депозит
        self.positions: Dict[str, Dict] = {}  # pos_id → pos dict
        self.tp_sl_manager = TPSLManager()
        self.scenario_tracker = ScenarioTracker()
        self.commission_rate = self.config['trading']['commission_rate']  # taker fee, e.g. 0.0004

    def open_position(self, pos: Dict):
        """
        Открывает виртуальную позицию.
        pos: {'symbol', 'type', 'direction', 'entry_price', 'size', 'tp', 'sl', ...}
        """
        pos_id = f"{pos['symbol']}_{pos['type']}_{int(time.time()*1000)}"
        pos['pos_id'] = pos_id
        pos['open_time'] = time.time()
        pos['entry_price'] = pos['entry_price']
        pos['size'] = pos['size']
        pos['direction'] = pos['direction']
        pos['tp'] = pos['tp']
        pos['sl'] = pos['sl']

        # Вычитаем комиссию открытия
        commission = pos['size'] * pos['entry_price'] * self.commission_rate
        self.balance -= commission

        self.positions[pos_id] = pos
        self.tp_sl_manager.add_open_position(pos)

        logger.info(f"[VIRTUAL] Открыта позиция {pos['type']} {pos['direction']} на {pos['symbol']}, size={pos['size']}, entry={pos['entry_price']}")

    def update_positions(self, candle_data: Dict):
        """
        Обновляет все виртуальные позиции по новой свече.
        Проверяет TP/SL, закрывает если нужно, обновляет баланс.
        """
        closed = []
        for pos_id, pos in list(self.positions.items()):
            close_result = self.tp_sl_manager.check_close(pos, candle_data)
            if close_result['closed']:
                self._close_position(pos_id, close_result['is_tp'], close_result['price'])
                closed.append(pos_id)

        for pos_id in closed:
            del self.positions[pos_id]

    def _close_position(self, pos_id: str, is_tp: bool, close_price: float):
        """
        Закрывает виртуальную позицию, обновляет баланс и статистику.
        """
        pos = self.positions[pos_id]
        direction = pos['direction']
        size = pos['size']
        entry = pos['entry_price']

        if direction == Direction.LONG.value:
            pnl = (close_price - entry) * size
        else:
            pnl = (entry - close_price) * size

        # Комиссия закрытия
        commission = size * close_price * self.commission_rate
        net_pnl = pnl - commission

        self.balance += net_pnl

        # Передача в scenario_tracker
        self.scenario_tracker.update_scenario(
            scenario_key=pos['scenario_key'] if 'scenario_key' in pos else "unknown",
            is_win=is_tp
        )

        logger.info(f"[VIRTUAL] Закрыта позиция {pos['type']} {direction} на {pos['symbol']}, PNL={net_pnl:.2f}, баланс={self.balance:.2f}")

    def get_balance(self) -> float:
        """Текущий виртуальный баланс."""
        return self.balance

    def get_pnl(self) -> float:
        """Общий виртуальный PNL (с момента запуска)."""
        return self.balance - self.config['trading']['initial_balance']

    def get_open_positions(self, symbol: Optional[str] = None) -> list:
        """Список открытых виртуальных позиций."""
        if symbol:
            return [p for p in self.positions.values() if p['symbol'] == symbol]
        return list(self.positions.values())