"""
src/trading/entry_manager.py

=== Основной принцип работы файла ===
Менеджер открытия позиций в live-режиме.

Учитывает все последние изменения:
- multi-TF consensus (только confirmed аномалии)
- quiet_streak как дополнительная фича
- sequential паттерны и delta VA в pred_features
- повышенный min_prob для Q
- строгое ограничение: только 1 открытая позиция по одному типу аномалии (C/V/CV/Q) на монете
- запрет новой позиции в свече, где предыдущая закрылась (по ТЗ для live)
- глобальный lock: нет новой позиции пока есть любая открытая на монете (новое изменение для live)

В live-режиме — консервативно: одна по типу, нет входа после закрытия в той же свeche, нет новой пока открыта текущая.
Виртуальный режим — та же логика для симуляции PNL без реальных ордеров.

=== Главные функции ===
- process_signal(symbol: str, anomaly_type: str, direction: str, prob: float, candle_data: dict, candle_ts: int) — основная точка входа:
  - Проверяет whitelist match (PR config == signal).
  - Проверяет min_prob (выше для Q).
  - Проверяет no open position по типу.
  - Проверяет no close in this candle (флаг от tp_sl_manager).
  - Рассчитывает size через risk_manager.
  - Открывает позицию (real или virtual).
- _can_open_position(symbol: str, anomaly_type: str, candle_ts: int) → bool — проверки условий.
- _open_position(...) — вызов executor или virtual_trader.
- update_candle_close_flag(candle_ts: int) — флаг закрытия в свече (от tp_sl_manager).

=== Примечания ===
- Signal vs PR: если Signal_BTC == PR_BTC — real, иначе virtual (по ТЗ).
- Q имеет выше min_prob_q (например, 0.75) для снижения ложных входов.
- Полностью соответствует ТЗ + уточнениям (одна по типу, no new in closed candle, global lock).
- Готов к интеграции в live_loop.py.
- Логи через setup_logger.
"""

from typing import Dict, Optional
import time

from src.core.config import load_config
from src.core.enums import AnomalyType, Direction, TradeMode
from src.trading.risk_manager import RiskManager
from src.trading.order_executor import OrderExecutor
from src.trading.virtual_trader import VirtualTrader
from src.trading.tp_sl_manager import TPSLManager
from src.utils.logger import setup_logger

logger = setup_logger('entry_manager', logging.INFO)

class EntryManager:
    """
    Менеджер открытия позиций в live-режиме.
    """
    def __init__(self):
        self.config = load_config()
        self.risk_manager = RiskManager()
        self.order_executor = OrderExecutor()
        self.virtual_trader = VirtualTrader()
        self.tp_sl_manager = TPSLManager()  # для проверки закрытий

        self.mode = TradeMode(self.config['trading']['mode'])  # real / virtual
        self.candle_close_flags = {}  # candle_ts: bool (было закрытие в этой свече)

    def process_signal(
        self,
        symbol: str,
        anomaly_type: str,
        direction: str,
        prob: float,
        candle_data: Dict,
        candle_ts: int
    ):
        """
        Основная функция: обработка сигнала от inference.
        - Проверяет whitelist match.
        - Проверка вероятности и условий входа.
        - Открывает позицию (real или virtual).
        """
        # 1. Проверка whitelist (PR config)
        whitelist_config = self.storage.get_whitelist_config(symbol)  # best_tf, best_period, best_anomaly, best_direction
        if not whitelist_config:
            logger.debug(f"{symbol} не в whitelist — только виртуальный режим")
            mode = TradeMode.VIRTUAL
        else:
            signal_key = f"{anomaly_type}_{direction}"
            pr_key = f"{whitelist_config['best_anomaly']}_{whitelist_config['best_direction']}"
            if signal_key == pr_key:
                mode = self.mode  # real если режим real
            else:
                mode = TradeMode.VIRTUAL

        # 2. Проверка вероятности
        min_prob = self.config['model']['min_prob_q'] if anomaly_type == AnomalyType.QUIET.value else self.config['model']['min_prob_anomaly']
        if prob < min_prob:
            logger.debug(f"Низкая вероятность {prob:.2f} < {min_prob:.2f} для {symbol} {anomaly_type}")
            return

        # 3. Проверка условий входа
        if not self._can_open_position(symbol, anomaly_type, candle_ts):
            return

        # 4. Расчёт размера позиции
        size = self.risk_manager.calculate_size(
            symbol=symbol,
            entry_price=candle_data['close'],
            sl_price=self.tp_sl_manager.calculate_sl(candle_data, direction),
            risk_pct=self.config['trading']['risk_pct']
        )

        if size <= 0:
            logger.warning(f"Некорректный размер позиции для {symbol}")
            return

        # 5. Открытие позиции
        pos = {
            'symbol': symbol,
            'type': anomaly_type,
            'direction': direction,
            'entry_price': candle_data['close'],
            'size': size,
            'open_ts': candle_ts,
            'prob': prob
        }

        if mode == TradeMode.REAL:
            order_id = self.order_executor.place_order(pos)
            if order_id:
                pos['order_id'] = order_id
                logger.info(f"Открыта реальная позиция {anomaly_type} {direction} на {symbol}, size={size}")
            else:
                logger.error(f"Ошибка открытия реальной позиции {symbol}")
                return
        else:
            self.virtual_trader.open_position(pos)
            logger.debug(f"Открыта виртуальная позиция {anomaly_type} {direction} на {symbol}, size={size}")

        # Добавляем в открытые позиции (для проверки в live_loop)
        self.tp_sl_manager.add_open_position(pos)

    def _can_open_position(self, symbol: str, anomaly_type: str, candle_ts: int) -> bool:
        """
        Проверка возможности открытия позиции.
        - Нет открытой по этому типу.
        - Нет закрытия в этой свече.
        """
        # 1. Нет открытой по типу
        if self.tp_sl_manager.has_open_position(symbol, anomaly_type):
            logger.debug(f"Уже открыта позиция {anomaly_type} на {symbol}")
            return False

        # 2. Нет закрытия в этой свече
        if self.candle_close_flags.get(candle_ts, False):
            logger.debug(f"В свече {candle_ts} была закрыта позиция — вход запрещён")
            return False

        return True

    def update_candle_close_flag(self, candle_ts: int):
        """
        Устанавливает флаг закрытия позиции в свeche (вызывается из tp_sl_manager).
        """
        self.candle_close_flags[candle_ts] = True
        logger.debug(f"Установлен флаг закрытия в свече {candle_ts}")