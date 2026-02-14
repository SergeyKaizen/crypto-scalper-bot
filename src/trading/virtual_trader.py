# src/trading/virtual_trader.py
"""
Виртуальная торговля (shadow trading) для сбора статистики PR.
Теперь с полноценной симуляцией закрытия позиций по TP/SL на каждой новой свече.
Закрытие ТОЛЬКО по TP или SL — как указано в ТЗ.
"""

from dataclasses import dataclass
import time
from typing import Dict, Optional

from ..utils.logger import logger
from ..core.types import AnomalySignal, Direction, Position
from ..data.storage import Storage
from ..core.config import get_config


@dataclass
class VirtualPosition(Position):
    """Расширенная позиция для виртуальной торговли."""
    current_price: float = 0.0      # обновляется на каждой свече
    close_price: Optional[float] = None
    close_time: Optional[int] = None
    close_reason: Optional[str] = None  # "tp" или "sl"


class VirtualTrader:
    def __init__(self):
        self.config = get_config()
        self.storage = Storage()
        self.active_positions: Dict[str, VirtualPosition] = {}  # coin → position
        logger.info("VirtualTrader инициализирован")

    def process_signal(self, signal: AnomalySignal, model_pred):
        """Открывает виртуальную позицию при аномалии."""
        coin = signal.coin
        if not coin:
            logger.warning("Сигнал без монеты")
            return

        if coin in self.active_positions:
            logger.debug("Уже есть открытая позиция, пропуск", coin=coin)
            return

        # Здесь должен быть расчёт entry_price, size, tp_price, sl_price
        # Пока берём условные значения (замени на реальный risk_manager)
        entry_price = 99999.0  # будет заменено на реальную цену свечи
        size = 1.0
        tp_price = entry_price * 1.005 if signal.direction_hint == Direction.LONG else entry_price * 0.995
        sl_price = entry_price * 0.995 if signal.direction_hint == Direction.LONG else entry_price * 1.005

        position = VirtualPosition(
            coin=coin,
            side=signal.direction_hint,
            entry_price=entry_price,
            size=size,
            entry_time=signal.timestamp,
            tp_price=tp_price,
            sl_price=sl_price,
            anomaly_signal=signal,
            current_price=entry_price  # начальная цена
        )

        self.active_positions[coin] = position
        logger.info("Открыта виртуальная позиция",
                    coin=coin,
                    side=signal.direction_hint.value,
                    entry_price=entry_price,
                    tp_price=tp_price,
                    sl_price=sl_price)

    def update_on_new_candle(self, coin: str, new_price: float, timestamp: int):
        """
        Вызывается на каждой новой свече.
        Проверяет ВСЕ активные позиции этой монеты и закрывает по TP/SL.
        """
        if coin not in self.active_positions:
            return

        pos = self.active_positions[coin]
        pos.current_price = new_price

        is_long = pos.side == Direction.LONG

        # Проверка TP / SL
        if (is_long and new_price >= pos.tp_price) or (not is_long and new_price <= pos.tp_price):
            self.close_position(coin, new_price, "tp", timestamp)
        elif (is_long and new_price <= pos.sl_price) or (not is_long and new_price >= pos.sl_price):
            self.close_position(coin, new_price, "sl", timestamp)

    def close_position(self, coin: str, close_price: float, reason: str, close_time: int):
        """Закрывает позицию и обновляет PR."""
        if coin not in self.active_positions:
            return

        pos = self.active_positions.pop(coin)
        pos.close_price = close_price
        pos.close_time = close_time
        pos.close_reason = reason

        # Расчёт pnl и pr_contribution (по формуле из ТЗ)
        pnl_pct = (close_price - pos.entry_price) / pos.entry_price * 100 if pos.side == Direction.LONG else \
                  (pos.entry_price - close_price) / pos.entry_price * 100

        tp_length = abs(pos.tp_price - pos.entry_price) / pos.entry_price * 100
        sl_length = abs(pos.sl_price - pos.entry_price) / pos.entry_price * 100

        pr_contribution = tp_length if reason == "tp" else -sl_length

        logger.info("Виртуальная позиция закрыта",
                    coin=coin,
                    reason=reason,
                    pnl_pct=f"{pnl_pct:+.4f}%",
                    hold_time_sec=(close_time - pos.entry_time) / 1000)

        # Обновляем PR в базе
        self._update_pr_snapshot(pos.coin, pos.anomaly_signal, pos.side, pr_contribution, reason)

    def _update_pr_snapshot(self, coin: str, signal: AnomalySignal, side: Direction, pr_contrib: float, reason: str):
        """Обновление таблицы pr_snapshots после закрытия."""
        is_tp = reason == "tp"
        is_sl = reason == "sl"

        self.storage.execute("""
            INSERT INTO pr_snapshots 
            (symbol, tf, period, anomaly_type, direction, pr_value, winrate, 
             total_deals, tp_hits, sl_hits, last_update)
            VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(symbol, tf, period, anomaly_type, direction) DO UPDATE SET
                pr_value = pr_value + ?,
                total_deals = total_deals + 1,
                tp_hits = tp_hits + ?,
                sl_hits = sl_hits + ?,
                last_update = CURRENT_TIMESTAMP
        """, [
            coin,
            signal.tf,
            100,  # период — можно брать из конфига
            signal.anomaly_type.value,
            side.value,
            pr_contrib,
            1.0 if is_tp else 0.0,
            1 if is_tp else 0,
            1 if is_sl else 0,
            pr_contrib,
            1 if is_tp else 0,
            1 if is_sl else 0
        ])

        logger.debug("PR обновлён после виртуальной сделки",
                     coin=coin,
                     reason=reason,
                     pr_contrib=f"{pr_contrib:+.4f}")