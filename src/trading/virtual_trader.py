# src/trading/virtual_trader.py
"""
Виртуальная торговля + обновление PR после каждой закрытой сделки.
Добавлено подробное логирование всех ключевых событий.
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
    close_price: Optional[float] = None
    pnl_pct: float = 0.0
    close_time: Optional[int] = None
    close_reason: Optional[str] = None


class VirtualTrader:
    def __init__(self):
        self.config = get_config()
        self.storage = Storage()
        self.active_positions: Dict[str, VirtualPosition] = {}
        logger.info("VirtualTrader инициализирован", max_active=len(self.active_positions))

    def process_signal(self, signal: AnomalySignal, model_pred):
        coin = signal.coin
        if not coin:
            logger.warning("Сигнал без монеты", signal=signal)
            return

        if coin in self.active_positions:
            logger.debug("Позиция уже открыта, пропуск", coin=coin)
            return

        position = VirtualPosition(
            coin=coin,
            side=signal.direction_hint,
            entry_price=0.0,  # заполнится позже
            size=0.0,
            entry_time=signal.timestamp,
            tp_price=0.0,
            sl_price=0.0,
            anomaly_signal=signal,
            max_hold_time_ms=self._get_max_hold_ms()
        )

        self.active_positions[coin] = position
        logger.info("Виртуальная позиция открыта",
                    coin=coin,
                    side=signal.direction_hint.value,
                    anomaly=signal.anomaly_type.value,
                    tf=signal.tf)

    def close_position(self, coin: str, close_price: float, reason: str):
        if coin not in self.active_positions:
            logger.debug("Попытка закрыть несуществующую позицию", coin=coin)
            return

        pos = self.active_positions.pop(coin)
        pos.close_price = close_price
        pos.close_time = int(time.time() * 1000)
        pos.close_reason = reason

        pnl_pct = (close_price - pos.entry_price) / pos.entry_price * 100 if pos.side == Direction.LONG else \
                  (pos.entry_price - close_price) / pos.entry_price * 100

        pos.pnl_pct = pnl_pct

        logger.info("Виртуальная позиция закрыта",
                    coin=coin,
                    reason=reason,
                    pnl_pct=f"{pnl_pct:+.4f}%",
                    hold_time_sec=(pos.close_time - pos.entry_time) / 1000)

        self._update_pr_snapshot(pos)

    def _update_pr_snapshot(self, pos: VirtualPosition):
        is_tp = pos.close_reason == "tp"
        is_sl = pos.close_reason == "sl"
        is_timeout = pos.close_reason == "timeout"

        tp_len = abs(pos.tp_price - pos.entry_price) / pos.entry_price * 100
        sl_len = abs(pos.sl_price - pos.entry_price) / pos.entry_price * 100
        pr_contrib = tp_len if is_tp else -sl_len

        logger.debug("Обновление PR",
                     coin=pos.coin,
                     reason=pos.close_reason,
                     pr_contrib=f"{pr_contrib:+.4f}",
                     tp_hit=1 if is_tp else 0,
                     sl_hit=1 if is_sl or is_timeout else 0)

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
            pos.coin, pos.anomaly_signal.tf, 100,  # период пока хардкод, потом из конфига
            pos.anomaly_signal.anomaly_type.value, pos.side.value,
            pr_contrib, 1.0 if is_tp else 0.0,
            1 if is_tp else 0,
            1 if is_sl or is_timeout else 0,
            pr_contrib,
            1 if is_tp else 0,
            1 if is_sl or is_timeout else 0
        ])

    def _get_max_hold_ms(self) -> Optional[int]:
        minutes = self.config.get("trading", {}).get("max_hold_time_minutes")
        if minutes and minutes > 0:
            logger.debug("Тайм-аут включён", minutes=minutes)
            return int(minutes * 60 * 1000)
        logger.debug("Тайм-аут выключен")
        return None

    def check_timeouts(self):
        now = int(time.time() * 1000)
        for coin, pos in list(self.active_positions.items()):
            if pos.max_hold_time_ms and (now - pos.entry_time) >= pos.max_hold_time_ms:
                logger.warning("Тайм-аут сработал", coin=coin, hold_sec=(now - pos.entry_time)/1000)
                self.close_position(coin, pos.entry_price * 0.999, "timeout")  # условная цена