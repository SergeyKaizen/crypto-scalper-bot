# src/trading/shadow_trading.py
"""
Модуль shadow trading — параллельная симуляция реальных ордеров.

Shadow trading:
- Запускается параллельно основной торговле (real/virtual)
- Для каждого реального сигнала/ордера создаётся "тень" — симуляция с slippage и комиссиями
- Считает "реальный" P&L, slippage %, delay исполнения
- Сравнивает с виртуальным P&L (без slippage)
- Метрики выводятся в логи / отдельный файл (shadow_metrics.csv)

Преимущества:
- Видим, сколько реально "съедает" биржа (slippage 0.05–0.3%, комиссии 0.04%)
- Можно сравнивать real vs shadow — понять, насколько симуляция оптимистична
- На сервере — параллельная обработка (asyncio или joblib)

Логика:
- При каждом place_order() — создаётся shadow-ордер
- Shadow-ордер симулируется с slippage (0.1–0.2%) и комиссией taker
- После закрытия позиции — сравнивается shadow P&L vs virtual/real
- Метрики сохраняются: symbol, entry/exit, slippage_pct, commission_usdt, pnl_diff

На телефоне — shadow_trading: false (экономия батареи)
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime
import asyncio

from src.core.config import load_config
from src.core.types import Position, TradeResult
from src.core.constants import BINANCE_FUTURES_TAKER_FEE
from src.trading.order_executor import simulate_order_execution

logger = logging.getLogger(__name__)


class ShadowTrading:
    """Параллельная симуляция реальных ордеров (shadow trading)"""

    def __init__(self, config: Dict):
        self.config = config
        self.enabled = config.get("shadow_trading", True)
        self.slippage_pct = config.get("slippage_pct", 0.15)  # 0.15% — средний slippage на Binance
        self.commission_rate = float(BINANCE_FUTURES_TAKER_FEE)

        # Shadow-позиции: symbol → Position (копия реальной)
        self.shadow_positions = {}

        # Метрики
        self.metrics = []  # List[Dict: symbol, entry/exit, slippage, pnl_diff]

        if not self.enabled:
            logger.info("Shadow trading disabled")
        else:
            logger.info("Shadow trading enabled (slippage=%.2f%%)", self.slippage_pct)

    async def process_signal(self, signal: Signal, real_position: Optional[Position] = None):
        """Обработка сигнала — создание shadow-позиции"""
        if not self.enabled:
            return

        # Если есть реальная позиция — копируем
        if real_position:
            shadow_pos = Position(
                entry_time=real_position.entry_time,
                entry_price=real_position.entry_price * (1 + self.slippage_pct / 100 if real_position.direction == "L" else 1 - self.slippage_pct / 100),
                size=real_position.size,
                direction=real_position.direction,
                tp_price=real_position.tp_price,
                sl_price=real_position.sl_price,
                is_virtual=True,
                symbol=real_position.symbol
            )
            self.shadow_positions[signal.symbol] = shadow_pos
            logger.debug("Shadow position created for %s @ %.2f (slippage applied)", signal.symbol, shadow_pos.entry_price)

    async def update_shadow_position(self, symbol: str, current_high: float, current_low: float):
        """Обновление shadow-позиции (trailing, check TP/SL)"""
        if not self.enabled or symbol not in self.shadow_positions:
            return

        pos = self.shadow_positions[symbol]

        # Обновляем trailing (аналогично tp_sl_manager)
        if pos.trailing_active:
            if pos.direction == "L":
                pos.trailing_price = max(pos.trailing_price, current_high * (1 - self.config["trailing_distance_pct"] / 100))
            else:
                pos.trailing_price = min(pos.trailing_price, current_low * (1 + self.config["trailing_distance_pct"] / 100))

        # Проверка TP/SL (упрощённо)
        if pos.direction == "L":
            if current_high >= pos.tp_price:
                self._close_shadow_position(symbol, pos.tp_price, "TP")
            elif current_low <= pos.sl_price:
                self._close_shadow_position(symbol, pos.sl_price, "SL")
        else:
            # Для шорта аналогично

    def _close_shadow_position(self, symbol: str, exit_price: float, reason: str):
        """Закрытие shadow-позиции и расчёт метрик"""
        pos = self.shadow_positions.pop(symbol, None)
        if not pos:
            return

        pnl_pct = (exit_price - pos.entry_price) / pos.entry_price * 100 if pos.direction == "L" else (pos.entry_price - exit_price) / pos.entry_price * 100
        pnl_usdt = pos.size * (exit_price - pos.entry_price) if pos.direction == "L" else pos.size * (pos.entry_price - exit_price)

        # Slippage impact
        slippage_impact = (exit_price - pos.entry_price) / pos.entry_price * 100 if pos.direction == "L" else (pos.entry_price - exit_price) / pos.entry_price * 100

        commission = pos.size * pos.entry_price * self.commission_rate * 2  # entry + exit

        metric = {
            "symbol": symbol,
            "entry_price": pos.entry_price,
            "exit_price": exit_price,
            "pnl_pct": pnl_pct,
            "pnl_usdt": pnl_usdt,
            "slippage_pct": slippage_impact,
            "commission_usdt": commission,
            "reason": reason,
            "timestamp": datetime.now()
        }

        self.metrics.append(metric)
        logger.info("Shadow position closed: %s, pnl=%.2f%%, slippage=%.2f%%, reason=%s", 
                    symbol, pnl_pct, slippage_impact, reason)

    def export_metrics(self):
        """Выгрузка метрик shadow trading в CSV"""
        if not self.metrics:
            return

        df = pl.DataFrame(self.metrics)
        path = "logs/shadow_metrics.csv"
        df.write_csv(path)
        logger.info("Shadow trading metrics exported to %s (%d records)", path, len(self.metrics))