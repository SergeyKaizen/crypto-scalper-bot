# src/backtest/pr_calculator.py
"""
Модуль расчёта Profitable Rating (PR) и фильтра монет.

Основные функции:
- calculate_pr() — расчёт PR (classic / normalized)
- update_pr_for_symbol() — обновление PR после закрытой сделки
- get_top_coins() — топ-монеты по PR + min_deals
- determine_best_direction() — выбор L/S/LS по PR
- determine_best_signal_type() — выбор C/V/CV/Q

Логика:
- PR считается по виртуальным сделкам за pr_analysis_period_candles (250)
- Classic: (TP_count × avg_TP_size) - (SL_count × avg_SL_size) / total_trades
- Normalized: classic × log10(total_trades + 5) / 2.0 — штраф за малое кол-во
- min_deals_in_pr_period — если сделок < min → PR=0 (монета не активируется)
- PR пересчитывается после каждой закрытой виртуальной сделки (или каждые 10 мин на телефоне)
- Лучшее направление: max(PR_L, PR_S, PR_LS)
- Лучший тип сигнала: max(PR_C, PR_V, PR_CV, PR_Q)

Хранит статистику:
- trades: List[TradeResult] — все закрытые сделки по монете
- pr_history: Dict[symbol: List[float]] — история PR
"""

import logging
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import math

import polars as pl

from src.core.config import load_config
from src.core.types import TradeResult
from src.core.enums import Direction, AnomalyType

logger = logging.getLogger(__name__)


class PRCalculator:
    """Расчёт Profitable Rating и фильтр монет"""

    def __init__(self, config: Dict):
        self.config = config
        self.trades = defaultdict(list)  # symbol → List[TradeResult]
        self.pr_cache = defaultdict(float)  # symbol → current PR
        self.pr_history = defaultdict(list)  # symbol → List[PR values]
        self.last_recalc_time = {}  # symbol → last recalc timestamp

        self.period_candles = config["pr"]["analysis_period_candles"]
        self.pr_mode = config["pr"]["mode"]
        self.min_deals = config["pr"]["min_deals_in_pr_period"]
        self.min_pr_threshold = config["pr"]["min_pr_threshold"]

        logger.info("PRCalculator initialized: mode=%s, period=%d candles, min_deals=%d", 
                    self.pr_mode, self.period_candles, self.min_deals)

    def add_trade(self, symbol: str, trade: TradeResult):
        """Добавление результата закрытой сделки"""
        self.trades[symbol].append(trade)
        logger.debug("Added trade for %s: pnl=%.2f%%, reason=%s", symbol, trade.pnl_pct, trade.reason)

        # Пересчёт PR (можно оптимизировать — делать не после каждой сделки)
        self.update_pr_for_symbol(symbol)

    def update_pr_for_symbol(self, symbol: str):
        """Пересчёт PR для одной монеты"""
        trades = self.trades[symbol]
        if not trades:
            return

        # Фильтруем сделки за последние period_candles
        current_time = trades[-1].exit_time.timestamp()
        cutoff_time = current_time - self.period_candles * 60  # Примерно (1m = 60 сек)
        recent_trades = [t for t in trades if t.exit_time.timestamp() >= cutoff_time]

        if len(recent_trades) < self.min_deals:
            self.pr_cache[symbol] = 0.0
            return

        # Считаем TP/SL
        tp_trades = [t for t in recent_trades if t.reason == "TP"]
        sl_trades = [t for t in recent_trades if t.reason == "SL"]

        tp_count = len(tp_trades)
        sl_count = len(sl_trades)
        total_trades = tp_count + sl_count

        if total_trades == 0:
            self.pr_cache[symbol] = 0.0
            return

        avg_tp_size = sum(t.pnl_pct for t in tp_trades) / tp_count if tp_count > 0 else 0
        avg_sl_size = abs(sum(t.pnl_pct for t in sl_trades) / sl_count) if sl_count > 0 else 0

        pr_raw = (tp_count * avg_tp_size - sl_count * avg_sl_size) / total_trades

        if self.pr_mode == "normalized":
            pr = pr_raw * math.log10(total_trades + 5) / 2.0
        else:
            pr = pr_raw

        # Фильтр min_deals
        if total_trades < self.min_deals:
            pr = 0.0

        self.pr_cache[symbol] = pr
        self.pr_history[symbol].append(pr)

        logger.debug("Updated PR for %s: %.4f (raw=%.4f, trades=%d)", symbol, pr, pr_raw, total_trades)

    def get_top_coins(self, top_n: int = 20) -> List[Tuple[str, float]]:
        """Топ-монеты по PR"""
        sorted_coins = sorted(self.pr_cache.items(), key=lambda x: x[1], reverse=True)
        return sorted_coins[:top_n]

    def determine_best_direction(self, symbol: str) -> str:
        """L / S / LS — по максимальному PR"""
        # В реальном коде — считаем PR_L, PR_S, PR_LS отдельно
        # Здесь упрощённо — берём общее
        pr = self.pr_cache.get(symbol, 0.0)
        if pr > self.min_pr_threshold:
            return "LS"  # По умолчанию — оба направления
        return "NONE"

    def determine_best_signal_type(self, symbol: str) -> str:
        """C / V / CV / Q — по максимальному PR"""
        # Упрощённо — возвращаем "CV" как самый сильный
        return "CV"  # В реальном коде — отдельный расчёт по типам