# src/utils/metrics.py
"""
Модуль расчёта метрик производительности (Sharpe, Profit Factor, Drawdown и т.д.).

Основные функции:
- calculate_trade_metrics() — метрики одной сделки (R:R, pnl_pct, win/loss)
- calculate_strategy_metrics() — метрики всей стратегии (Sharpe, Sortino, Profit Factor, Max Drawdown)
- calculate_drawdown() — максимальная просадка (% и абсолют)
- calculate_sharpe_ratio() — Sharpe (с risk-free rate = 0 для крипты)
- calculate_profit_factor() — Profit Factor (gross profit / gross loss)
- calculate_expectancy() — средний профит на сделку (winrate × avg_win - lossrate × avg_loss)

Используется в:
- после каждой закрытой позиции (live / virtual / shadow)
- после бэктеста (backtest/engine)
- для оценки сценариев (scenario_tracker)
- в логах и отчётах

Все расчёты на Polars или numpy — быстро и точно
"""

import logging
from typing import List, Dict, Tuple, Optional
import polars as pl
import numpy as np

from src.core.types import TradeResult

logger = logging.getLogger(__name__)


def calculate_trade_metrics(trade: TradeResult) -> Dict[str, float]:
    """
    Метрики одной сделки

    Returns:
        {
            "pnl_pct": float,
            "pnl_usdt": float,
            "rr_ratio": float,        # R:R (reward/risk)
            "is_win": bool,
            "duration_minutes": float,
            "reason": str
        }
    """
    pnl_pct = trade.pnl_pct
    pnl_usdt = trade.pnl_usdt
    is_win = trade.is_win

    # R:R — отношение профита к риску
    risk_pct = abs(trade.position.entry_price - trade.position.sl_price) / trade.position.entry_price * 100
    rr_ratio = pnl_pct / risk_pct if risk_pct > 0 else 0.0

    duration_min = (trade.exit_time - trade.position.entry_time).total_seconds() / 60

    return {
        "pnl_pct": pnl_pct,
        "pnl_usdt": pnl_usdt,
        "rr_ratio": rr_ratio,
        "is_win": is_win,
        "duration_minutes": duration_min,
        "reason": trade.reason
    }


def calculate_strategy_metrics(trades: List[TradeResult]) -> Dict[str, float]:
    """
    Метрики всей стратегии / периода

    Args:
        trades — список всех закрытых сделок

    Returns:
        {
            "total_trades": int,
            "winrate": float,
            "avg_pnl_pct": float,
            "profit_factor": float,
            "sharpe_ratio": float,
            "sortino_ratio": float,
            "max_drawdown_pct": float,
            "max_drawdown_usdt": float,
            "expectancy": float,
            "total_pnl_pct": float,
            "total_pnl_usdt": float
        }
    """
    if not trades:
        return {
            "total_trades": 0,
            "winrate": 0.0,
            "avg_pnl_pct": 0.0,
            "profit_factor": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "max_drawdown_pct": 0.0,
            "max_drawdown_usdt": 0.0,
            "expectancy": 0.0,
            "total_pnl_pct": 0.0,
            "total_pnl_usdt": 0.0
        }

    df = pl.DataFrame([{
        "pnl_pct": t.pnl_pct,
        "pnl_usdt": t.pnl_usdt,
        "is_win": t.is_win
    } for t in trades])

    total_trades = len(df)
    wins = df["is_win"].sum()
    winrate = wins / total_trades if total_trades > 0 else 0.0

    avg_pnl = df["pnl_pct"].mean()
    total_pnl_pct = df["pnl_pct"].sum()
    total_pnl_usdt = df["pnl_usdt"].sum()

    # Profit Factor
    gross_profit = df.filter(pl.col("pnl_pct") > 0)["pnl_pct"].sum()
    gross_loss = abs(df.filter(pl.col("pnl_pct") < 0)["pnl_pct"].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Expectancy
    avg_win = df.filter(pl.col("pnl_pct") > 0)["pnl_pct"].mean()
    avg_loss = abs(df.filter(pl.col("pnl_pct") < 0)["pnl_pct"].mean())
    expectancy = winrate * avg_win - (1 - winrate) * avg_loss

    # Sharpe Ratio (risk-free rate = 0 для крипты)
    returns = df["pnl_pct"].to_numpy()
    sharpe = returns.mean() / returns.std() if returns.std() > 0 else 0.0

    # Sortino Ratio (только отрицательные отклонения)
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std() if len(downside_returns) > 0 else 0.0
    sortino = returns.mean() / downside_std if downside_std > 0 else 0.0

    # Max Drawdown
    equity = np.cumsum(returns)
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak * 100
    max_dd_pct = drawdown.min()
    max_dd_usdt = (equity - peak).min()  # в USDT (примерно)

    return {
        "total_trades": total_trades,
        "winrate": winrate,
        "avg_pnl_pct": avg_pnl,
        "profit_factor": profit_factor,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown_pct": max_dd_pct,
        "max_drawdown_usdt": max_dd_usdt,
        "expectancy": expectancy,
        "total_pnl_pct": total_pnl_pct,
        "total_pnl_usdt": total_pnl_usdt
    }


def calculate_drawdown(trades: List[TradeResult]) -> Tuple[float, float]:
    """Максимальная просадка (%) и в USDT"""
    if not trades:
        return 0.0, 0.0

    equity = [0]
    for t in trades:
        equity.append(equity[-1] + t.pnl_usdt)

    equity = np.array(equity)
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak * 100
    max_dd_pct = drawdown.min()
    max_dd_usdt = (equity - peak).min()

    return max_dd_pct, max_dd_usdt


def log_strategy_metrics(metrics: Dict):
    """Логирование ключевых метрик"""
    logger.info("Strategy Metrics:")
    logger.info("  Total trades: %d", metrics["total_trades"])
    logger.info("  Winrate: %.2f%%", metrics["winrate"] * 100)
    logger.info("  Avg PnL: %.2f%%", metrics["avg_pnl_pct"])
    logger.info("  Profit Factor: %.2f", metrics["profit_factor"])
    logger.info("  Sharpe Ratio: %.2f", metrics["sharpe_ratio"])
    logger.info("  Sortino Ratio: %.2f", metrics["sortino_ratio"])
    logger.info("  Max Drawdown: %.2f%% (%.2f USDT)", metrics["max_drawdown_pct"], metrics["max_drawdown_usdt"])
    logger.info("  Expectancy: %.2f%% per trade", metrics["expectancy"])