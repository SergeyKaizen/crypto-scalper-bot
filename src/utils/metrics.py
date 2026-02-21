"""
src/utils/metrics.py

=== Основной принцип работы файла ===

Этот файл содержит функции для расчёта и агрегации торговых метрик по закрытым сделкам.
Он используется для:
- Оценки результатов бектеста и live торговли.
- Вычисления ключевых показателей: winrate, profit factor, average R:R, net profit, max drawdown, Sharpe ratio.
- Группировки статистики по типам аномалий (C/V/CV/Q) и направлениям (L/S) — для анализа PR и сценариев.
- Красивого вывода метрик в лог.

Все метрики рассчитываются на основе списка закрытых сделок (trades).
Нет зависимости от модели — чистая статистика на основе PNL и исходов.

=== Главные функции и за что отвечают ===

- calculate_trade_metrics(trades: list[dict]) → dict
  Основная функция. Принимает список закрытых сделок и возвращает все ключевые метрики.

- aggregate_by_anomaly(trades: list[dict]) → pd.DataFrame
  Группирует метрики по типу аномалии (C, V, CV, Q).

- aggregate_by_direction(trades: list[dict]) → pd.DataFrame
  Группирует метрики по направлению (L / S).

- log_metrics(metrics: dict, prefix: str = "") 
  Красиво выводит все метрики в лог.

=== Структура сделки (trades) ===
Каждая сделка — словарь с полями:
{
    'pnl': float,           # чистый PNL в USDT
    'is_tp': bool,          # закрыто по TP (True) или SL (False)
    'rr': float,            # R:R этой сделки
    'anomaly_type': str,    # 'C', 'V', 'CV', 'Q'
    'direction': str,       # 'L' или 'S'
    'symbol': str,
    ...
}

=== Примечания ===
- Sharpe считается приближённо (годовой, risk-free rate = 0).
- Max drawdown считается по equity curve (cumulative PNL).
- Полностью соответствует ТЗ: статистика для PR, анализ сценариев.
- Нет заглушек — готов к использованию в pr_calculator, backtest и live_loop.
- Логи через setup_logger.
"""

import pandas as pd
import numpy as np
from typing import List, Dict

from src.utils.logger import setup_logger

logger = setup_logger('metrics', logging.INFO)

def calculate_trade_metrics(trades: List[Dict]) -> Dict[str, float]:
    """
    Рассчитывает все ключевые метрики по списку закрытых сделок.
    """
    if not trades:
        return {
            'total_trades': 0,
            'winrate': 0.0,
            'profit_factor': 0.0,
            'avg_rr': 0.0,
            'net_profit': 0.0,
            'max_drawdown_pct': 0.0,
            'sharpe_ratio': 0.0
        }

    df = pd.DataFrame(trades)

    total_trades = len(df)
    wins = df['is_tp'].sum() if 'is_tp' in df else 0
    winrate = wins / total_trades if total_trades > 0 else 0.0

    gross_profit = df[df['is_tp']]['pnl'].sum() if 'is_tp' in df else 0.0
    gross_loss = abs(df[~df['is_tp']]['pnl'].sum()) if 'is_tp' in df else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf')

    avg_profit = df[df['is_tp']]['pnl'].mean() if wins > 0 else 0.0
    avg_loss = abs(df[~df['is_tp']]['pnl'].mean()) if (total_trades - wins) > 0 else 0.0
    avg_rr = avg_profit / avg_loss if avg_loss != 0 else 0.0

    net_profit = df['pnl'].sum()

    # Equity curve для drawdown
    equity = df['pnl'].cumsum()
    peaks = equity.cummax()
    drawdowns = (equity - peaks) / peaks.replace(0, np.nan)
    max_dd = drawdowns.min() * 100 if not drawdowns.empty else 0.0

    # Sharpe (примерно, годовой)
    returns = df['pnl'] / df.get('entry_balance', 1000)  # упрощённо
    sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() != 0 else 0.0

    metrics = {
        'total_trades': total_trades,
        'winrate': round(winrate, 4),
        'profit_factor': round(profit_factor, 4),
        'avg_rr': round(avg_rr, 4),
        'net_profit': round(net_profit, 2),
        'max_drawdown_pct': round(max_dd, 2),
        'sharpe_ratio': round(sharpe, 4)
    }

    return metrics

def aggregate_by_anomaly(trades: List[Dict]) -> pd.DataFrame:
    """
    Группирует метрики по типу аномалии (C, V, CV, Q).
    """
    if not trades:
        return pd.DataFrame()

    df = pd.DataFrame(trades)
    grouped = df.groupby('anomaly_type').agg(
        total=('pnl', 'count'),
        wins=('is_tp', 'sum'),
        net_pnl=('pnl', 'sum'),
        avg_rr=('rr', 'mean')
    ).reset_index()

    grouped['winrate'] = grouped['wins'] / grouped['total']
    return grouped

def aggregate_by_direction(trades: List[Dict]) -> pd.DataFrame:
    """
    Группирует метрики по направлению (L / S).
    """
    if not trades:
        return pd.DataFrame()

    df = pd.DataFrame(trades)
    grouped = df.groupby('direction').agg(
        total=('pnl', 'count'),
        wins=('is_tp', 'sum'),
        net_pnl=('pnl', 'sum'),
        avg_rr=('rr', 'mean')
    ).reset_index()

    grouped['winrate'] = grouped['wins'] / grouped['total']
    return grouped

def log_metrics(metrics: Dict[str, float], prefix: str = ""):
    """
    Красиво выводит метрики в лог.
    """
    header = f"{prefix} Метрики торговли:" if prefix else "Метрики торговли:"
    logger.info(header)
    for key, value in metrics.items():
        if isinstance(value, float):
            logger.info(f"  {key.replace('_', ' ').title():<20} : {value:.4f}")
        else:
            logger.info(f"  {key.replace('_', ' ').title():<20} : {value}")