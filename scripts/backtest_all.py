# scripts/backtest_all.py
"""
Скрипт полного бэктеста всех монет в модуле.

Что делает:
1. Загружает конфиг (hardware + trading_mode)
2. Получает список всех монет из БД/модуля монет
3. Для каждой монеты запускает бэктест (BacktestEngine)
4. Собирает все TradeResult → передаёт в PRCalculator
5. Выводит общую статистику (winrate, profit factor, drawdown, Sharpe)
6. Сохраняет результаты в CSV (backtest_results.csv)
7. Поддерживает parallel (joblib) — ускорение на сервере в 10–20 раз
8. На телефоне — только 5 монет, без parallel

Запуск:
    python scripts/backtest_all.py --hardware phone_tiny --mode balanced
    python scripts/backtest_all.py --hardware server --timeframe 5m

Аргументы:
    --hardware: phone_tiny / colab / server
    --mode: conservative / balanced / aggressive / custom
    --timeframe: 1m / 5m / 15m (по умолчанию 1m)
    --limit: ограничение монет (для теста)
"""

import argparse
import asyncio
import logging
from pathlib import Path
from typing import List, Dict

import polars as pl
from joblib import Parallel, delayed

from src.core.config import load_config
from src.data.storage import Storage
from src.backtest.engine import BacktestEngine
from src.backtest.pr_calculator import PRCalculator
from src.utils.metrics import calculate_strategy_metrics, log_strategy_metrics
from src.utils.logger import setup_logger

setup_logger()
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Полный бэктест всех монет")
    parser.add_argument("--hardware", default="phone_tiny", choices=["phone_tiny", "colab", "server"],
                        help="Профиль железа")
    parser.add_argument("--mode", default="balanced", choices=["conservative", "balanced", "aggressive", "custom"],
                        help="Режим торговли")
    parser.add_argument("--timeframe", default="1m", choices=["1m", "3m", "5m", "10m", "15m"],
                        help="Основной таймфрейм бэктеста")
    parser.add_argument("--limit", type=int, help="Ограничить кол-во монет (для теста)")
    return parser.parse_args()


async def run_backtest_for_symbol(symbol: str, timeframe: str, config: Dict) -> List[TradeResult]:
    """Запуск бэктеста для одной монеты"""
    logger.info("Бэктест %s (%s) начат", symbol, timeframe)

    engine = BacktestEngine(config)
    results = engine.run_backtest(symbol, timeframe)

    logger.info("Бэктест %s (%s) завершён: %d сделок", symbol, timeframe, len(results))
    return results


def main():
    args = parse_args()
    config = load_config(hardware_profile=args.hardware, trading_mode=args.mode)

    storage = Storage(config)

    # Получаем список монет
    symbols = asyncio.run(storage.get_current_coins())
    if args.limit:
        symbols = symbols[:args.limit]

    logger.info("Найдено монет: %d (ограничение: %s)", len(symbols), args.limit or "нет")

    # Parallel (только на сервере)
    if config.get("parallel", False):
        logger.info("Запуск parallel бэктеста (joblib)")
        all_results = Parallel(n_jobs=-1)(
            delayed(run_backtest_for_symbol)(symbol, args.timeframe, config) for symbol in symbols
        )
    else:
        all_results = []
        for symbol in symbols:
            results = asyncio.run(run_backtest_for_symbol(symbol, args.timeframe, config))
            all_results.append(results)

    # Сбор всех сделок
    all_trades = []
    for symbol, results in zip(symbols, all_results):
        for trade in results:
            trade.symbol = symbol  # Добавляем символ в TradeResult
            all_trades.append(trade)

    # Расчёт общей статистики
    metrics = calculate_strategy_metrics(all_trades)
    log_strategy_metrics(metrics)

    # Сохранение результатов
    df = pl.DataFrame([{
        "symbol": t.position.symbol,
        "entry_time": t.position.entry_time,
        "entry_price": t.position.entry_price,
        "exit_time": t.exit_time,
        "exit_price": t.exit_price,
        "pnl_pct": t.pnl_pct,
        "pnl_usdt": t.pnl_usdt,
        "is_win": t.is_win,
        "reason": t.reason
    } for t in all_trades])

    output_path = Path("backtest_results.csv")
    df.write_csv(output_path)
    logger.info("Результаты бэктеста сохранены: %s (%d сделок)", output_path, len(df))

    # Обновляем PR
    pr_calc = PRCalculator(config)
    for trade in all_trades:
        pr_calc.add_trade(trade.position.symbol, trade)

    top_coins = pr_calc.get_top_coins(10)
    logger.info("Топ-10 монет по PR:")
    for symbol, pr in top_coins:
        logger.info("  %s: PR = %.4f", symbol, pr)


if __name__ == "__main__":
    main()