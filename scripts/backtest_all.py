# scripts/backtest_all.py
"""
Параллельный бэктест по всем монетам из whitelist (или топ-N по PR).

Ключевые особенности (по ТЗ и твоим уточнениям):
- Запускает BacktestEngine для каждой монеты в параллели (ThreadPoolExecutor)
- Бэктест берёт последние 250 свечей на каждом из 5 TF
- PR считается по новой формуле (expected tp_length/sl_length) — без sum по истории
- Поддержка hybrid-режима: виртуальная позиция + проверка маркировки для live
- Фильтр монет по PR_value, min_trades, min_age_months
- Обновляет whitelist после бэктеста
- Сохраняет PR-снимки для каждой монеты

Запуск:
python scripts/backtest_all.py
python scripts/backtest_all.py --top_n 20
"""

import argparse
import time
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.backtest.engine import BacktestEngine
from src.data.storage import Storage
from src.core.config import load_config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

def run_backtest_for_symbol(config: dict, symbol: str) -> Dict:
    logger.info(f"Запуск бэктеста для {symbol}")
    try:
        engine = BacktestEngine(config, symbol)
        results = engine.run_full_backtest()  # hybrid + новый PR
        results["symbol"] = symbol
        results["pr_value"] = results.get("pr_value", 0.0)  # новая формула PR
        results["total_trades"] = results.get("total_trades", 0)
        return results
    except Exception as e:
        logger.error(f"Ошибка бэктеста для {symbol}: {e}")
        return {"symbol": symbol, "error": str(e), "pr_value": 0.0, "total_trades": 0}

def filter_and_update_whitelist(config: dict, all_results: List[Dict]):
    min_pr = config["filters"].get("min_pr", 1.0)
    min_trades = config["filters"].get("min_trades", 5)
    min_age_months = config["filters"].get("min_age_months", 6)

    filtered = []
    for res in all_results:
        symbol = res["symbol"]
        pr_value = res.get("pr_value", 0.0)
        trades = res.get("total_trades", 0)
        error = res.get("error")

        if error:
            logger.warning(f"{symbol} исключён из whitelist: ошибка бэктеста")
            continue

        if pr_value < min_pr:
            logger.info(f"{symbol} исключён: PR_value {pr_value:.4f} < {min_pr}")
            continue

        if trades < min_trades:
            logger.info(f"{symbol} исключён: сделок {trades} < {min_trades}")
            continue

        filtered.append(res)

    if filtered:
        storage = Storage()
        storage.add_to_whitelist(filtered)
        logger.info(f"Whitelist обновлён: {len(filtered)} монет прошли фильтр (hybrid PR)")
    else:
        logger.warning("Ни одна монета не прошла фильтр → whitelist остался прежним")

def main():
    parser = argparse.ArgumentParser(description="Параллельный бэктест по всем монетам + обновление whitelist")
    parser.add_argument("--hardware", default="phone_tiny", choices=["phone_tiny", "colab", "server"],
                        help="Профиль железа")
    parser.add_argument("--mode", default="balanced", choices=["conservative", "balanced", "aggressive", "custom"],
                        help="Режим торговли")
    parser.add_argument("--top_n", type=int, default=None,
                        help="Ограничить топ-N монет по PR_value (если не указан — все из whitelist)")
    args = parser.parse_args()

    config = load_config()
    storage = Storage()

    symbols = storage.get_whitelisted_symbols()
    if not symbols:
        logger.warning("Whitelist пуст → бэктест невозможен")
        return

    if args.top_n:
        symbols = symbols[:args.top_n]

    logger.info(f"Запуск параллельного бэктеста по {len(symbols)} монетам (hybrid-режим)")

    windows = config["backtest"].get("walk_forward_windows", 5)
    for window in range(windows):
        logger.info(f"Walk-forward окно {window+1}/{windows}")
        results = []
        max_workers = config.get("hardware", {}).get("max_workers", 8)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_symbol = {executor.submit(run_backtest_for_symbol, config, sym): sym for sym in symbols}
            for future in as_completed(future_to_symbol):
                results.append(future.result())
        filter_and_update_whitelist(config, results)

    passed = len([r for r in results if "error" not in r and r.get("pr_value", 0) >= config["filters"].get("min_pr", 1.0)])
    logger.info(f"Бэктест завершён | Прошли фильтр: {passed}/{len(results)} монет")


if __name__ == "__main__":
    main()