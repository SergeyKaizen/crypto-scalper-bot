# scripts/backtest_all.py
"""
Параллельный бэктест по всем монетам из списка (или из whitelist).

Ключевые особенности:
- Параллельный запуск BacktestEngine для каждой монеты (ThreadPoolExecutor / multiprocessing)
- Фильтр монет по возрасту (min_age_months), минимальному количеству сделок и PR
- Обновление whitelist после бэктеста (только монеты, прошедшие фильтр)
- Сохранение PR-снимков для каждой монеты в storage
- Логирование прогресса и финальной статистики
- Поддержка лимита монет для запуска (top_n из конфига или все)
- Полная совместимость с остальным проектом (inference, storage, virtual_trader)

Запуск:
python scripts/backtest_all.py
python scripts/backtest_all.py --top_n 50
"""

import argparse
import time
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp

from src.backtest.engine import BacktestEngine
from src.data.storage import Storage
from src.core.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

def run_backtest_for_symbol(config: dict, symbol: str) -> Dict[str, Any]:
    """
    Запуск бэктеста для одной монеты.
    Возвращает словарь с результатами (для фильтра и сохранения в storage).
    """
    logger.info(f"Запуск бэктеста для {symbol}")
    try:
        engine = BacktestEngine(config, symbol)
        results = engine.run_full_backtest()
        
        # Добавляем символ и ключевые метрики
        results["symbol"] = symbol
        results["pr"] = results.get("profit_factor", 0.0)
        results["trades_count"] = results.get("trades_count", 0)
        
        return results
    except Exception as e:
        logger.error(f"Ошибка бэктеста для {symbol}: {e}")
        return {"symbol": symbol, "error": str(e), "pr": 0.0, "trades_count": 0}

def filter_and_update_whitelist(config: dict, all_results: List[Dict]):
    """
    Фильтрует монеты по критериям и обновляет whitelist в storage.
    Критерии (из конфига):
    - min_pr (минимальный PR/profit_factor)
    - min_trades (минимальное кол-во сделок в бэктесте)
    - min_age_months (минимальный возраст монеты на бирже)
    """
    min_pr = config["backtest"].get("min_pr", 1.2)
    min_trades = config["backtest"].get("min_trades", 50)
    min_age_months = config["backtest"].get("min_age_months", 3)

    filtered = []
    for res in all_results:
        symbol = res["symbol"]
        pr = res.get("pr", 0.0)
        trades = res.get("trades_count", 0)
        error = res.get("error")

        if error:
            logger.warning(f"{symbol} исключён из whitelist: ошибка бэктеста")
            continue

        if pr < min_pr:
            logger.info(f"{symbol} исключён: PR {pr:.2f} < {min_pr}")
            continue

        if trades < min_trades:
            logger.info(f"{symbol} исключён: сделок {trades} < {min_trades}")
            continue

        # Проверка возраста монеты (примерно по первой свече в storage)
        df = Storage(config).load_candles(symbol, "1m", limit=1, min_timestamp=0)
        if df is None or df.is_empty():
            logger.warning(f"{symbol} исключён: нет данных")
            continue

        first_ts = df["open_time"].min() / 1000  # ms → sec
        age_months = (time.time() - first_ts) / (86400 * 30)
        if age_months < min_age_months:
            logger.info(f"{symbol} исключён: возраст {age_months:.1f} мес < {min_age_months}")
            continue

        filtered.append(res)

    if filtered:
        Storage(config).update_whitelist(filtered)
        logger.info(f"Whitelist обновлён: {len(filtered)} монет прошли фильтр")
    else:
        logger.warning("Ни одна монета не прошла фильтр → whitelist остался прежним")

def main():
    parser = argparse.ArgumentParser(description="Параллельный бэктест по всем монетам + обновление whitelist")
    parser.add_argument("--top_n", type=int, default=None,
                        help="Ограничить топ-N монет по PR (если не указан — все из whitelist)")
    args = parser.parse_args()

    config = load_config()
    storage = Storage(config)

    # Получаем список монет для бэктеста
    symbols = storage.get_whitelist()
    if not symbols:
        logger.warning("Whitelist пуст → бэктест невозможен")
        return

    if args.top_n:
        # Сортировка по PR из последнего снимка (если есть)
        pr_history = []
        for sym in symbols:
            df = storage.get_pr_history(sym, limit=1)
            pr = df["pr"].max() if not df.is_empty() else 0.0
            pr_history.append({"symbol": sym, "pr": pr})
        pr_history.sort(key=lambda x: x["pr"], reverse=True)
        symbols = [x["symbol"] for x in pr_history[:args.top_n]]
        logger.info(f"Ограничение бэктеста до топ-{args.top_n} монет по PR")

    logger.info(f"Запуск параллельного бэктеста по {len(symbols)} монетам")

    results = []
    with ThreadPoolExecutor(max_workers=mp.cpu_count()) as executor:
        future_to_symbol = {executor.submit(run_backtest_for_symbol, config, sym): sym for sym in symbols}
        for future in as_completed(future_to_symbol):
            sym = future_to_symbol[future]
            try:
                res = future.result()
                results.append(res)
            except Exception as e:
                logger.error(f"Ошибка в пуле для {sym}: {e}")

    # Фильтр и обновление whitelist
    filter_and_update_whitelist(config, results)

    # Финальная статистика
    passed = len([r for r in results if "error" not in r and r.get("pr", 0) >= config["backtest"].get("min_pr", 1.2)])
    logger.info(f"Бэктест завершён | Прошли фильтр: {passed}/{len(results)} монет")


if __name__ == "__main__":
    main()