# scripts/backtest_all.py
"""
Параллельный бэктест по всем монетам из whitelist (или топ-N по PR).

Ключевые особенности (по ТЗ и твоим уточнениям):
- Запускает BacktestEngine для каждой монеты в параллели (ThreadPoolExecutor)
- Бэктест берёт последние 250 свечей на каждом из 5 TF (1m, 3m, 5m, 10m, 15m)
- PR считается по формулам ТЗ (PR_L, PR_S, PR_LS) — только кол-во сделок и длина TP/SL
- Фильтр монет по возрасту (min_age_months), минимальному количеству сделок и PR_LS
- Обновляет whitelist после бэктеста (только прошедшие фильтр)
- Сохраняет PR-снимки для каждой монеты в storage
- Логирование прогресса и финальной статистики
- Поддержка лимита монет (--top_n) для быстрого теста

Запуск:
python scripts/backtest_all.py
python scripts/backtest_all.py --top_n 20
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
    Возвращает словарь с результатами (PR_LS и другие метрики для фильтра).
    """
    logger.info(f"Запуск бэктеста для {symbol}")
    try:
        engine = BacktestEngine(config, symbol)
        results = engine.run_full_backtest()
        
        # Добавляем символ и ключевые метрики
        results["symbol"] = symbol
        results["pr_ls"] = results.get("pr_ls", 0.0)
        results["total_trades"] = results.get("total_trades", 0)
        
        return results
    except Exception as e:
        logger.error(f"Ошибка бэктеста для {symbol}: {e}")
        return {"symbol": symbol, "error": str(e), "pr_ls": 0.0, "total_trades": 0}

def filter_and_update_whitelist(config: dict, all_results: List[Dict]):
    """
    Фильтрует монеты по критериям и обновляет whitelist в storage.
    Критерии (из конфига):
    - min_pr_ls (минимальный PR_LS)
    - min_trades (минимальное кол-во сделок в бэктесте)
    - min_age_months (минимальный возраст монеты)
    """
    min_pr_ls = config["filters"].get("min_pr", 1.3)
    min_trades = config["filters"].get("min_trades", 50)
    min_age_months = config["filters"].get("min_age_months", 3)

    filtered = []
    for res in all_results:
        symbol = res["symbol"]
        pr_ls = res.get("pr_ls", 0.0)
        trades = res.get("total_trades", 0)
        error = res.get("error")

        if error:
            logger.warning(f"{symbol} исключён из whitelist: ошибка бэктеста")
            continue

        if pr_ls < min_pr_ls:
            logger.info(f"{symbol} исключён: PR_LS {pr_ls:.4f} < {min_pr_ls}")
            continue

        if trades < min_trades:
            logger.info(f"{symbol} исключён: сделок {trades} < {min_trades}")
            continue

        # Проверка возраста монеты (по первой свече в storage)
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
    parser.add_argument("--hardware", default="phone_tiny", choices=["phone_tiny", "colab", "server"],
                        help="Профиль железа")
    parser.add_argument("--mode", default="balanced", choices=["conservative", "balanced", "aggressive", "custom"],
                        help="Режим торговли")
    parser.add_argument("--top_n", type=int, default=None,
                        help="Ограничить топ-N монет по PR_LS (если не указан — все из whitelist)")
    args = parser.parse_args()

    config = load_config(hardware=args.hardware, mode=args.mode)  # ← ФИКС: передаём hardware и mode
    storage = Storage(config)

    # Получаем список монет для бэктеста
    # FIX Фаза 5: get_whitelist → get_whitelisted_symbols (согласование с Phase 1)
    symbols = storage.get_whitelisted_symbols()
    if not symbols:
        logger.warning("Whitelist пуст → бэктест невозможен")
        return

    if args.top_n:
        # Сортировка по PR_LS из последнего снимка
        pr_history = []
        for sym in symbols:
            df = storage.get_pr_history(sym, limit=1)
            pr_ls = df["pr_ls"].max() if not df.is_empty() else 0.0
            pr_history.append({"symbol": sym, "pr_ls": pr_ls})
        pr_history.sort(key=lambda x: x["pr_ls"], reverse=True)
        symbols = [x["symbol"] for x in pr_history[:args.top_n]]
        logger.info(f"Ограничение бэктеста до топ-{args.top_n} монет по PR_LS")

    logger.info(f"Запуск параллельного бэктеста по {len(symbols)} монетам")

    results = []
    # === ФИКС ПУНКТА 27: ОГРАНИЧЕННОЕ КОЛИЧЕСТВО ВОРКЕРОВ ===
    max_workers = config.get("hardware", {}).get("max_workers", 8)  # берём из конфига, fallback на 8
    logger.info(f"Запускаем ThreadPoolExecutor с {max_workers} воркерами (вместо mp.cpu_count())")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
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
    passed = len([r for r in results if "error" not in r and r.get("pr_ls", 0) >= config["filters"].get("min_pr", 1.3)])
    logger.info(f"Бэктест завершён | Прошли фильтр: {passed}/{len(results)} монет")


if __name__ == "__main__":
    main()