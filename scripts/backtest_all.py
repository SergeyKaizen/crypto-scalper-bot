# scripts/backtest_all.py
"""
Скрипт полного бэктеста по всем монетам.

Цель:
- Выявить текущие лучшие монеты для торговли на данный момент (по PR)
- Обновить белый список монет для live-торговли
- Сохранить статистику всех прошедших фильтр монет

Логика (по ТЗ):
- Анализируем период из конфига (в свечах или часах)
- Для каждой монеты запускаем виртуальный бэктест → считаем PR
- Оставляем только те, у кого:
  - PR ≥ min_pr (из конфига)
  - кол-во сделок ≥ min_trades
  - монета существует на бирже ≥ min_age_months
- Торгуем ВСЕ прошедшие монеты (никакого топ-N)
- Параллельный запуск (joblib) для ускорения на сервере/colab
- На телефоне (tiny) — последовательно и только топ-5 монет (для тестового запуска)

Запуск:
python scripts/backtest_all.py --mode full --trading_mode balanced
"""

import argparse
import time
from datetime import datetime
from typing import List, Dict
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from src.core.config import load_config
from src.backtest.engine import BacktestEngine  # основной движок виртуального бэктеста
from src.backtest.pr_calculator import PRCalculator
from src.data.binance_client import BinanceClient
from src.data.storage import Storage
from src.utils.logger import get_logger

logger = get_logger(__name__)

def run_backtest_for_symbol(symbol: str, config: dict) -> Dict:
    """
    Запускает полный виртуальный бэктест для одной монеты.
    
    Возвращает словарь с результатами или None если монета не прошла фильтр.
    """
    try:
        logger.info(f"Запуск бэктеста для {symbol}")
        
        engine = BacktestEngine(config, symbol=symbol)
        results = engine.run_full_backtest()
        
        if not results or "trades_count" not in results:
            logger.warning(f"{symbol} — нет сделок или ошибка")
            return None
        
        pr_calc = PRCalculator(config)
        pr_value = pr_calc.calculate_pr(results)
        
        # Проверяем фильтры из ТЗ
        min_pr = config["filter"]["min_pr"]
        min_trades = config["filter"]["min_trades"]
        min_age_months = config["filter"]["min_age_months"]
        
        age_months = results.get("symbol_age_months", 0)
        
        if (pr_value < min_pr or
            results["trades_count"] < min_trades or
            age_months < min_age_months):
            logger.info(f"{symbol} не прошла фильтр: PR={pr_value:.2f}, сделок={results['trades_count']}, возраст={age_months} мес")
            return None
        
        return {
            "symbol": symbol,
            "pr": pr_value,
            "trades_count": results["trades_count"],
            "winrate": results.get("winrate", 0) * 100,
            "profit_factor": results.get("profit_factor", 0),
            "max_drawdown": results.get("max_drawdown", 0),
            "symbol_age_months": age_months,
            "best_tf": results.get("best_tf", "unknown"),
            "best_window": results.get("best_window", "unknown"),
            "best_anomaly_type": results.get("best_anomaly_type", "unknown")
        }
    
    except Exception as e:
        logger.error(f"Ошибка бэктеста {symbol}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Полный бэктест по всем монетам")
    parser.add_argument("--trading_mode", type=str, default="balanced",
                        choices=["conservative", "balanced", "aggressive", "custom"],
                        help="Режим агрессивности")
    parser.add_argument("--hardware", type=str, default="auto",
                        choices=["auto", "phone_tiny", "colab", "server"],
                        help="Режим железа")
    args = parser.parse_args()

    # Загружаем конфиг с merge default + hardware + trading_mode
    config = load_config(trading_mode=args.trading_mode, hardware_mode=args.hardware)
    
    logger.info(f"Запуск backtest_all.py | режим: {args.trading_mode} | hardware: {config['hardware_mode']}")

    # Получаем актуальный список фьючерсных монет
    exchange = BinanceClient(config)
    all_symbols = exchange.get_futures_symbols(active_only=True)
    
    # Фильтр по минимальному возрасту монеты (уже в run_backtest_for_symbol, но можно заранее отсечь)
    min_age = config["filter"]["min_age_months"]
    if min_age > 0:
        listing_dates = exchange.get_listing_dates()  # метод должен быть в BinanceClient
        all_symbols = [s for s in all_symbols if listing_dates.get(s, 0) >= min_age]

    logger.info(f"Найдено {len(all_symbols)} подходящих монет для анализа")

    # Параллельный запуск (на телефоне n_jobs=1)
    n_jobs = 1 if config["hardware_mode"] == "phone_tiny" else -1  # -1 = все ядра
    
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(run_backtest_for_symbol)(symbol, config)
        for symbol in tqdm(all_symbols, desc="Бэктест монет")
    )

    # Фильтруем None и собираем прошедшие монеты
    passed_symbols = [r for r in results if r is not None]
    
    if not passed_symbols:
        logger.warning("НИ ОДНА монета не прошла фильтр PR / сделок / возраста")
        return

    # Сортируем по PR (для удобства просмотра, но торгуем всех)
    passed_symbols.sort(key=lambda x: x["pr"], reverse=True)

    # Выводим таблицу в консоль
    df = pd.DataFrame(passed_symbols)
    print("\n" + "="*80)
    print(f"Монеты, прошедшие фильтр ({len(passed_symbols)} шт):")
    print(df[["symbol", "pr", "trades_count", "winrate", "profit_factor", "max_drawdown"]].round(2))
    print("="*80 + "\n")

    # Сохраняем в CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f"backtest_results_{args.trading_mode}_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Результаты сохранены в {csv_path}")

    # Здесь можно обновить белый список в Storage или config
    # storage = Storage(config)
    # storage.update_whitelist([r["symbol"] for r in passed_symbols])


if __name__ == "__main__":
    main()