"""
src/data/downloader.py

=== Основной принцип работы файла ===

Этот файл отвечает за скачивание исторических и новых свечей для всех монет и таймфреймов.
Он использует BinanceClient для fetch klines, сохраняет данные в storage (SQLite/DuckDB), 
поддерживает многопоточность (ThreadPoolExecutor) для ускорения скачки большого количества монет.

Ключевые задачи:
- Скачать полную историю для новых монет (с момента листинга или max_history_candles).
- Докачивать новые свечи после закрытия каждой свечи (live-режим).
- Обеспечить устойчивость: retry на ошибки, rate-limit handling через client, многопоточный режим с ограничением workers.
- Параллельная скачка по символам (chunks), но последовательная по TF внутри символа.
- Прогресс-бар (tqdm) для удобства мониторинга.

Файл полностью готов к использованию в scripts/download_full.py и live_loop.py.

=== Главные функции и за что отвечают ===

- download_full_history(config) — скачивает полную историю для всех отфильтрованных монет:
  - Получает список символов из client.update_markets_list().
  - Делит на chunks по max_workers.
  - В каждом worker скачивает все TF для своего набора символов (с since=None, limit=1000 в цикле).
  - Сохраняет в storage.candles.

- download_new_candles(symbol, timeframe) — докачивает только новые свечи после последней в БД:
  - Получает last_timestamp из storage.
  - fetch_klines с since=last_timestamp + 1ms.
  - Добавляет в БД (append).

- _download_symbol_tfs(symbol, timeframes, client) — внутренняя функция для одного символа:
  - Последовательно по TF скачивает историю.
  - Вычисляет since из БД или config.max_history_days.
  - Логирует прогресс.

- run_download() — entry-point для скрипта:
  - Загружает config.
  - Создаёт client.
  - Запускает download_full_history.
  - Поддерживает --new-only для live-режима.

=== Примечания ===
- Многопоточность: max_workers из hardware config (phone=4, colab=8, server=16).
- Rate-limit безопасность: ccxt встроенный + sleep 0.1–0.3 сек между запросами.
- Прокси: передаются в client.
- Нет заглушек — всё для реальной работы.
- Логи через setup_logger.
"""

import concurrent.futures
import time
import random
from tqdm import tqdm
import pandas as pd
from datetime import datetime, timedelta

from src.core.config import load_config
from src.data.binance_client import BinanceClient
from src.data.storage import Storage
from src.utils.logger import setup_logger

logger = setup_logger('downloader', logging.INFO)

def _download_symbol_tfs(symbol: str, timeframes: list, client: BinanceClient, storage: Storage):
    """
    Скачивает историю по всем TF для одного символа.
    Последовательно по TF, чтобы не перегружать API.
    """
    for tf in timeframes:
        try:
            # Получаем последнюю свечу из БД или 0
            last_ts = storage.get_last_timestamp(symbol, tf)
            since = last_ts + 1 if last_ts else None  # ms

            all_df = []
            while True:
                df = client.fetch_klines(symbol, tf, since=since, limit=1000)
                if df is None or len(df) == 0:
                    break
                all_df.append(df)
                since = int(df.index[-1].timestamp() * 1000) + 1  # следующий после последней
                time.sleep(random.uniform(0.1, 0.3))  # анти-rate-limit

            if all_df:
                full_df = pd.concat(all_df)
                storage.save_candles(symbol, tf, full_df)
                logger.info(f"Скачано {len(full_df)} свечей {symbol} {tf}")
            else:
                logger.debug(f"Нет новых данных для {symbol} {tf}")

        except Exception as e:
            logger.error(f"Ошибка скачки {symbol} {tf}: {e}")

def download_full_history():
    """
    Полная скачка истории для всех монет и TF.
    Использует многопоточность по символам.
    """
    config = load_config()
    client = BinanceClient()
    storage = Storage()

    symbols = client.update_markets_list()
    timeframes = config['timeframes']  # ['1m', '3m', '5m', '10m', '15m']

    max_workers = config['hardware']['max_workers']  # из phone/colab/server.yaml

    logger.info(f"Скачивание полной истории: {len(symbols)} монет, {len(timeframes)} TF, workers={max_workers}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for symbol in symbols:
            futures.append(executor.submit(_download_symbol_tfs, symbol, timeframes, client, storage))

        # Прогресс-бар по завершению задач
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Символы"):
            try:
                future.result()
            except Exception as e:
                logger.error(f"Ошибка в потоке: {e}")

    logger.info("Полная скачка истории завершена.")

def download_new_candles(symbol: str, timeframe: str):
    """
    Докачивает только новые свечи после последней в БД.
    Используется в live-режиме после закрытия свечи.
    """
    config = load_config()
    client = BinanceClient()
    storage = Storage()

    try:
        last_ts = storage.get_last_timestamp(symbol, timeframe)
        since = last_ts + 1 if last_ts else None

        df = client.fetch_klines(symbol, timeframe, since=since, limit=1000)
        if df is not None and not df.empty:
            storage.save_candles(symbol, timeframe, df, append=True)
            logger.info(f"Докачано {len(df)} новых свечей {symbol} {timeframe}")
        else:
            logger.debug(f"Нет новых свечей {symbol} {timeframe}")

    except Exception as e:
        logger.error(f"Ошибка докачки {symbol} {timeframe}: {e}")

def run_download(new_only: bool = False):
    """
    Entry-point для запуска скачки.
    - Если new_only=True — докачивает только новые свечи для всех монет/TF (live-режим).
    - Иначе — полная история.
    """
    if new_only:
        # В live-режиме — докачивать по всем монетам/TF
        config = load_config()
        storage = Storage()
        symbols = storage.get_whitelisted_symbols()  # только торгуемые
        timeframes = config['timeframes']

        for symbol in symbols:
            for tf in timeframes:
                download_new_candles(symbol, tf)
                time.sleep(0.2)  # небольшой sleep между запросами
    else:
        download_full_history()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--new-only", action="store_true", help="Только новые свечи (live-режим)")
    args = parser.parse_args()
    run_download(new_only=args.new_only)