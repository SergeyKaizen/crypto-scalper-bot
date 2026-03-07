# scripts/download_full.py
"""
Скрипт полной начальной загрузки исторических данных.

Что делает:
1. Загружает актуальный список всех USDT perpetual фьючерсов
2. Добавляет новые монеты в БД/модуль монет (если их ещё нет)
3. Удаляет delisted монеты и их данные (если есть)
4. Для каждой монеты скачивает полную историю 1m свечей от даты листинга
5. Догоняет live-свечи, если история уже есть
6. Учитывает лимиты железа (max_history_candles из конфига)
7. Работает с функциями из downloader.py

Запуск:
    python scripts/download_full.py
    python scripts/download_full.py --new-only
"""

import argparse
from src.core.config import load_config
from src.data.downloader import download_full_history, run_download
from src.utils.logger import setup_logger

logger = setup_logger("download_full", logging.INFO)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--new-only", action="store_true", help="Только докачать новые свечи (live-режим)")
    args = parser.parse_args()

    config = load_config()
    logger.info(f"=== Запуск скачивания данных (hardware: {config.get('hardware', {}).get('profile', 'default')}) ===")

    if args.new_only:
        run_download(new_only=True)
    else:
        download_full_history()

    logger.info("=== Скачивание завершено ===")

if __name__ == "__main__":
    main()