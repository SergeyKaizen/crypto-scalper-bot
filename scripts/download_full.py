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
7. Работает асинхронно (asyncio) + retry при ошибках
8. Поддерживает разные профили (phone_tiny / colab / server)

Запуск:
    python scripts/download_full.py --hardware phone_tiny
    python scripts/download_full.py --hardware server

Рекомендуется запускать:
- раз в сутки на сервере (обновление списка + догон новых свечей)
- один раз при первом запуске на телефоне/colab
"""

import asyncio
import logging
from datetime import datetime
from typing import List

import polars as pl

from src.core.config import load_config
from src.data.downloader import Downloader
from src.data.storage import Storage
from src.utils.logger import setup_logger

setup_logger()  # Инициализация глобального логгера
logger = logging.getLogger(__name__)


async def main():
    config = load_config()  # Автоопределение или из аргументов
    downloader = Downloader(config)
    storage = Storage(config)

    logger.info("=== Начало полной загрузки данных ===")
    logger.info("Hardware profile: %s", config.get("hardware_profile", "auto"))
    logger.info("Max coins: %d, Max history candles: %d", 
                config["max_coins"], config["data"]["max_history_candles"])

    # 1. Обновляем список монет
    logger.info("Обновление списка фьючерсных монет...")
    current_coins = await downloader.update_markets_list()

    # 2. Фильтруем только те, что разрешены в конфиге
    allowed_coins = current_coins[:config["max_coins"]]
    logger.info("Будет обработано монет: %d (из %d доступных)", len(allowed_coins), len(current_coins))

    # 3. Скачиваем полную историю для каждой монеты
    for symbol in allowed_coins:
        logger.info("Обработка %s...", symbol)

        # Проверяем, есть ли уже данные
        last_ts = await storage.get_last_timestamp(symbol, "1m")
        if last_ts:
            logger.info("  Уже есть данные до %s, догоняем live...", datetime.fromtimestamp(last_ts / 1000))
            await downloader.fetch_new_candles(symbol, "1m")
        else:
            logger.info("  Полная история отсутствует → скачиваем с листинга")
            await downloader.download_full_history(symbol, "1m")

        # Опционально — resample на higher TF
        if config["data"]["resample_higher_tf"]:
            logger.info("  Resampling на higher TF для %s...", symbol)
            for tf in config["timeframes"][1:]:  # кроме 1m
                await downloader.resampler.resample(symbol, tf)

    logger.info("=== Полная загрузка данных завершена ===")
    logger.info("Монет в базе: %d", len(await storage.get_current_coins()))

    await downloader.close()
    await storage.close()


if __name__ == "__main__":
    # Поддержка аргументов (можно расширить argparse)
    import sys
    hardware = "phone_tiny" if "--hardware" not in sys.argv else sys.argv[sys.argv.index("--hardware") + 1]

    config = load_config(hardware_profile=hardware)
    asyncio.run(main())