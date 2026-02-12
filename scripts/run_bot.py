# scripts/run_bot.py
"""
Главный скрипт запуска бота в live-режиме (real / virtual).

Запуск:
    python scripts/run_bot.py --hardware phone_tiny --mode balanced --trading virtual
    python scripts/run_bot.py --hardware server --mode aggressive --trading real

Аргументы:
    --hardware       : phone_tiny / colab / server (по умолчанию автоопределение)
    --mode           : conservative / balanced / aggressive / custom (по умолчанию balanced)
    --trading        : real / virtual (по умолчанию virtual)
    --log-level      : DEBUG / INFO / WARNING / ERROR (по умолчанию INFO)
    --no-warmup      : отключить warm-up фазу
    --symbol         : ограничить торговлю одной монетой (для теста)

Логика:
1. Парсит аргументы
2. Загружает конфиг (default + hardware + mode)
3. Инициализирует Downloader, Storage, FeatureEngine, InferenceEngine, LiveLoop и т.д.
4. Запускает warm-up (если включён)
5. Запускает LiveLoop.start() — основной цикл
6. Ловит Ctrl+C → graceful shutdown (закрытие соединений, сохранение модели)

На телефоне:
- low_power_mode → sleep чаще, quiet_mode реже, PR recalc реже
- max_coins=5, max_tf=3

На сервере:
- parallel=true, GPU, max_coins=150+
"""

import argparse
import asyncio
import logging
import signal
import sys
from datetime import datetime

from src.core.config import load_config
from src.utils.logger import setup_logger
from src.trading.live_loop import LiveLoop
from src.data.downloader import Downloader
from src.data.storage import Storage
from src.model.inference import InferenceEngine
from src.backtest.virtual_trader import VirtualTrader

setup_logger()
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Запуск Crypto Scalper Bot в live-режиме")
    parser.add_argument("--hardware", default=None, choices=["phone_tiny", "colab", "server"],
                        help="Профиль железа (по умолчанию автоопределение)")
    parser.add_argument("--mode", default="balanced", choices=["conservative", "balanced", "aggressive", "custom"],
                        help="Режим торговли")
    parser.add_argument("--trading", default="virtual", choices=["real", "virtual"],
                        help="Режим исполнения ордеров")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Уровень логирования")
    parser.add_argument("--no-warmup", action="store_true", help="Отключить warm-up фазу")
    parser.add_argument("--symbol", help="Ограничить торговлю одной монетой (для теста)")
    return parser.parse_args()


def setup_logging(level: str):
    """Переопределяем уровень логирования из аргументов"""
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Неверный уровень логирования: {level}")
    logging.getLogger().setLevel(numeric_level)
    logger.info("Уровень логирования установлен: %s", level)


async def main():
    args = parse_args()
    setup_logging(args.log_level)

    logger.info("Запуск Crypto Scalper Bot | %s | %s | %s", 
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"), args.hardware or "auto", args.mode)

    # 1. Загрузка конфига
    config = load_config(hardware_profile=args.hardware, trading_mode=args.mode)
    config["trading"]["mode"] = args.trading

    if args.no_warmup:
        config["warm_up"]["enabled"] = False

    if args.symbol:
        config["current_coins"] = [args.symbol]
        logger.warning("Ограничение торговли только одной монетой: %s", args.symbol)

    # 2. Инициализация компонентов
    downloader = Downloader(config)
    storage = Storage(config)
    live_loop = LiveLoop(config)

    # 3. Warm-up (если включён)
    if config["warm_up"]["enabled"]:
        logger.info("Запуск warm-up фазы (%d свечей)", config["warm_up"]["duration_candles"])
        await live_loop._warm_up_phase()
    else:
        logger.info("Warm-up отключён")

    # 4. Запуск основного цикла
    logger.info("Запуск главного цикла live_loop...")
    await live_loop.start()

    # Graceful shutdown
    def shutdown(signal_received):
        logger.info("Получен сигнал %s → graceful shutdown...", signal_received)
        live_loop.stop()
        asyncio.run(downloader.close())
        asyncio.run(storage.close())

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # Держим event loop живым
    while live_loop.running:
        await asyncio.sleep(1)

    logger.info("Бот остановлен")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Остановка по Ctrl+C")
    except Exception as e:
        logger.critical("Критическая ошибка в live_loop: %s", e, exc_info=True)
        sys.exit(1)