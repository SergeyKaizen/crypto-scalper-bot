# scripts/run_bot.py
"""
Главный скрипт запуска бота (интрадей-режим).

После всех фиксов:
- Обучение и retrain происходят автоматически внутри live_loop
- Warm-up происходит автоматически внутри live_loop
- Бэктест можно запустить отдельно через backtest_all.py
- Бот запускается одной командой и работает 24/7
"""

import argparse
import asyncio
import logging
import signal
import sys

from src.core.config import load_config
from src.trading.live_loop import live_loop
from src.utils.logger import setup_logger

logger = setup_logger("run_bot", logging.INFO)

def parse_args():
    parser = argparse.ArgumentParser(description="Запуск Crypto Scalper Bot (интрадей)")
    parser.add_argument("--skip_backtest", action="store_true",
                        help="Пропустить предварительный бэктест (рекомендуется после первого запуска)")
    parser.add_argument("--config", type=str, default="config/bot_config.yaml",
                        help="Путь к конфигу (по умолчанию bot_config.yaml)")
    return parser.parse_args()

def signal_handler(sig, frame):
    """Graceful shutdown — только лог, позиции НЕ закрываются"""
    logger.info("Получен сигнал остановки. Graceful shutdown...")
    # State persistence вызывается внутри live_loop
    sys.exit(0)

async def main():
    args = parse_args()
    config = load_config()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    if not args.skip_backtest:
        logger.info("Запуск предварительного бэктеста (можно пропустить через --skip_backtest)...")
        try:
            from scripts.backtest_all import main as backtest_main
            backtest_main()
        except Exception as e:
            logger.warning(f"Бэктест пропущен из-за ошибки: {e} (бот продолжит работу)")

    logger.info("Запуск live-торговли (интрадей-режим)...")
    try:
        asyncio.run(live_loop())  # watchdog и state persistence уже внутри live_loop
    except KeyboardInterrupt:
        logger.info("Бот остановлен пользователем.")
    except Exception as e:
        logger.exception(f"Критическая ошибка в live_loop: {e}")

if __name__ == "__main__":
    main()