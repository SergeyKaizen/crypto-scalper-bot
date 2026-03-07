# scripts/run_bot.py
"""
Главный скрипт запуска бота.
"""

import os
import sys
import argparse
import asyncio
import time
import subprocess
from datetime import datetime, timedelta

from src.core.config import load_config
from src.backtest.engine import BacktestEngine
from src.data.storage import Storage
from src.trading.live_loop import live_loop
from src.utils.logger import setup_logger

logger = setup_logger("run_bot", logging.INFO)

def parse_args():
    parser = argparse.ArgumentParser(description="Запуск Scalper Bot")
    parser.add_argument("--config", type=str, default="config/bot_config.yaml")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_backtest", action="store_true")
    return parser.parse_args()

def main():
    args = parse_args()
    config = load_config()

    if not args.skip_train:
        logger.info("Обучение модели...")
        subprocess.run([sys.executable, "scripts/train.py"])  # FIX: вызов скрипта (Trainer класса нет)

    if not args.skip_backtest:
        logger.info("Запуск бэктеста...")
        from scripts.backtest_all import main as backtest_main
        backtest_main()

    logger.info("Warm-up...")
    time.sleep(5)

    logger.info("Запуск торговли...")
    try:
        asyncio.run(live_loop())
    except KeyboardInterrupt:
        logger.info("Остановка бота...")

if __name__ == "__main__":
    main()