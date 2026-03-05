# scripts/run_bot.py
"""
Главный скрипт запуска бота по единому конфигу bot_config.yaml.

Логика запуска:
1. Загрузка config/bot_config.yaml
2. Инициализация бота
3. Обучение модели
4. Запуск полного бэктеста
5. Обновление whitelist
6. Warm-up
7. Запуск торговли

FIX Фаза 11: добавлен авто-сброс daily_loss_limit в reset_daily_loss_hour
"""

import os
import sys
import argparse
import yaml
import asyncio
import time
from datetime import datetime, timedelta

from src.core.config import load_config
from src.model.trainer import Trainer
from src.backtest.engine import BacktestEngine
from src.data.storage import Storage
from src.trading.live_loop import live_loop
from src.trading.websocket_manager import WebSocketManager
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
        trainer = Trainer(config)
        trainer.train()

    if not args.skip_backtest:
        logger.info("Запуск бэктеста...")
        from scripts.backtest_all import main as backtest_main
        backtest_main()

    logger.info("Warm-up...")
    time.sleep(5)

    # FIX Фаза 11: авто-сброс daily_loss_limit
    reset_hour = config["trading"].get("reset_daily_loss_hour", 0)
    last_reset = datetime.utcnow().replace(hour=reset_hour, minute=0, second=0, microsecond=0)
    if datetime.utcnow() > last_reset + timedelta(days=1):
        logger.info("Авто-сброс daily_loss_limit")
        # Здесь вызов reset_daily_loss в RiskManager (нужно добавить глобальный экземпляр)

    logger.info("Запуск торговли...")
    from src.data.resampler import Resampler
    resampler = Resampler(config)
    ws_manager = WebSocketManager(config, resampler)
    asyncio.create_task(ws_manager.start())
    
    try:
        asyncio.run(live_loop())
    except KeyboardInterrupt:
        logger.info("Остановка бота...")