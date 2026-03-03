"""
scripts/run_bot.py

Главный скрипт запуска бота по единому конфигу bot_config.yaml.

Логика запуска (строго по твоим требованиям):
1. Загрузка config/bot_config.yaml
2. Инициализация бота (все модули)
3. Обучение модели (trainer.py)
4. Запуск полного бэктеста (backtest_all.py) → расчёт PR по формулам ТЗ
5. Обновление торгового списка (whitelist)
6. Warm-up (прогрев кэша resampler и inference)
7. Запуск торговли (live_loop) в выбранном режиме (real/virtual)

Запуск:
python scripts/run_bot.py
python scripts/run_bot.py --skip_train --skip_backtest   # пропустить обучение и бэктест
"""

import os
import sys
import argparse
import yaml
import asyncio
import time
from datetime import datetime

from src.core.config import load_config
from src.model.trainer import Trainer
from src.backtest.engine import BacktestEngine
from src.data.storage import Storage
from src.trading.live_loop import live_loop
from src.trading.websocket_manager import WebSocketManager
from src.utils.logger import setup_logger

logger = setup_logger("run_bot", logging.INFO)

def parse_args():
    parser = argparse.ArgumentParser(description="Запуск Scalper Bot по единому конфигу")
    parser.add_argument("--config", type=str, default="config/bot_config.yaml",
                        help="Путь к главному конфиг-файлу")
    parser.add_argument("--skip_train", action="store_true", help="Пропустить обучение модели")
    parser.add_argument("--skip_backtest", action="store_true", help="Пропустить бэктест и обновление whitelist")
    parser.add_argument("--only_warmup_trade", action="store_true", help="Только warm-up и торговля (для тестов)")
    return parser.parse_args()

def main():
    args = parse_args()
    
    print("="*80)
    print("Scalper Bot Launcher (единый конфиг)")
    print("="*80)
    
    config = load_config()
    
    if not args.only_warmup_trade:
        if not args.skip_train:
            logger.info("Шаг 3: Обучение модели...")
            trainer = Trainer(config)
            trainer.train()
            logger.info("Обучение завершено.")

        if not args.skip_backtest:
            logger.info("Шаг 4: Запуск полного бэктеста...")
            from scripts.backtest_all import main as backtest_main
            backtest_main()
            logger.info("Бэктест завершён, whitelist обновлён.")

    logger.info("Шаг 6: Warm-up (прогрев кэша и модели)...")
    time.sleep(5)
    logger.info("Warm-up завершён.")

    logger.info("Шаг 7: Запуск торговли...")
    from src.data.resampler import Resampler
    resampler = Resampler(config)
    ws_manager = WebSocketManager(config, resampler)
    asyncio.create_task(ws_manager.start())
    
    try:
        asyncio.run(live_loop())
    except KeyboardInterrupt:
        logger.info("Остановка бота по запросу пользователя...")
    except Exception as e:
        logger.error(f"Критическая ошибка в LiveLoop: {e}")

if __name__ == "__main__":
    main()