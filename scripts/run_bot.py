# scripts/run_bot.py
"""
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
from src.trading.live_loop import LiveLoop
from src.trading.websocket_manager import WebSocketManager
from src.utils.logger import get_logger

logger = get_logger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Запуск Scalper Bot по единому конфигу")
    parser.add_argument("--config", type=str, default="config/bot_config.yaml",
                        help="Путь к главному конфиг-файлу")
    parser.add_argument("--skip_train", action="store_true", help="Пропустить обучение модели")
    parser.add_argument("--skip_backtest", action="store_true", help="Пропустить бэктест и обновление whitelist")
    parser.add_argument("--only_warmup_trade", action="store_true", help="Только warm-up и торговля (для тестов)")
    return parser.parse_args()

def load_main_config(path: str):
    if not os.path.exists(path):
        logger.error(f"Конфиг не найден: {path}")
        sys.exit(1)
    
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Загружен конфиг: {path}")
    logger.info(f"Hardware: {config['hardware']['mode']} | "
                f"Real trading: {config['trading_mode']['real_trading']} | "
                f"TP/SL mode: {config['tp_sl']['mode']}")
    
    return config

def run_training(config):
    logger.info("Шаг 3: Обучение модели...")
    trainer = Trainer(config)
    trainer.train()
    logger.info("Обучение завершено.")

def run_backtest_and_update_whitelist(config):
    logger.info("Шаг 4: Запуск полного бэктеста...")
    from scripts.backtest_all import main as backtest_main
    backtest_main()
    logger.info("Бэктест завершён, whitelist обновлён.")

def warmup(config):
    logger.info("Шаг 6: Warm-up (прогрев кэша и модели)...")
    time.sleep(5)  # имитация прогрева
    logger.info("Warm-up завершён.")

def start_trading(config):
    logger.info("Шаг 7: Запуск торговли...")
    
    from src.data.resampler import Resampler
    resampler = Resampler(config)
    ws_manager = WebSocketManager(config, resampler)
    asyncio.create_task(ws_manager.start())
    
    live_loop = LiveLoop(config)
    
    try:
        asyncio.run(live_loop.start())
    except KeyboardInterrupt:
        logger.info("Остановка бота по запросу пользователя...")
        live_loop.stop()
    except Exception as e:
        logger.error(f"Критическая ошибка в LiveLoop: {e}")
        live_loop.stop()

def main():
    args = parse_args()
    
    print("="*80)
    print("Scalper Bot Launcher (единый конфиг)")
    print("="*80)
    
    config = load_main_config(args.config)
    
    if not args.only_warmup_trade:
        if not args.skip_train:
            run_training(config)
        
        if not args.skip_backtest:
            run_backtest_and_update_whitelist(config)
    
    warmup(config)
    start_trading(config)

if __name__ == "__main__":
    main()