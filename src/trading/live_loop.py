"""
src/trading/live_loop.py

=== Основной принцип работы файла ===

Бесконечный цикл реальной/виртуальной торговли на Binance Futures.
Теперь использует реальный WebSocket + resampler вместо polling.

После аудита (Phase 4):
- УДАЛЁН global open_positions + Lock
- Полностью async + event-driven (websocket_manager)
- Все open/close — только через PositionManager (single source of truth)
- Убран ThreadPool + sleep(0.1)
"""

import asyncio
from datetime import datetime, timedelta
import logging
import signal
import sys

from src.core.config import load_config
from src.data.binance_client import BinanceClient
from src.data.storage import Storage
from src.features.feature_engine import FeatureEngine
from src.model.inference import InferenceEngine
from src.model.trainer import retrain
from src.model.scenario_tracker import ScenarioTracker
from src.trading.entry_manager import EntryManager
from src.trading.position_manager import PositionManager
from src.trading.tp_sl_manager import TP_SL_Manager
from src.trading.risk_manager import RiskManager
from src.backtest.pr_calculator import PRCalculator
from src.trading.websocket_manager import WebSocketManager
from src.data.resampler import Resampler
from src.utils.logger import setup_logger

logger = setup_logger('live_loop', logging.INFO)

STATE_FILE = "live_state.pkl"  # оставлено для совместимости, но больше не используется (state в PositionManager)

async def live_loop():
    config = load_config()
    client = BinanceClient()
    storage = Storage()
    inference = InferenceEngine(config)
    scenario_tracker = ScenarioTracker()
    position_manager = PositionManager()
    entry_manager = EntryManager(scenario_tracker)
    tp_sl_manager = TP_SL_Manager()
    risk_manager = RiskManager()
    pr_calculator = PRCalculator()

    resampler = Resampler(config)
    websocket_manager = WebSocketManager(config, storage, resampler)
    websocket_manager.start()

    logger.info("Warm-up: прогрев на 1000 свечах...")
    symbols = storage.get_whitelisted_symbols()[:3]
    for symbol in symbols:
        for tf in config['timeframes']:
            df = resampler.get_window(tf, 1000)
            if not df.empty:
                await FeatureEngine(config).build_features({tf: df})

    last_markets_update = datetime.utcnow() - timedelta(days=8)
    last_retrain = {tf: datetime.utcnow() - timedelta(days=8) for tf in config['timeframes']}

    signal.signal(signal.SIGINT, lambda s, f: asyncio.create_task(shutdown()))
    signal.signal(signal.SIGTERM, lambda s, f: asyncio.create_task(shutdown()))

    symbols = storage.get_whitelisted_symbols()
    logger.info(f"Запуск async live_loop (event-driven). Монеты: {len(symbols)}, TF: {config['timeframes']}")

    async def process_closed_candle(symbol: str, tf: str, candle_data: dict):
        """Event-driven обработка закрытой свечи (вызывается из WebSocketManager)"""
        window_df = resampler.get_window(tf, config.get('seq_len', 100))
        if window_df.empty:
            return

        # === FeatureEngine + anomalies (через manager) ===
        feature_engine = FeatureEngine(config)
        features_dict = await feature_engine.build_features({tf: window_df})
        anomalies = feature_engine.anomaly_detector.detect_anomalies(
            features_dict["features"], tf, symbol
        )

        # Остальная логика сигналов и открытия — через EntryManager + PositionManager
        await entry_manager.process_signals(
            symbol, tf, features_dict, anomalies, position_manager, risk_manager
        )

    # Регистрация callback в WebSocketManager (event-driven)
    websocket_manager.register_closed_candle_callback(process_closed_candle)

    while True:
        try:
            now = datetime.utcnow()

            # Ежедневное обновление whitelist
            if (now - last_markets_update) > timedelta(days=1):
                logger.info("Ежедневное обновление списка монет")
                client.update_markets_list()
                last_markets_update = now
                symbols = storage.get_whitelisted_symbols()

            # Еженедельный retrain
            for tf in config['timeframes']:
                if (now - last_retrain[tf]) > timedelta(days=config.get('retrain_interval_days', 7)):
                    logger.info(f"Еженедельное переобучение для {tf}")
                    await retrain(config, timeframe=tf)
                    last_retrain[tf] = now

            await asyncio.sleep(1)  # минимальный sleep для event-loop

        except Exception as e:
            logger.exception("Критическая ошибка в live_loop")
            await asyncio.sleep(30)


async def shutdown():
    """Graceful shutdown — позиции НЕ закрываем"""
    logger.info("Graceful shutdown: позиции остаются на бирже")
    logger.info("Бот остановлен.")
    sys.exit(0)


if __name__ == "__main__":
    asyncio.run(live_loop())