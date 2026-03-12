"""
src/trading/live_loop.py

=== Основной принцип работы файла ===

Бесконечный цикл реальной/виртуальной торговли на Binance Futures.
Теперь использует реальный WebSocket + resampler вместо polling.
"""

import time
import concurrent.futures
from datetime import datetime, timedelta
import logging
import signal
import sys
import asyncio
from collections import defaultdict
import pickle
import os

from src.core.config import load_config
from src.data.binance_client import BinanceClient
from src.data.storage import Storage
from src.features.feature_engine import FeatureEngine
from src.features.anomaly_detector import detect_anomalies
from src.model.inference import InferenceEngine
from src.model.trainer import retrain
from src.model.scenario_tracker import ScenarioTracker
from src.trading.entry_manager import EntryManager
from src.trading.order_executor import OrderExecutor
from src.trading.tp_sl_manager import TP_SL_Manager
from src.trading.risk_manager import RiskManager
from src.trading.virtual_trader import VirtualTrader
from src.backtest.pr_calculator import PRCalculator
from src.trading.websocket_manager import WebSocketManager
from src.data.resampler import Resampler
from src.utils.logger import setup_logger

logger = setup_logger('live_loop', logging.INFO)

open_positions = defaultdict(list)
last_retrain = {}
last_markets_update = None
STATE_FILE = "live_state.pkl"   # ← добавлено для state persistence

def signal_handler(sig, frame):
    logger.warning(f"Получен сигнал {signal.Signals(sig).name}. Запускаем graceful shutdown...")
    shutdown()
    sys.exit(0)

def save_state():
    """State persistence — сохраняем открытые позиции"""
    try:
        with open(STATE_FILE, "wb") as f:
            pickle.dump(dict(open_positions), f)
        logger.info(f"Состояние сохранено ({len(open_positions)} позиций)")
    except Exception as e:
        logger.error(f"Ошибка сохранения state: {e}")

def load_state():
    """Загрузка открытых позиций при старте"""
    global open_positions
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "rb") as f:
                open_positions = defaultdict(list, pickle.load(f))
            logger.info(f"Загружено {len(open_positions)} открытых позиций из state")
        except Exception as e:
            logger.error(f"Ошибка загрузки state: {e}")

def shutdown():
    """Graceful shutdown — ПОЗИЦИИ НЕ ЗАКРЫВАЕМ (ордера остаются на бирже)"""
    logger.info("Graceful shutdown: позиции НЕ закрываются — ордера на бирже остаются активны")
    save_state()
    logger.info("Бот остановлен.")

def live_loop():
    config = load_config()
    client = BinanceClient()
    storage = Storage()
    inference = InferenceEngine(config)
    scenario_tracker = ScenarioTracker()
    entry_manager = EntryManager(scenario_tracker)
    order_executor = OrderExecutor()
    tp_sl_manager = TP_SL_Manager()
    risk_manager = RiskManager()
    virtual_trader = VirtualTrader() if config.get('trading', {}).get('mode') == 'virtual' else None
    pr_calculator = PRCalculator()

    # === Подключение resampler и websocket_manager (Этап 1) ===
    resampler = Resampler(config)
    websocket_manager = WebSocketManager(config, storage, resampler)
    websocket_manager.start()

    # === Warm-up: 1000 свечей при старте (возвращено по ТЗ) ===
    logger.info("Warm-up: прогрев на 1000 свечах для понимания состояния рынка...")
    symbols = storage.get_whitelisted_symbols()[:3]
    for symbol in symbols:
        for tf in config['timeframes']:
            df = resampler.get_window(tf, 1000)
            if not df.empty:
                _ = FeatureEngine(config).build_features({tf: df})

    load_state()  # ← state persistence

    global last_markets_update
    last_markets_update = datetime.utcnow() - timedelta(days=8)

    timeframes = config['timeframes']
    for tf in timeframes:
        last_retrain[tf] = datetime.utcnow() - timedelta(days=8)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    symbols = storage.get_whitelisted_symbols()

    logger.info(f"Запуск live_loop на WebSocket + resampler. Монеты: {len(symbols)}, TF: {timeframes}")

    # Watchdog (heartbeat)
    last_heartbeat = datetime.utcnow()

    while True:
        try:
            if (datetime.utcnow() - last_markets_update) > timedelta(days=1):
                logger.info("Ежедневное обновление списка монет")
                client.update_markets_list()
                last_markets_update = datetime.utcnow()
                symbols = storage.get_whitelisted_symbols()

            now = datetime.utcnow()
            for tf in timeframes:
                if (now - last_retrain[tf]) > timedelta(days=config.get('retrain_interval_days', 7)):
                    logger.info(f"Еженедельное переобучение модели для {tf}")
                    asyncio.run(retrain(config, timeframe=tf))
                    last_retrain[tf] = now

            # Watchdog heartbeat — проверка каждые 5 минут
            if (now - last_heartbeat) > timedelta(minutes=5):
                logger.info("Heartbeat OK")
                last_heartbeat = now

            with concurrent.futures.ThreadPoolExecutor(max_workers=config['hardware']['max_workers']) as executor:
                futures = []
                for symbol in symbols:
                    for tf in timeframes:
                        futures.append(executor.submit(
                            process_candle,
                            symbol, tf, storage, inference, scenario_tracker, entry_manager,
                            tp_sl_manager, risk_manager, order_executor,
                            virtual_trader, pr_calculator, config, resampler
                        ))

                for future in concurrent.futures.as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        logger.error(f"Ошибка обработки свечи: {e}")

            time.sleep(0.1)

        except Exception as e:
            logger.exception("Критическая ошибка в live_loop")
            time.sleep(30)


def process_candle(symbol: str, timeframe: str, storage: Storage, inference: InferenceEngine, 
                   scenario_tracker: ScenarioTracker, entry_manager: EntryManager, 
                   tp_sl_manager: TP_SL_Manager, risk_manager: RiskManager, 
                   order_executor: OrderExecutor, virtual_trader, pr_calculator, config: dict, resampler: Resampler):
    window_df = resampler.get_window(timeframe, config.get('seq_len', 100))
    if window_df.empty:
        return

    last_candle = window_df.row(-1)
    candle_data = {'close': last_candle['close'], 'timestamp': last_candle['open_time']}
    candle_ts = candle_data.get('timestamp', int(time.time() * 1000))

    features_by_tf = {}
    feature_engine = FeatureEngine(config)
    for tf in config['timeframes']:
        df = resampler.get_window(tf, config.get('seq_len', 100))
        if not df.empty:
            features_by_tf[tf] = feature_engine.build_features({tf: df})

    if not features_by_tf:
        return

    anomalies = detect_anomalies(features_by_tf, timeframe)
    tf_anomalies = anomalies.get(timeframe, {})

    signals = []
    for window, anom in tf_anomalies.items():
        if not anom.get('confirmed'):
            continue

        anomaly_type = anom['type']
        features_input = features_by_tf[timeframe][window]
        prob_long, prob_short, uncertainty = inference.predict(features_input)

        wl = storage.get_whitelist_settings(symbol)
        direction = wl.get('direction', 'L')
        predict_prob = prob_long if direction == 'L' else prob_short

        if predict_prob < config['trading'].get('min_prob', 0.65):
            continue

        if not wl or wl.get('tf') != timeframe or wl.get('anomaly_type') != anomaly_type:
            if virtual_trader:
                tp_sl = tp_sl_manager.calculate_tp_sl(features_by_tf[timeframe][window])
                virtual_trader.open_position(symbol, anomaly_type, predict_prob, tp_sl)
            continue

        feats = features_by_tf[timeframe][window]
        weight = scenario_tracker.get_weight(scenario_tracker._binarize_features(feats))

        signals.append({
            'anom': anom,
            'window': window,
            'feats': feats,
            'prob': predict_prob,
            'weight': weight
        })

    if signals:
        signals.sort(key=lambda x: x['weight'], reverse=True)
        top_sig = signals[0]

        anomaly_type = top_sig['anom']['type']
        direction = wl.get('direction', 'L')

        tp_sl = tp_sl_manager.calculate_tp_sl(features_by_tf[timeframe][window], anomaly_type)
        if isinstance(tp_sl, dict):
            tp_price = tp_sl.get('tp') or tp_sl.get('tp_price', 0)
            sl_price = tp_sl.get('sl') or tp_sl.get('sl_price', 0)
        elif isinstance(tp_sl, (list, tuple)) and len(tp_sl) >= 2:
            tp_price = tp_sl[0]
            sl_price = tp_sl[1]
        else:
            tp_price = candle_data['close'] * (1.02 if direction == 'L' else 0.98)
            sl_price = tp_sl_manager.calculate_sl(candle_data, direction)

        size = risk_manager.calculate_position_size(
            symbol=symbol,
            entry_price=candle_data['close'],
            tp_price=tp_price,
            sl_price=sl_price
        )

        if size <= 0:
            return

        position = {
            'symbol': symbol,
            'anomaly_type': anomaly_type,
            'direction': direction,
            'entry_price': candle_data['close'],
            'size': size,
            'open_ts': candle_ts,
            'prob': top_sig['prob'],
            'consensus_count': top_sig['anom'].get('consensus_count', 1)
        }

        mode = entry_manager._resolve_mode(symbol, anomaly_type, direction)
        if mode == 'real':
            order_id = order_executor.place_order(position)
            if order_id:
                position['order_id'] = order_id
                logger.info(f"Открыта реальная позиция {anomaly_type} {direction} на {symbol}, size={size}")
        else:
            if virtual_trader:
                virtual_trader.open_position(symbol, anomaly_type, position['prob'], tp_sl)
                logger.debug(f"Открыта виртуальная позиция {anomaly_type} {direction} на {symbol}, size={size}")

        tp_sl_manager.add_open_position(position)
        open_positions[symbol].append(position)

        for sig in signals[1:]:
            if virtual_trader:
                virtual_trader.open_position(
                    symbol, sig['anom']['type'], sig['prob'],
                    tp_sl_manager.calculate_tp_sl(sig['feats'], sig['anom']['type'])
                )

    if symbol in open_positions:
        for pos in open_positions[symbol][:]:
            current_price = resampler.get_window(timeframe, 1).row(-1)['close']
            if tp_sl_manager.check_tp_sl(pos, current_price):
                closed_pos = order_executor.close_position(pos, virtual_trader)
                handle_closed_position(closed_pos, pr_calculator, risk_manager, config)
                open_positions[symbol].remove(pos)


def handle_closed_position(position: dict, pr_calculator: PRCalculator, risk_manager: RiskManager, config: dict):
    net_pl = position.get('net_pl', 0)
    risk_manager.update_deposit(net_pl)
    pr_calculator.update_pr(
        position['symbol'], position['anomaly_type'], position['direction'],
        position.get('hit_tp', False), net_pl
    )
    logger.info(f"Закрыта позиция {position['symbol']} {position['direction']} PL: {net_pl}")


if __name__ == "__main__":
    live_loop()