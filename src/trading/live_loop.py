"""
src/trading/live_loop.py

=== Основной принцип работы файла ===

Бесконечный цикл реальной/виртуальной торговли на Binance Futures.

Ключевые задачи (по ТЗ + последние уточнения):
- Докачка новых свечей по всем TF
- Сбор признаков по всем TF и окнам
- Детекция аномалий + мульти-TF consensus
- Сбор всех потенциальных сигналов в свече
- Выбор top-1 по весу из scenario_tracker (самый большой вес)
- Открытие только 1 позиции (глобальный lock: нет открытой на монете)
- Остальные сигналы — виртуальные для PR
- Запрет новой позиции в свече после закрытия предыдущей
- Retrain модели каждую неделю отдельно по каждому TF
- Ежедневное обновление списка монет (auto delisted remove)
- Graceful shutdown на сигналы

=== Главные функции ===
- live_loop() — основной цикл
- process_candle(symbol, timeframe) — сбор сигналов → выбор top-1 → открытие
- handle_closed_position(position) — PR и депозит update
- shutdown() — закрытие позиций

=== Примечания ===
- Только 1 позиция в live: top-1 по весу + глобальный lock
- Если несколько сигналов — остальные виртуальные
- retrain: last_retrain[tf] → проверка timedelta(days=7)
- quiet_streak: per-symbol/per-TF в quiet_streaks dict
- consensus: только для младших TF требуют подтверждения старших
- polling fallback (WS можно добавить позже)
"""

import time
import concurrent.futures
from datetime import datetime, timedelta
import logging
import signal
import sys
from collections import defaultdict

from src.core.config import load_config
from src.data.binance_client import BinanceClient
from src.data.downloader import download_new_candles
from src.data.storage import Storage
from src.features.feature_engine import compute_features
from src.features.anomaly_detector import detect_anomalies
from src.model.inference import Inference
from src.model.trainer import Trainer
from src.model.scenario_tracker import ScenarioTracker
from src.trading.entry_manager import EntryManager
from src.trading.order_executor import OrderExecutor
from src.trading.tp_sl_manager import TP_SL_Manager
from src.trading.risk_manager import RiskManager
from src.trading.virtual_trader import VirtualTrader
from src.backtest.pr_calculator import PRCalculator
from src.utils.logger import setup_logger

logger = setup_logger('live_loop', logging.INFO)

# State
last_markets_update = None
open_positions = defaultdict(list)  # symbol → list[positions]
last_retrain = {}  # tf → datetime последнего retrain
quiet_streaks = defaultdict(lambda: defaultdict(int))  # symbol → tf → streak

def signal_handler(sig, frame):
    logger.info("Сигнал завершения. Закрываем позиции...")
    shutdown()
    sys.exit(0)

def shutdown():
    for symbol, positions in open_positions.items():
        for pos in positions:
            OrderExecutor.close_position(pos)
    logger.info("Бот остановлен.")

def live_loop():
    config = load_config()
    client = BinanceClient()
    storage = Storage()
    inference = Inference()
    trainer = Trainer()
    scenario_tracker = ScenarioTracker()
    entry_manager = EntryManager(scenario_tracker)  # передаём tracker для весов
    order_executor = OrderExecutor()
    tp_sl_manager = TP_SL_Manager()
    risk_manager = RiskManager()
    virtual_trader = VirtualTrader() if config['trading']['mode'] == 'virtual' else None
    pr_calculator = PRCalculator()

    global last_markets_update
    last_markets_update = datetime.utcnow() - timedelta(days=8)

    timeframes = config['timeframes']
    for tf in timeframes:
        last_retrain[tf] = datetime.utcnow() - timedelta(days=8)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    symbols = storage.get_whitelisted_symbols()

    logger.info(f"Запуск live_loop. Монеты: {len(symbols)}, TF: {timeframes}")

    while True:
        try:
            # 1. Ежедневное обновление списка монет
            if (datetime.utcnow() - last_markets_update) > timedelta(days=1):
                logger.info("Ежедневное обновление списка монет")
                client.update_markets_list()
                last_markets_update = datetime.utcnow()
                symbols = storage.get_whitelisted_symbols()

            # 2. Докачка новых свечей
            for symbol in symbols:
                for tf in timeframes:
                    download_new_candles(symbol, tf)
                    time.sleep(0.1)

            # 3. Еженедельный retrain per-TF
            now = datetime.utcnow()
            for tf in timeframes:
                if (now - last_retrain[tf]) > timedelta(days=config.get('retrain_interval_days', 7)):
                    logger.info(f"Еженедельное переобучение модели для {tf}")
                    trainer.retrain(timeframe=tf)
                    last_retrain[tf] = now

            # 4. Обработка новых свечей
            with concurrent.futures.ThreadPoolExecutor(max_workers=config['hardware']['max_workers']) as executor:
                futures = []
                for symbol in symbols:
                    for tf in timeframes:
                        futures.append(executor.submit(
                            process_candle,
                            symbol, tf, storage, inference, scenario_tracker, entry_manager,
                            tp_sl_manager, risk_manager, order_executor,
                            virtual_trader, pr_calculator, config
                        ))

                for future in concurrent.futures.as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        logger.error(f"Ошибка обработки свечи: {e}")

            time.sleep(60)  # основной цикл ~1m

        except Exception as e:
            logger.exception("Критическая ошибка в live_loop")
            time.sleep(30)


def process_candle(symbol: str, timeframe: str, storage: Storage, inference: Inference, scenario_tracker: ScenarioTracker,
                   entry_manager: EntryManager, tp_sl_manager: TP_SL_Manager,
                   risk_manager: RiskManager, order_executor: OrderExecutor,
                   virtual_trader: VirtualTrader, pr_calculator: PRCalculator, config: dict):
    """
    Обработка новой свечи:
    - Сбор признаков по всем TF
    - Детекция аномалий + consensus
    - Сбор всех potential сигналов в свече
    - Выбор top-1 по весу (самый большой вес)
    - Открытие только этой позиции (если проходит проверки)
    - Остальные — виртуальные для PR
    """
    features_by_tf = {}
    for tf in config['timeframes']:
        df = storage.get_candles(symbol, tf, limit=config['seq_len'])
        if not df.empty:
            features_by_tf[tf] = compute_features(df)

    if not features_by_tf:
        return

    anomalies = detect_anomalies(features_by_tf, timeframe)

    tf_anomalies = anomalies.get(timeframe, {})

    signals = []
    for window, anom in tf_anomalies.items():
        if not anom['confirmed']:
            continue

        anomaly_type = anom['type']
        if anomaly_type == 'Q':
            continue

        quiet_streak = quiet_streaks[symbol][timeframe]
        if anomaly_type != 'Q':
            quiet_streaks[symbol][timeframe] = 0
        else:
            quiet_streaks[symbol][timeframe] += 1
            quiet_streak = quiet_streaks[symbol][timeframe]

        predict_prob = inference.predict(
            features_by_tf[timeframe][window],
            anomaly_type,
            extra_features={'quiet_streak': quiet_streak}
        )

        if predict_prob < config['trading']['min_prob']:
            continue

        wl = storage.get_whitelist_settings(symbol)
        if not wl or wl['tf'] != timeframe or wl['anomaly_type'] != anomaly_type:
            if virtual_trader:
                tp_sl = tp_sl_manager.calculate_tp_sl(features_by_tf[timeframe][window])
                virtual_trader.open_virtual_position(symbol, anomaly_type, predict_prob, tp_sl)
            continue

        feats = features_by_tf[timeframe][window]
        weight = scenario_tracker.get_weight(scenario_tracker._binarize_features(feats))

        signals.append({
            'anom': anom,
            'window': window,
            'feats': feats,
            'prob': predict_prob,
            'quiet_streak': quiet_streak,
            'weight': weight
        })

    if signals:
        # Выбор top-1 по весу
        signals.sort(key=lambda x: x['weight'], reverse=True)
        top_sig = signals[0]

        anomaly_type = top_sig['anom']['type']
        direction = wl['direction']  # L/S/LS → resolve

        sl_price = tp_sl_manager.calculate_sl(candle_data, direction)
        size = risk_manager.calculate_size(
            symbol=symbol,
            entry_price=candle_data['close'],
            sl_price=sl_price,
            risk_pct=config['trading']['risk_pct']
        )

        if size <= 0:
            logger.warning(f"Некорректный размер позиции для {symbol}")
            return

        position = {
            'symbol': symbol,
            'anomaly_type': anomaly_type,
            'direction': direction,
            'entry_price': candle_data['close'],
            'size': size,
            'open_ts': candle_ts,
            'prob': top_sig['prob'],
            'quiet_streak': top_sig['quiet_streak'],
            'consensus_count': top_sig['anom'].get('consensus_count', 1)
        }

        mode = entry_manager._resolve_mode(symbol, anomaly_type, direction)
        if mode == TradeMode.REAL:
            order_id = order_executor.place_order(position)
            if order_id:
                position['order_id'] = order_id
                logger.info(f"Открыта реальная позиция {anomaly_type} {direction} на {symbol}, size={size}, weight={top_sig['weight']:.4f}")
            else:
                logger.error(f"Ошибка открытия реальной позиции {symbol}")
                return
        else:
            virtual_trader.open_position(position)
            logger.debug(f"Открыта виртуальная позиция {anomaly_type} {direction} на {symbol}, size={size}, weight={top_sig['weight']:.4f}")

        tp_sl_manager.add_open_position(position)

        # Остальные сигналы — виртуальные
        for sig in signals[1:]:
            virtual_trader.open_virtual_position(
                symbol, sig['anom']['type'], sig['prob'],
                tp_sl_manager.calculate_tp_sl(sig['feats'], sig['anom']['type'])
            )

    # Мониторинг открытых позиций
    if symbol in open_positions:
        for pos in open_positions[symbol][:]:
            current_price = storage.get_last_candle(symbol, timeframe)['close']
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