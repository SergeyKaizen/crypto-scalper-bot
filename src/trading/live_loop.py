"""
src/trading/live_loop.py

=== Основной принцип работы файла ===

Бесконечный цикл реальной/виртуальной торговли на Binance Futures.

Ключевые задачи (по ТЗ + утверждённые изменения):
- Докачка новых свечей по всем TF (downloader)
- Сбор признаков по всем TF и окнам (feature_engine)
- Детекция аномалий по всем TF + мульти-TF consensus (anomaly_detector)
- Предикт только при confirmed аномалии (consensus >= min_tf_consensus)
- Открытие позиции если предикт > min_prob и монета в whitelist
- Мониторинг TP/SL, update депозита и PR после закрытия
- Retrain модели **каждую неделю отдельно по каждому TF**
- Ежедневное обновление списка монет (update_markets_list → auto delisted remove)
- Graceful shutdown на сигналы

=== Главные функции ===

- live_loop() — основной цикл
- process_candle(symbol, timeframe) — обработка новой свечи
- handle_closed_position(position) — PR и депозит update
- shutdown() — закрытие позиций при остановке

=== Примечания ===
- retrain: last_retrain[tf] → проверка timedelta(days=7)
- quiet_streak: per-symbol/per-TF в quiet_streaks dict
- consensus: только для младших TF (1m,3m,5m) требуют подтверждения старших
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
    entry_manager = EntryManager()
    order_executor = OrderExecutor()
    tp_sl_manager = TP_SL_Manager()
    risk_manager = RiskManager()
    virtual_trader = VirtualTrader() if config['trading']['mode'] == 'virtual' else None
    pr_calculator = PRCalculator()

    global last_markets_update
    last_markets_update = datetime.utcnow() - timedelta(days=8)  # force update on start

    # Инициализация last_retrain для каждого TF
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

            # 2. Докачка новых свечей по всем монетам и TF
            for symbol in symbols:
                for tf in timeframes:
                    download_new_candles(symbol, tf)
                    time.sleep(0.1)  # rate-limit

            # 3. Еженедельный retrain per-TF
            now = datetime.utcnow()
            for tf in timeframes:
                if (now - last_retrain[tf]) > timedelta(days=config.get('retrain_interval_days', 7)):
                    logger.info(f"Еженедельное переобучение модели для {tf}")
                    trainer.retrain(timeframe=tf)
                    last_retrain[tf] = now

            # 4. Обработка новых свечей (polling; WS можно добавить позже)
            with concurrent.futures.ThreadPoolExecutor(max_workers=config['hardware']['max_workers']) as executor:
                futures = []
                for symbol in symbols:
                    for tf in timeframes:
                        futures.append(executor.submit(
                            process_candle,
                            symbol, tf, storage, inference, entry_manager,
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


def process_candle(symbol: str, timeframe: str, storage: Storage, inference: Inference,
                   entry_manager: EntryManager, tp_sl_manager: TP_SL_Manager,
                   risk_manager: RiskManager, order_executor: OrderExecutor,
                   virtual_trader: VirtualTrader, pr_calculator: PRCalculator, config: dict):
    """
    Обработка новой свечи для монеты и TF
    """
    # Получаем данные по всем TF для мульти-TF consensus
    features_by_tf = {}
    for tf in config['timeframes']:
        df = storage.get_candles(symbol, tf, limit=config['seq_len'])
        if not df.empty:
            features_by_tf[tf] = compute_features(df)

    if not features_by_tf:
        return

    # Детекция аномалий + consensus
    anomalies = detect_anomalies(features_by_tf, timeframe)

    tf_anomalies = anomalies.get(timeframe, {})
    for window, anom in tf_anomalies.items():
        if not anom['confirmed']:
            continue

        anomaly_type = anom['type']
        if anomaly_type == 'Q':
            continue

        # Quiet streak per-symbol/per-TF
        quiet_streak = quiet_streaks[symbol][timeframe]
        if anomaly_type != 'Q':
            quiet_streaks[symbol][timeframe] = 0
        else:
            quiet_streaks[symbol][timeframe] += 1
            quiet_streak = quiet_streaks[symbol][timeframe]

        # Предикт модели (с extra quiet_streak)
        predict_prob = inference.predict(
            features_by_tf[timeframe][window],
            anomaly_type,
            extra_features={'quiet_streak': quiet_streak}
        )

        if predict_prob < config['trading']['min_prob']:
            continue

        # Whitelist check
        wl = storage.get_whitelist_settings(symbol)
        if not wl or wl['tf'] != timeframe or wl['anomaly_type'] != anomaly_type:
            if virtual_trader:
                tp_sl = tp_sl_manager.calculate_tp_sl(features_by_tf[timeframe][window])
                virtual_trader.open_virtual_position(symbol, anomaly_type, predict_prob, tp_sl)
            continue

        # Размер позиции, TP/SL
        direction = wl['direction']  # L/S/LS → resolve
        risk = risk_manager.calculate_risk(config['trading']['risk_pct'], config['deposit'])
        sl_distance = tp_sl_manager.sl_distance(features_by_tf[timeframe][window])
        position_size = risk_manager.calculate_position_size(risk, sl_distance)

        min_size = config.get('min_order_size', {}).get(symbol, 0.001)
        if position_size < min_size:
            logger.warning(f"Маленький размер позиции для {symbol}")
            continue

        tp, sl = tp_sl_manager.calculate_tp_sl(features_by_tf[timeframe][window], timeframe)
        position = entry_manager.open_position(
            symbol, direction, position_size, tp, sl,
            order_executor, virtual_trader,
            extra={'quiet_streak': quiet_streak}
        )

        if position:
            open_positions[symbol].append(position)

    # Мониторинг открытых позиций
    if symbol in open_positions:
        for pos in open_positions[symbol][:]:
            current_price = storage.get_last_candle(symbol, timeframe)['close']
            if tp_sl_manager.check_tp_sl(pos, current_price):
                closed_pos = order_executor.close_position(pos, virtual_trader)
                handle_closed_position(closed_pos, pr_calculator, risk_manager, config)
                open_positions[symbol].remove(pos)


def handle_closed_position(position: dict, pr_calculator: PRCalculator, risk_manager: RiskManager, config: dict):
    """Обработка закрытой позиции"""
    net_pl = position.get('net_pl', 0)
    risk_manager.update_deposit(net_pl)
    pr_calculator.update_pr(
        position['symbol'], position['anomaly_type'], position['direction'],
        position.get('hit_tp', False), net_pl
    )
    logger.info(f"Закрыта позиция {position['symbol']} {position['direction']} PL: {net_pl}")


if __name__ == "__main__":
    live_loop()