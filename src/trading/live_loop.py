# src/trading/live_loop.py
"""
Главный цикл работы бота в реальном времени (live-режим).

Основные задачи:
- Получение новых свечей (1m, 3m, 5m, 10m, 15m) через websocket/polling
- Расчёт признаков, аномалий и условий (C/V/CV/Q)
- Запуск inference модели (обычный + тихий режим по таймеру TF)
- Генерация сигналов → передача в order_executor
- Открытие виртуальных сделок ДЛЯ ВСЕХ монет (для PR)
- Открытие реальных сделок только для топ-монет (если совпадает настройка)
- Обновление trailing stop, soft entry, shadow trading
- Периодический пересчёт PR (особенно важно на телефоне — реже)
- Warm-up фаза — первые N свечей только виртуал + частое переобучение

Логика:
1. После каждой закрытой свечи — обновляем данные
2. Проверяем аномалии (C/V/CV) — если есть → inference
3. Проверяем таймер тихого режима по TF — если пора → inference (Q)
4. Если prob > порог → Signal → process_signal()
5. process_signal() → всегда виртуальная сделка (для PR)
6. Если монета топ и совпадает (TF + окно + тип + направление) → реальная
7. Shadow trading — параллельно симулирует реальные исходы

На телефоне:
- max_coins=5, max_tf=3, max_windows=3
- quiet_mode реже, PR пересчёт каждые 10 минут

На сервере:
- max_coins=150, parallel, GPU
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List

import polars as pl

from src.core.config import load_config
from src.core.types import Signal
from src.data.downloader import Downloader
from src.data.storage import Storage
from src.features.feature_engine import FeatureEngine
from src.model.inference import InferenceEngine
from src.backtest.virtual_trader import VirtualTrader
from src.trading.order_executor import OrderExecutor
from src.trading.risk_manager import RiskManager
from src.trading.entry_manager import EntryManager
from src.trading.tp_sl_manager import TpSlManager
from src.backtest.pr_calculator import PRCalculator
from src.trading.shadow_trading import ShadowTrading

logger = logging.getLogger(__name__)


class LiveLoop:
    """Главный цикл live-торговли"""

    def __init__(self, config: Dict):
        self.config = config
        self.downloader = Downloader(config)
        self.storage = Storage(config)
        self.feature_engine = FeatureEngine(config)
        self.inference_engine = InferenceEngine(config, model=None)  # модель загружается позже
        self.virtual_trader = VirtualTrader(config)
        self.order_executor = OrderExecutor(config)
        self.risk_manager = RiskManager(config)
        self.entry_manager = EntryManager(config)
        self.tp_sl_manager = TpSlManager(config)
        self.pr_calculator = PRCalculator(config)
        self.shadow_trading = ShadowTrading(config)

        self.running = False
        self.warm_up_completed = False
        self.warm_up_candles = 0
        self.warm_up_target = config["warm_up"]["duration_candles"]

        logger.info("LiveLoop initialized: mode=%s, hardware=%s", 
                    config["trading"]["mode"], config.get("hardware_profile", "auto"))

    async def start(self):
        """Запуск главного цикла"""
        self.running = True
        logger.info("LiveLoop started")

        # 1. Warm-up фаза
        await self._warm_up_phase()

        # 2. Запуск фоновых задач
        self.virtual_trader.start_background_tasks(asyncio.get_event_loop())

        # 3. Основной цикл
        while self.running:
            try:
                await self._process_new_data()
                await asyncio.sleep(1)  # Основной polling — 1 секунда
            except Exception as e:
                logger.error("Error in live loop: %s", e)
                await asyncio.sleep(10)  # Пауза при ошибке

        logger.info("LiveLoop stopped")

    async def _warm_up_phase(self):
        """Warm-up: первые N свечей — только виртуал + частое переобучение"""
        if not self.config["warm_up"]["enabled"]:
            self.warm_up_completed = True
            return

        logger.info("Starting warm-up phase (%d candles)", self.warm_up_target)

        while self.warm_up_candles < self.warm_up_target:
            await self._process_new_data(warm_up=True)
            self.warm_up_candles += 1
            if self.warm_up_candles % 100 == 0:
                logger.info("Warm-up progress: %d/%d candles", self.warm_up_candles, self.warm_up_target)
            await asyncio.sleep(0.5)

        self.warm_up_completed = True
        logger.info("Warm-up completed")

    async def _process_new_data(self, warm_up: bool = False):
        """Обработка новых свечей — главная логика цикла"""
        # 1. Получаем новые свечи для всех монет и TF
        all_data = await self._fetch_new_candles()

        # 2. Передаём в виртуальный трейдер (для PR всех монет)
        await self.virtual_trader.process_new_candles(all_data)

        # 3. Расчёт признаков и аномалий
        features_data = {}
        for symbol, tf_data in all_data.items():
            features = await self.feature_engine.build_features(tf_data)
            features_data[symbol] = features

        # 4. Inference и сигналы
        signals = await self.inference_engine.process_new_data(features_data)

        # 5. Обработка сигналов
        for signal in signals:
            await self._process_signal(signal, features_data[signal.symbol])

        # 6. Shadow trading — параллельно
        if self.config["shadow_trading"]:
            self.shadow_trading.process_signals(signals)

        # 7. Проверка trailing и soft entry
        self._update_positions(all_data)

    async def _fetch_new_candles(self) -> Dict[str, Dict[str, pl.DataFrame]]:
        """Получение новых свечей для всех монет и TF"""
        # В реальном коде — websocket или polling
        # Здесь упрощённо — имитация
        return {}  # Placeholder — в реальном коде используем downloader.fetch_new_candles

    async def _process_signal(self, signal: Signal, tf_data: Dict[str, pl.DataFrame]):
        """Обработка одного сигнала"""
        # 1. Всегда виртуальная сделка (для PR)
        await self.virtual_trader.simulate_virtual_trade(signal.symbol, signal, tf_data)

        # 2. Проверка — топ-монета ли?
        if self._is_top_coin(signal.symbol) and self._signal_matches_config(signal):
            # Реальная сделка (если режим real)
            if self.config["trading"]["mode"] == "real":
                # Расчёт размера, TP/SL
                entry_price = tf_data[signal.timeframe]["close"][-1]
                size = self.risk_manager.calculate_position_size(
                    entry_price, signal, tf_data[signal.timeframe], self._get_current_balance()
                )
                if size <= 0:
                    return

                tp = self.tp_sl_manager.calculate_tp(tf_data[signal.timeframe], entry_price, signal.direction)
                sl = self.tp_sl_manager.calculate_sl(tf_data[signal.timeframe], entry_price, signal.direction)

                # Открытие позиции
                await self.order_executor.place_order(
                    symbol=signal.symbol,
                    side="buy" if signal.direction in ["L", "LS"] else "sell",
                    amount=size,
                    price=entry_price,
                    order_type="market"
                )

                logger.info("REAL trade opened: %s %s @ %.2f, size=%.4f, TP=%.2f, SL=%.2f",
                            signal.symbol, signal.direction, entry_price, size, tp, sl)

            # Мягкие входы
            if self.entry_manager.enabled:
                self.entry_manager.open_soft_part(signal, signal.probability, entry_price, size)

    def _is_top_coin(self, symbol: str) -> bool:
        """Проверка — входит ли монета в топ по PR"""
        top_coins = self.pr_calculator.get_top_coins()
        return symbol in [c[0] for c in top_coins]

    def _signal_matches_config(self, signal: Signal) -> bool:
        """Совпадает ли сигнал с настройками топ-монеты"""
        # В реальном коде — проверка TF, window, anomaly_type, direction
        return True  # Placeholder

    def _update_positions(self, all_data: Dict):
        """Обновление trailing и soft entry на новых свечах"""
        # Placeholder — в реальном коде обновление Position.trailing_price
        pass

    def _get_current_balance(self) -> float:
        """Текущий баланс (real / virtual)"""
        # Placeholder — в реальном коде fetch_balance
        return 10000.0

    def stop(self):
        self.running = False
        logger.info("LiveLoop stopping...")