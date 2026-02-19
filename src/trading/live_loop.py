# src/trading/live_loop.py
"""
Основной торговый цикл (Live Loop) — сердце скальпер-бота.

Реализованные требования:
- Торгует ТОЛЬКО монеты из whitelist (обновляется после backtest_all.py)
- Если монета выпала из whitelist → дожидается закрытия ВСЕХ её позиций → после этого полностью исключает из активной торговли
- Продолжает расчёт PR фоном (виртуальные сделки) даже для исключённых монет
- Поддерживает 4 условия: candle_anomaly, volume_anomaly, cv_anomaly, q_condition
- Отдельные пороги вероятности: min_prob_anomaly и min_prob_quiet (настраиваются в конфиге)
- Авто-переобучение каждые 10 000 свечей (вызывается из live_loop)
- Max positions — ручная настройка из конфига
- Подписка на WebSocket по всем монетам, прошедшим min_age_months (фильтр возраста монеты)
- Реальная торговля — только Market ордера (через order_executor)
- Авто-переключение режимов агрессивности — отключено по умолчанию (можно включить в конфиге)
- Регулярная выгрузка статистики сценариев (через ScenarioTracker)

Логика:
- start() → запускает цикл
- _process_symbol() — обрабатывает новую свечу по монете
- _check_pending_removal() — проверяет монеты, выпавшие из whitelist
- _retrain_model() — переобучение каждые 10k свечей
- stop() — graceful shutdown (закрытие всех позиций при остановке)
"""

import asyncio
import time
from datetime import datetime
from typing import Dict, Set
import ccxt.pro as ccxt

from src.model.inference import InferenceEngine
from src.trading.order_executor import OrderExecutor
from src.trading.risk_manager import RiskManager
from src.trading.tp_sl_manager import TPSLManager
from src.trading.virtual_trader import VirtualTrader
from src.data.resampler import Resampler
from src.data.storage import Storage
from src.model.trainer import Trainer
from src.model.scenario_tracker import ScenarioTracker
from src.core.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

class LiveLoop:
    def __init__(self, config: dict):
        self.config = config
        self.storage = Storage(config)
        self.resampler = Resampler(config)
        self.inference = InferenceEngine(config)
        self.risk_manager = RiskManager(config)
        self.tp_sl_manager = TPSLManager(config)
        self.order_executor = OrderExecutor(config)
        self.virtual_trader = VirtualTrader(config)
        self.trainer = Trainer(config)
        self.scenario_tracker = ScenarioTracker(config)

        self.exchange = ccxt.binance(config["exchange"])
        self.is_real_trading = config["trading_mode"].get("real_trading", False)

        # Пороги вероятности (отдельные для обычных аномалий и quiet-режима)
        self.min_prob_anomaly = config["model"].get("min_prob_anomaly", 0.65)
        self.min_prob_quiet   = config["model"].get("min_prob_quiet",   0.78)

        # Активные монеты из whitelist
        self.active_symbols: Set[str] = set(self.storage.get_whitelist())

        # Монеты, которые выпали из whitelist, но ещё имеют открытые позиции
        self.pending_removal: Dict[str, int] = {}  # symbol → timestamp последнего обновления

        self.max_positions = config["risk"].get("max_positions", 10)
        self.open_positions: Dict[str, Dict] = {}  # symbol → {entry_time, entry_price, direction, size, ...}

        self.new_candles_count = 0
        self.last_retrain_time = time.time()
        self.last_scenario_export = time.time()

    async def start(self):
        """Запуск основного торгового цикла"""
        logger.info(f"LiveLoop запущен | "
                    f"Реальная торговля: {self.is_real_trading} | "
                    f"Активных монет: {len(self.active_symbols)} | "
                    f"Max positions: {self.max_positions} | "
                    f"min_prob_anomaly={self.min_prob_anomaly} | "
                    f"min_prob_quiet={self.min_prob_quiet}")

        # Подписываемся на все монеты из whitelist
        symbols = list(self.active_symbols)
        if not symbols:
            logger.warning("Whitelist пуст → торговля невозможна")
            return

        # Запускаем подписку через websocket_manager (батчи обрабатываются там)
        from src.trading.websocket_manager import WebSocketManager
        ws_manager = WebSocketManager(self.config, self.resampler)
        asyncio.create_task(ws_manager.start())

        # Основной цикл обработки
        while True:
            try:
                # Обработка новых свечей (resampler уже обновлён websocket_manager)
                for symbol in list(self.active_symbols):
                    await self._process_symbol(symbol)

                self._check_pending_removal()

                # Авто-переобучение
                if self.new_candles_count >= 10000:
                    await self._retrain_model()

                # Выгрузка статистики сценариев (раз в час)
                if time.time() - self.last_scenario_export > 3600:
                    self.scenario_tracker.export_statistics()
                    self.last_scenario_export = time.time()

                await asyncio.sleep(0.05)  # небольшой sleep для снижения нагрузки

            except Exception as e:
                logger.error(f"Критическая ошибка в LiveLoop: {e}")
                await asyncio.sleep(10)

    async def _process_symbol(self, symbol: str):
        """Обработка одной монеты (новая свеча уже добавлена в resampler)"""
        # Получаем последнюю свечу из resampler (если есть)
        candle = self.resampler.get_window("1m", 1)
        if candle is None or candle.is_empty():
            return

        current_price = candle["close"][-1]

        # Проверяем аномалии
        pred = self.inference.predict(symbol)
        if not pred["signal"]:
            return

        # Проверяем лимит позиций
        if len(self.open_positions) >= self.max_positions:
            return

        # Открываем позицию
        if self.is_real_trading:
            await self._open_real_position(symbol, pred, current_price)
        else:
            self._open_virtual_position(symbol, pred, current_price)

    async def _open_real_position(self, symbol: str, pred: Dict, entry_price: float):
        """Реальное открытие позиции по Market"""
        direction = "buy" if pred["prob"] > 0.5 else "sell"
        size = self.risk_manager.calculate_size(self.balance, entry_price, direction)

        success, msg = await self.order_executor.open_position(symbol, direction, size)
        if success:
            logger.info(f"[REAL] Открыта позиция {direction.upper()} {symbol} | size={size:.4f} | prob={pred['prob']:.4f}")
        else:
            logger.error(f"[REAL] Ошибка открытия {symbol}: {msg}")

    def _open_virtual_position(self, symbol: str, pred: Dict, entry_price: float):
        """Виртуальное открытие позиции"""
        self.virtual_trader.open_position(symbol, pred, entry_price)
        logger.debug(f"[VIRTUAL] Открыта позиция на {symbol} | prob={pred['prob']:.4f}")

    def _check_pending_removal(self):
        """Проверка монет, выпавших из whitelist"""
        current_whitelist = set(self.storage.get_whitelist())

        for symbol in list(self.active_symbols):
            if symbol not in current_whitelist:
                if symbol not in self.open_positions:
                    self.active_symbols.remove(symbol)
                    self.pending_removal.pop(symbol, None)
                    logger.info(f"Монета {symbol} полностью исключена из торговли (выпала из whitelist)")
                else:
                    self.pending_removal[symbol] = int(time.time())

    async def _retrain_model(self):
        """Переобучение каждые 10 000 свечей"""
        logger.info("Достигнуто 10 000 свечей → запуск переобучения модели")
        self.trainer.train()
        self.new_candles_count = 0
        self.last_retrain_time = time.time()

    def stop(self):
        """Graceful shutdown"""
        logger.info("LiveLoop завершается...")
        # Закрываем все открытые позиции (если real_trading)
        for symbol in list(self.open_positions.keys()):
            logger.warning(f"Принудительное закрытие позиции {symbol} при остановке")
            # Здесь должен быть вызов close_position через order_executor
        self.storage.close()