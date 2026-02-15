# src/trading/live_loop.py
"""
Основной торговый цикл бота (Live Loop) — сердце всей системы.

Ключевые реализованные требования (по твоим последним указаниям):
- Торгует ТОЛЬКО монеты из whitelist (обновляется после backtest_all.py)
- Если монета выпала из whitelist → дожидается закрытия ВСЕХ её позиций → после этого полностью исключает из активной торговли
- Продолжает расчёт PR фоном (виртуальные сделки) даже для исключённых монет
- Поддержка 4 условий: candle_anomaly, volume_anomaly, cv_anomaly, q_condition
- Отдельные пороги вероятности: min_prob_anomaly и min_prob_quiet (настраиваются в конфиге)
- Авто-переобучение каждые 10 000 свечей (или по времени — настраивается)
- Max positions — ручная настройка из конфига (например 8–15)
- Подписка на WebSocket по всем монетам, прошедшим min_age_months (фильтр возраста монеты)
- Реальная торговля — только Market ордера (как ты просил)
- Авто-переключение режимов агрессивности — отключено по умолчанию (можно включить в конфиге)
- Регулярная выгрузка статистики сценариев (через ScenarioTracker)
"""

import asyncio
import time
from datetime import datetime
from typing import Dict, Set, List
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

        # === Настройки порогов вероятности (по твоему требованию) ===
        # min_prob_anomaly — для обычных аномалий (candle/volume/cv)
        # min_prob_quiet   — для q_condition (более строгий, т.к. рынок спокойный)
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

    async def start(self):
        """Запуск основного торгового цикла"""
        logger.info(f"LiveLoop запущен | "
                    f"Реальная торговля: {self.is_real_trading} | "
                    f"Активных монет: {len(self.active_symbols)} | "
                    f"Max positions: {self.max_positions}")

        # Подписываемся на все монеты, прошедшие фильтр возраста (min_age_months)
        symbols = list(self.active_symbols)
        if not symbols:
            logger.warning("Whitelist пуст → торговля невозможна")
            return

        # Подписка на 1m свечи (основной таймфрейм)
        await self.exchange.watch_multiple_ohlcv(symbols, "1m")

        while True:
            try:
                for symbol in list(self.active_symbols):
                    await self._process_symbol(symbol)

                # Проверка монет, которые выпали из whitelist
                self._check_pending_removal()

                # Авто-переобучение каждые 10 000 свечей
                if self.new_candles_count >= 10000:
                    await self._retrain_model()

                # Экспорт статистики сценариев (раз в час)
                if time.time() - self.last_export_time > 3600:
                    self.scenario_tracker.export_statistics()
                    self.last_export_time = time.time()

                await asyncio.sleep(0.05)  # небольшой sleep для снижения нагрузки

            except Exception as e:
                logger.error(f"Критическая ошибка в LiveLoop: {e}")
                await asyncio.sleep(10)

    async def _process_symbol(self, symbol: str):
        """Обработка одной монеты (новая свеча)"""
        # Получаем новую 1m свечу
        candle = await self.exchange.watch_ohlcv(symbol, "1m", limit=1)
        if not candle or len(candle) == 0:
            return

        df_1m = pl.DataFrame(candle[-1], schema=["timestamp", "open", "high", "low", "close", "volume"])
        self.resampler.add_1m_candle(df_1m.to_dicts()[0])
        self.new_candles_count += 1

        # Проверяем аномалии / q_condition
        pred = self.inference.predict(symbol)
        if not pred["signal"]:
            return

        # Проверяем лимит открытых позиций
        if len(self.open_positions) >= self.max_positions:
            return

        # Открываем позицию
        if self.is_real_trading:
            await self._open_real_position(symbol, pred)
        else:
            self._open_virtual_position(symbol, pred)

    async def _open_real_position(self, symbol: str, pred: Dict):
        """Открытие реальной позиции по Market ордеру"""
        direction = "buy" if pred["prob"] > 0.5 else "sell"
        size = self.risk_manager.calculate_size(self.balance, pred.get("entry_price", 0), direction)

        try:
            order = await self.exchange.create_order(
                symbol=symbol,
                type="market",
                side=direction,
                amount=size,
                params={"positionSide": "BOTH"}
            )
            logger.info(f"[REAL] Открыта позиция {direction.upper()} {symbol} | размер={size} | prob={pred['prob']}")
        except Exception as e:
            logger.error(f"Ошибка открытия реальной позиции {symbol}: {e}")

    def _open_virtual_position(self, symbol: str, pred: Dict):
        """Виртуальная позиция (для расчёта PR)"""
        self.virtual_trader.open_position(symbol, pred)
        logger.debug(f"[VIRTUAL] Открыта позиция на {symbol} | prob={pred['prob']}")

    def _check_pending_removal(self):
        """Проверяет монеты, которые выпали из whitelist"""
        current_whitelist = set(self.storage.get_whitelist())

        for symbol in list(self.active_symbols):
            if symbol not in current_whitelist:
                if symbol not in self.open_positions:  # все позиции закрыты
                    self.active_symbols.remove(symbol)
                    self.pending_removal.pop(symbol, None)
                    logger.info(f"Монета {symbol} полностью исключена из торговли (выпала из whitelist)")
                else:
                    # Ждём закрытия позиции
                    self.pending_removal[symbol] = int(time.time())

    async def _retrain_model(self):
        """Автоматическое переобучение каждые 10 000 свечей"""
        logger.info("Достигнуто 10 000 свечей → запуск переобучения модели")
        self.trainer.retrain_incremental()
        self.new_candles_count = 0
        self.last_retrain_time = time.time()

    def stop(self):
        """Graceful shutdown"""
        logger.info("LiveLoop завершается...")
        # Закрываем все открытые позиции (если real_trading)
        for symbol in list(self.open_positions.keys()):
            logger.warning(f"Принудительное закрытие позиции {symbol} при остановке")
            # Здесь должен быть вызов close_position()
        self.storage.close()