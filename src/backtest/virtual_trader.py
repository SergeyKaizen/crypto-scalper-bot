# src/backtest/virtual_trader.py
"""
Фоновый процесс виртуальной торговли для обновления PR.

Ключевые принципы:
- Работает НЕЗАВИСИМО от режима real/virtual
- Симулирует сделки ДЛЯ ВСЕХ монет в модуле монет (не только топ)
- Каждая симулированная сделка идёт в статистику PR
- Не влияет на реальный/виртуальный депозит основного режима
- Запускается параллельно live_loop (в фоне)
- Обновляет PR после каждой закрытой виртуальной сделки
- На телефоне — пересчёт реже (каждые 10 минут), чтобы не убивать батарею

Логика:
1. После каждой новой свечи — проверяем сигналы (аномалии или тихий режим)
2. Если сигнал → симулируем виртуальную позицию (TP/SL/trailing/soft entry)
3. После закрытия позиции — TradeResult → add_trade в pr_calculator
4. PR пересчитывается → обновляется топ-монеты

На телефоне:
- max_coins = 5
- тихий режим реже
- симуляция только на 1m + 5m (или по конфигу)

На сервере:
- max_coins = 150
- параллельная симуляция (joblib)
- полный мульти-TF

Зависимости:
- src/backtest/engine.py — симуляция сделок
- src/backtest/pr_calculator.py — обновление PR
- src/trading/tp_sl_manager.py, risk_manager.py — расчёт параметров позиции
"""

import asyncio
import logging
from typing import Dict, List
from datetime import datetime

from src.core.config import load_config
from src.core.types import Signal, Position, TradeResult
from src.backtest.engine import BacktestEngine
from src.backtest.pr_calculator import PRCalculator
from src.features.feature_engine import FeatureEngine
from src.model.inference import InferenceEngine

logger = logging.getLogger(__name__)


class VirtualTrader:
    """Фоновый виртуальный трейдер для обновления PR всех монет"""

    def __init__(self, config: Dict):
        self.config = config
        self.engine = BacktestEngine(config)
        self.pr_calculator = PRCalculator(config)
        self.feature_engine = FeatureEngine(config)
        self.inference_engine = InferenceEngine(config, model=None)  # модель передаётся позже

        # Последнее время обновления PR по монетам (для телефона — реже)
        self.last_pr_update = {}  # symbol → timestamp

        logger.info("VirtualTrader initialized (all coins PR update)")

    async def process_new_candles(self, all_data: Dict[str, Dict[str, pl.DataFrame]]):
        """
        Основная точка входа — вызывается из live_loop после получения новых свечей

        all_data: Dict[symbol: Dict[tf: DataFrame]]
        """
        for symbol, tf_data in all_data.items():
            if symbol not in self.config["current_coins"]:  # Только монеты в модуле
                continue

            # 1. Проверяем сигналы для этой монеты
            signals = await self.inference_engine.process_new_data(tf_data)

            for signal in signals:
                # Симулируем виртуальную сделку
                await self.simulate_virtual_trade(symbol, signal, tf_data)

    async def simulate_virtual_trade(self, symbol: str, signal: Signal, tf_data: Dict[str, pl.DataFrame]):
        """Симуляция виртуальной сделки (не влияет на основной депозит)"""
        # Берём основной TF из сигнала
        main_tf = signal.timeframe
        df = tf_data.get(main_tf)
        if df is None or len(df) < 100:
            return

        # Находим индекс последней свечи
        entry_idx = len(df) - 1

        # Симуляция
        trade_result = self.engine.simulate_trade(df, entry_idx, {
            "direction": signal.direction,
            "probability": signal.probability
        })

        # Добавляем в PR
        self.pr_calculator.add_trade(symbol, trade_result)
        logger.debug("Virtual trade simulated for %s: pnl=%.2f%%, reason=%s", 
                     symbol, trade_result.pnl_pct, trade_result.reason)

    async def periodic_pr_recalc(self):
        """Фоновая периодическая проверка и пересчёт PR (особенно важно на телефоне)"""
        while True:
            await asyncio.sleep(600)  # Каждые 10 минут

            for symbol in self.config["current_coins"]:
                # На телефоне — реже
                if self.config.get("low_power_mode", False):
                    last = self.last_pr_update.get(symbol, 0)
                    if time.time() - last < 600:
                        continue

                self.pr_calculator.update_pr_for_symbol(symbol)
                self.last_pr_update[symbol] = time.time()

            logger.info("Periodic PR recalc completed")

    def start_background_tasks(self, loop: asyncio.AbstractEventLoop):
        """Запуск фоновых задач"""
        loop.create_task(self.periodic_pr_recalc())
        logger.info("VirtualTrader background tasks started")

# Пример запуска (в live_loop.py)
if __name__ == "__main__":
    config = load_config()
    trader = VirtualTrader(config)

    # Имитация
    dummy_data = {
        "BTCUSDT": {
            "1m": pl.DataFrame(...)  # Пример
        }
    }
    asyncio.run(trader.process_new_candles(dummy_data))