# src/trading/live_loop.py
"""
Главный реал-тайм цикл бота.
Теперь использует Resampler для получения всех таймфреймов из 1m.
Обновлено: 14 февраля 2026
"""

import asyncio
import time
from typing import Dict

import polars as pl

from ..utils.logger import logger
from ..core.config import get_config
from ..core.types import AnomalySignal, ModelInput, TradeConfig
from ..features.feature_engine import FeatureEngine
from ..model.inference import InferenceEngine
from .order_executor import OrderExecutor
from .virtual_trader import VirtualTrader
from .risk_manager import RiskManager
from .websocket_manager import WebSocketManager
from ..data.resampler import Resampler


class LiveLoop:
    def __init__(self):
        self.config = get_config()
        self.feature_engine = FeatureEngine(self.config)
        self.inference = InferenceEngine(self.config)
        self.order_executor = OrderExecutor(self.config)
        self.virtual_trader = VirtualTrader()
        self.risk_manager = RiskManager(self.config)
        self.ws_manager = WebSocketManager(self)
        self.resampler = Resampler()  # ← используем ресэмплинг

        self.current_dfs: Dict[str, Dict[str, pl.DataFrame]] = {}  # coin → {tf: df}
        self.traded_coins: Dict[str, TradeConfig] = {}

        logger.info("LiveLoop запущен", hardware=self.config["hardware"]["type"])

    async def run(self):
        logger.info("Запуск основного цикла")
        await self.ws_manager.start()

        asyncio.create_task(self._console_command_handler())

        try:
            while True:
                await asyncio.sleep(0.25)
                await self._process_all_coins()
        except asyncio.CancelledError:
            logger.info("Цикл остановлен")
        except Exception as e:
            logger.exception("Критическая ошибка", error=str(e))
        finally:
            await self.ws_manager.stop()
            logger.info("LiveLoop завершён")

    async def _process_all_coins(self):
        for coin, tfs in list(self.current_dfs.items()):
            await self._process_coin(coin, tfs)

    async def _process_coin(self, coin: str, tfs: Dict[str, pl.DataFrame]):
        """Обработка одной монеты."""
        try:
            # 1. Ресэмплим все TF из 1m (если 1m есть)
            df_1m = tfs.get("1m")
            if df_1m is None or df_1m.is_empty():
                logger.debug("Нет 1m данных", coin=coin)
                return

            all_tfs = self.resampler.get_all_timeframes(coin, df_1m)

            # 2. Feature engineering
            model_input: ModelInput = self.feature_engine.process(all_tfs)

            # 3. Если нет аномалий — только симуляция закрытия
            if not model_input.anomalies:
                self._simulate_existing_positions(coin, all_tfs)
                return

            logger.info("Аномалии обнаружены", coin=coin, count=len(model_input.anomalies))

            # 4. Отдельный предикт на каждую аномалию
            for signal in model_input.anomalies:
                signal.coin = coin

                single_input = self.feature_engine.process_for_single_anomaly(all_tfs, signal.anomaly_type)
                pred = self.inference.predict(single_input)

                if coin not in self.traded_coins:
                    self.virtual_trader.process_signal(signal, pred)
                else:
                    req = self.traded_coins[coin]
                    if (signal.tf == req.tf and
                        signal.anomaly_type == req.anomaly_type and
                        signal.direction_hint == req.direction and
                        pred.confidence >= self.config["trading"]["min_confidence"][signal.anomaly_type.value]):

                        position = self.risk_manager.calculate_position(signal, pred, all_tfs[signal.tf])
                        if position:
                            await self.order_executor.execute_open(position)

            # 5. Симуляция закрытия виртуальных позиций на текущей свече
            self._simulate_existing_positions(coin, all_tfs)

        except Exception as e:
            logger.exception("Ошибка обработки монеты", coin=coin, error=str(e))

    def _simulate_existing_positions(self, coin: str, tfs: Dict[str, pl.DataFrame]):
        """Проверяет и закрывает виртуальные позиции по TP/SL."""
        df_1m = tfs.get("1m")
        if df_1m is None or df_1m.is_empty():
            return

        latest = df_1m.tail(1)
        new_price = latest["close"][0]
        timestamp = latest["timestamp"][0]

        self.virtual_trader.update_on_new_candle(coin, new_price, timestamp)

    async def _console_command_handler(self):
        logger.info("Консольные команды: pr, exit")

        while True:
            try:
                cmd = await asyncio.to_thread(input, ">>> ")
                cmd = cmd.strip().lower()

                if cmd == "pr":
                    await self.show_pr_table()
                elif cmd in ("exit", "quit"):
                    logger.info("Выход из консоли")
                    break
                elif cmd:
                    print("Команды: pr, exit")
            except EOFError:
                break
            except Exception as e:
                logger.error("Ошибка консоли", error=str(e))

            await asyncio.sleep(0.1)

    async def show_pr_table(self, min_pr: float = 0.0, min_deals: int = 5, limit: int = 30):
        df = self.storage.get_pr_table(min_pr=min_pr, min_deals=min_deals)

        if df.is_empty():
            print("PR таблица пуста.")
            return

        top = df.head(limit)

        print("\n" + "="*100)
        print(f"Топ-{limit} по PR (PR ≥ {min_pr}, сделок ≥ {min_deals})")
        print("="*100)
        print(top.select([
            "symbol", "tf", "period", "anomaly_type", "direction",
            pl.col("pr_value").round(4).alias("PR"),
            pl.col("winrate").round(2).alias("Win%"),
            "total_deals", "tp_hits", "sl_hits", "last_update"
        ]).to_pandas().to_string(index=False))
        print("="*100 + "\n")

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        path = f"pr_table_{timestamp}.csv"
        top.write_csv(path)
        logger.info("PR сохранён", path=path)

    async def on_new_candle(self, coin: str, tf: str, candle: dict):
        """Точка входа из WebSocket."""
        if tf != "1m":
            return  # основной источник — 1m

        new_row = pl.DataFrame([candle])

        if coin not in self.current_dfs:
            self.current_dfs[coin] = {}
        if "1m" not in self.current_dfs[coin]:
            self.current_dfs[coin]["1m"] = new_row
        else:
            self.current_dfs[coin]["1m"] = pl.concat([
                self.current_dfs[coin]["1m"],
                new_row
            ]).unique("timestamp").sort("timestamp")

        await self._process_coin(coin, self.current_dfs[coin])