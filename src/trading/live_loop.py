# src/trading/live_loop.py
"""
Главный цикл бота.
Добавлена обработка консольной команды 'pr' — показывает таблицу PR и сохраняет в CSV.
Вывод только по ручному вводу (без автоповтора каждые 10 минут).
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


class LiveLoop:
    def __init__(self):
        self.config = get_config()
        self.feature_engine = FeatureEngine(self.config)
        self.inference = InferenceEngine(self.config)
        self.order_executor = OrderExecutor(self.config)
        self.virtual_trader = VirtualTrader()
        self.risk_manager = RiskManager(self.config)
        self.ws_manager = WebSocketManager(self)

        self.current_dfs: Dict[str, Dict[str, pl.DataFrame]] = {}
        self.traded_coins: Dict[str, TradeConfig] = {}

        logger.info("LiveLoop инициализирован")

    async def run(self):
        logger.info("Запуск основного цикла")
        await self.ws_manager.start()

        # Запускаем обработчик консольных команд в фоне
        asyncio.create_task(self._console_command_handler())

        try:
            while True:
                await asyncio.sleep(0.25)
                await self._process_all_coins()
        except asyncio.CancelledError:
            logger.info("Цикл остановлен")
        except Exception as e:
            logger.exception("Критическая ошибка в live_loop", error=str(e))
        finally:
            await self.ws_manager.stop()
            logger.info("LiveLoop завершён")

    async def _process_all_coins(self):
        for coin, tfs in list(self.current_dfs.items()):
            await self._process_coin(coin, tfs)

    async def _process_coin(self, coin: str, tfs: Dict[str, pl.DataFrame]):
        # Здесь твоя существующая логика обработки монеты
        # ... (не меняем, оставляем как было)
        pass

    async def _console_command_handler(self):
        """
        Фоновая задача для чтения команд из терминала.
        Поддерживается только команда 'pr' — показывает таблицу PR и сохраняет CSV.
        """
        logger.info("Консольный ввод активирован. Доступные команды: pr, exit")

        while True:
            try:
                cmd = await asyncio.to_thread(input, ">>> ")
                cmd = cmd.strip().lower()

                if cmd == "pr":
                    await self.show_pr_table(min_pr=0.0, min_deals=5, limit=30)
                elif cmd in ("exit", "quit", "q"):
                    logger.info("Выход из консольного ввода по команде")
                    break
                elif cmd:
                    logger.warning("Неизвестная команда", cmd=cmd)
                    print("Доступные команды: pr, exit")

            except EOFError:  # Ctrl+D или конец ввода
                logger.info("Конец ввода (EOF)")
                break
            except Exception as e:
                logger.error("Ошибка в консольном обработчике", error=str(e))

            await asyncio.sleep(0.1)  # не жрём CPU

    async def show_pr_table(self, min_pr: float = 0.0, min_deals: int = 5, limit: int = 30):
        """
        Выводит таблицу PR в консоль (топ по pr_value) и сразу сохраняет в CSV.
        Вызывается ТОЛЬКО по команде 'pr' из терминала.
        """
        df = self.storage.get_pr_table(min_pr=min_pr, min_deals=min_deals)

        if df.is_empty():
            logger.info("Таблица PR пуста или не прошла фильтр",
                        min_pr=min_pr,
                        min_deals=min_deals)
            print("Нет монет, удовлетворяющих условиям (PR ≥ {} и сделок ≥ {}).".format(min_pr, min_deals))
            return

        top = df.head(limit)

        logger.info("Команда 'pr' выполнена",
                    shown_rows=len(top),
                    total_rows=len(df),
                    min_pr=min_pr,
                    min_deals=min_deals)

        # Красивый текстовый вывод в консоль
        print("\n" + "="*110)
        print(f"Текущий топ-{limit} монет по Profitable Rating "
              f"(фильтр: PR ≥ {min_pr}, сделок ≥ {min_deals})")
        print("="*110)
        print(top.select([
            "symbol",
            "tf",
            "period",
            "anomaly_type",
            "direction",
            pl.col("pr_value").round(4).alias("PR"),
            pl.col("winrate").round(2).alias("Win%"),
            "total_deals",
            "tp_hits",
            "sl_hits",
            "last_update"
        ]).to_pandas().to_string(index=False))
        print("="*110 + "\n")

        # Автоматический экспорт в CSV
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        csv_path = f"pr_table_{timestamp}.csv"
        top.write_csv(csv_path)
        logger.info("Топ PR сохранён в CSV", path=csv_path, rows=len(top))
        print(f"Таблица сохранена в файл: {csv_path}\n")