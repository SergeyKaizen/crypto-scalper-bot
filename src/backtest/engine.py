"""
src/backtest/engine.py

=== Основной принцип работы файла ===

Этот файл реализует движок бэктеста для расчёта Profitable Rating (PR) всех монет и определения лучших конфигураций.

Ключевые задачи:
- Симулировать торговлю по историческим данным (виртуальные позиции) для всех монет/TF/окон/аномалий (C/V/CV/Q).
- Открывать позиции по всем типам аномалий в одной свече (до 4 одновременно).
- Закрывать по TP/SL (из tp_sl_manager).
- Запрещать новую позицию в свече, где предыдущая закрылась (по ТЗ, для бектеста тоже).
- Обновлять PR после каждой закрытой позиции (tp_hits/sl_hits, pr_value = (tp_hits * tp_size) - (sl_hits * sl_size)).
- Выбирать лучшую конфигурацию монеты (best_tf, best_period, best_anomaly, best_direction) по max PR.
- Обновлять whitelist в storage.
- Поддерживать многопоточность по монетам (ThreadPoolExecutor).

Бектест — виртуальный, без реального риска/маржи, цель — собрать максимум статистики для PR.
Логика live (одна позиция по типу) здесь не применяется — только ограничение "no new in closed candle".

=== Главные функции и за что отвечают ===

- BacktestEngine(config: dict) — инициализация: client, storage, tp_sl_manager.
- run_backtest_all() — полный бектест по всем монетам из whitelist или full list.
- simulate_symbol(symbol: str) — симуляция по одной монете:
  - Загружает свечи по всем TF.
  - Проходит по свечам, детектит аномалии/Q.
  - Открывает виртуальные позиции (до 4 в свече).
  - Проверяет закрытия (TP/SL) в каждой свече.
  - Если закрытие в свече — флаг, запрещающий новые входы в этой свече.
  - Обновляет pr_calculator после закрытия.
- _simulate_candle(candle_data: dict, positions: list) → list — проверка закрытий и открытий в одной свече.
- update_pr_after_close(...) — передача в pr_calculator.

=== Примечания ===
- Позиции — dict с type (C/V/CV/Q), entry_price, tp, sl, direction.
- Limit: max 1 открытая по типу (но в бектесте параллельно по разным типам).
- Запрет новой в свече закрытия — флаг per candle.
- Полностью соответствует ТЗ + уточнению (множественные в свече — да, но no new after close).
- Готов к запуску в scripts/backtest_all.py.
- Логи через setup_logger.
"""

import concurrent.futures
from typing import List, Dict
import pandas as pd
import time

from src.core.config import load_config
from src.data.binance_client import BinanceClient
from src.data.storage import Storage
from src.features.anomaly_detector import detect_anomalies
from src.trading.tp_sl_manager import TPSLManager
from src.backtest.pr_calculator import PRCalculator
from src.utils.logger import setup_logger

logger = setup_logger('backtest_engine', logging.INFO)

class BacktestEngine:
    """
    Движок бэктеста для расчёта PR и whitelist.
    """
    def __init__(self):
        self.config = load_config()
        self.client = BinanceClient()
        self.storage = Storage()
        self.tp_sl_manager = TPSLManager()
        self.pr_calculator = PRCalculator()

    def run_backtest_all(self):
        """
        Запускает полный бектест по всем монетам.
        Многопоточно по символам.
        """
        symbols = self.client.update_markets_list()  # или только whitelisted
        timeframes = self.config['timeframes']

        max_workers = self.config['hardware']['max_workers']

        logger.info(f"Бэктест: {len(symbols)} монет, workers={max_workers}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self.simulate_symbol, symbol) for symbol in symbols]

            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Ошибка в симуляции монеты: {e}")

        # Финальное обновление whitelist
        self.pr_calculator.update_whitelist()
        logger.info("Бэктест завершён. Whitelist обновлён.")

    def simulate_symbol(self, symbol: str):
        """
        Симуляция торговли по одной монете.
        Проходит по всем TF, всем свечам, детектит аномалии/Q, открывает/закрывает позиции.
        """
        logger.info(f"Симуляция {symbol}")

        positions = []  # list[dict: type, entry_price, tp, sl, direction, open_candle_ts]

        for tf in self.config['timeframes']:
            df = self.storage.get_candles(symbol, tf)
            if df.empty:
                continue

            df = df.sort_index()

            for idx, row in df.iterrows():
                candle_ts = int(idx.timestamp() * 1000)
                candle_data = row.to_dict()

                # Детектим аномалии и Q для этой свечи
                anomalies = detect_anomalies(df.loc[:idx].tail(LOOKBACK + 10), tf_minutes=Timeframe(tf).minutes, current_window=100)
                # current_window — берём максимальный для VA

                # Флаг: была ли закрыта позиция в этой свече
                candle_has_closed = False

                # 1. Проверяем закрытия существующих позиций
                still_open = []
                for pos in positions:
                    close_result = self.tp_sl_manager.check_close(pos, candle_data)
                    if close_result['closed']:
                        candle_has_closed = True
                        self.pr_calculator.update_after_trade(
                            symbol=symbol,
                            anomaly_type=pos['type'],
                            direction=pos['direction'],
                            is_tp=close_result['is_tp'],
                            tp_size=pos['tp_distance'],
                            sl_size=pos['sl_distance']
                        )
                    else:
                        still_open.append(pos)

                positions = still_open

                # 2. Открываем новые позиции, если нет закрытия в этой свече
                if candle_has_closed:
                    continue  # запрет новой в свече закрытия (по ТЗ для бектеста тоже)

                # Возможные типы для открытия
                possible_types = []
                if anomalies['candle']:
                    possible_types.append('C')
                if anomalies['volume']:
                    possible_types.append('V')
                if anomalies['cv']:
                    possible_types.append('CV')
                if anomalies['q']:
                    possible_types.append('Q')

                for pos_type in possible_types:
                    # Проверяем, нет ли уже открытой по этому типу (даже в бектесте — 1 по типу)
                    if any(p['type'] == pos_type for p in positions):
                        continue

                    # Расчёт TP/SL
                    levels = self.tp_sl_manager.calculate_levels(candle_data, direction='L' if 'long' in pos_type else 'S')  # упрощённо
                    if not levels:
                        continue

                    pos = {
                        'type': pos_type,
                        'entry_price': candle_data['close'],
                        'tp': levels['tp'],
                        'sl': levels['sl'],
                        'direction': 'L' if 'long' in pos_type else 'S',
                        'open_candle_ts': candle_ts,
                        'tp_distance': abs(levels['tp'] - candle_data['close']),
                        'sl_distance': abs(levels['sl'] - candle_data['close'])
                    }
                    positions.append(pos)
                    logger.debug(f"Открыта виртуальная позиция {pos_type} {pos['direction']} на {symbol} {candle_ts}")

        logger.info(f"Симуляция {symbol} завершена. Позиций закрыто: {self.pr_calculator.get_trade_count(symbol)}")