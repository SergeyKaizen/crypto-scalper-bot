"""
src/backtest/engine.py

Движок бэктеста — симулирует торговлю на исторических данных по всем TF одновременно.
Использует:
- FeatureEngine → генерирует признаки
- Model inference → даёт вероятности
- EntryManager → решает, входить или нет
- RiskManager → рассчитывает размер позиции
- TP_SL_Manager → управляет выходами (TP, SL, trailing, partials)
- VirtualTrader → симулирует исполнение с учётом slippage, fees, funding

Ключевые принципы:
- Нет look-ahead (все решения на данных до текущей свечи)
- Мульти-TF consensus
- Monte-Carlo для оценки распределения исходов
- Сохранение всех сделок и equity curve
"""

import logging
import numpy as np
import pandas as pd
import polars as pl
from datetime import datetime
from typing import Dict, List, Tuple, Optional

from src.core.config import load_config
from src.core.constants import TF_SHORT_NAMES
from src.data.storage import Storage
from src.features.feature_engine import FeatureEngine
from src.model.inference import InferenceEngine
from src.trading.entry_manager import EntryManager
from src.trading.risk_manager import RiskManager
from src.trading.tp_sl_manager import TPSLManager
from src.trading.virtual_trader import VirtualTrader
from src.utils.logger import setup_logger

logger = logging.getLogger(__name__)

class BacktestEngine:
    def __init__(self, config: dict, symbol: str):
        self.config = config
        self.symbol = symbol
        self.storage = Storage(config)
        self.feature_engine = FeatureEngine(config)
        self.inference = InferenceEngine(config)
        self.entry_manager = EntryManager(config)
        self.risk_manager = RiskManager(config)
        self.tp_sl_manager = TPSLManager(config)
        self.virtual_trader = VirtualTrader(config, symbol)

        self.timeframes = config["timeframes"]
        self.seq_len = config["seq_len"]
        self.min_tf_consensus = config.get("min_tf_consensus", 2)
        self.results = {
            "trades": [],
            "equity_curve": [],
            "metrics": {}
        }

    def load_data(self) -> Dict[str, pl.DataFrame]:
        """Загружает последние N свечей по всем TF"""
        data = {}
        limit = self.config["data"]["backtest_candles"] + self.seq_len + 50  # запас
        for tf in self.timeframes:
            df = self.storage.get_candles(self.symbol, tf, limit=limit)
            if df is None or len(df) < self.seq_len + 10:
                logger.warning(f"Недостаточно данных для {self.symbol} {tf}")
                continue
            data[tf] = df
        return data

    def run_full_backtest(self) -> Dict:
        """Запуск полного бэктеста по сегментам"""
        data = self.load_data()
        if not data:
            logger.error(f"Нет данных для {self.symbol}")
            return {"error": "no data"}

        # Разбиваем на непересекающиеся сегменты (по 1000–2000 свечей)
        segments = self._split_into_segments(data)

        for segment_data in segments:
            segment_results = self._simulate_segment(segment_data)
            self.results["trades"].extend(segment_results["trades"])
            self.results["equity_curve"].extend(segment_results["equity_curve"])

        # Финальный расчёт метрик
        self.results["metrics"] = self._calculate_metrics()
        self.storage.save_backtest_results(self.symbol, self.results)

        return self.results["metrics"]

    def _split_into_segments(self, data: Dict[str, pl.DataFrame]) -> List[Dict[str, pl.DataFrame]]:
        """Разбивает данные на сегменты для экономии памяти"""
        min_len = min(len(df) for df in data.values())
        segment_size = 1500  # ~1–2 месяца на 1m
        segments = []
        for start in range(self.seq_len, min_len - 100, segment_size):
            end = min(start + segment_size, min_len)
            segment = {tf: df.slice(start - self.seq_len, end - start + self.seq_len) for tf, df in data.items()}
            segments.append(segment)
        return segments

    def _simulate_segment(self, data: Dict[str, pl.DataFrame]) -> Dict:
        """Реальная симуляция одного сегмента без рандома"""
        trades = []
        equity = [self.config["initial_balance"]]
        current_time = 0

        # Идём по самой младшей TF (1m)
        base_tf = "1m"
        base_df = data.get(base_tf)
        if base_df is None:
            return {"trades": [], "equity_curve": []}

        for i in range(self.seq_len, len(base_df) - 1):
            current_candle = base_df.row(i)
            current_time = current_candle["open_time"]

            # Собираем окно по всем TF до текущего момента
            window = {}
            for tf, df in data.items():
                tf_window = df.filter(pl.col("open_time") <= current_time).tail(self.seq_len + 10)
                if len(tf_window) < self.seq_len:
                    continue
                window[tf] = tf_window

            if len(window) < self.min_tf_consensus:
                equity.append(equity[-1])
                continue

            # Генерируем признаки
            features = self.feature_engine.build_features(window)

            # Inference модели
            prob_long, prob_short, uncertainty = self.inference.predict(features)

            # Проверяем условие входа
            entry_signal = self.entry_manager.check_entry(
                prob_long=prob_long,
                prob_short=prob_short,
                uncertainty=uncertainty,
                features=features,
                current_price=current_candle["close"],
                current_time=current_time
            )

            if entry_signal:
                direction = entry_signal["direction"]
                confidence = entry_signal["confidence"]

                # Рассчитываем размер позиции
                position_size = self.risk_manager.calculate_position_size(
                    current_price=current_candle["close"],
                    direction=direction,
                    confidence=confidence
                )

                # Открываем виртуальную позицию
                entry_order = self.virtual_trader.open_position(
                    direction=direction,
                    price=current_candle["close"],
                    size=position_size,
                    timestamp=current_time
                )

                if entry_order:
                    self.tp_sl_manager.register_position(entry_order)

                    exit_info = self._simulate_position_lifecycle(
                        data, i, current_time, entry_order, base_tf
                    )

                    if exit_info:
                        trade = {
                            "entry_time": current_time,
                            "direction": direction,
                            "entry_price": entry_order["price"],
                            "exit_price": exit_info["exit_price"],
                            "size": position_size,
                            "pnl": exit_info["pnl"],
                            "pnl_pct": exit_info["pnl_pct"],
                            "duration": exit_info["duration"],
                            "reason": exit_info["reason"]
                        }
                        trades.append(trade)

                        new_equity = equity[-1] + exit_info["pnl"]
                        equity.append(new_equity)
                        self.tp_sl_manager.close_position(entry_order["id"])
                    else:
                        equity.append(equity[-1])
                else:
                    equity.append(equity[-1])
            else:
                updates = self.tp_sl_manager.update_all_positions(
                    current_price=current_candle["close"],
                    current_time=current_time,
                    high=current_candle["high"],
                    low=current_candle["low"]
                )

                for update in updates:
                    if update["closed"]:
                        pnl = self.virtual_trader.close_position(
                            update["position_id"],
                            update["exit_price"],
                            current_time
                        )
                        if pnl is not None:
                            # FIX Фаза 3: учёт commission и slippage
                            commission = self.config["trading"].get("commission", 0.0004)
                            slippage = self.config["trading"].get("slippage_pct", 0.0005)
                            pnl = pnl * (1 - commission * 2) - (slippage * abs(pnl))
                            equity[-1] += pnl

                equity.append(equity[-1])

        return {"trades": trades, "equity_curve": equity}

    def _simulate_position_lifecycle(self, data: Dict, start_idx: int, start_time: int, entry_order: Dict, base_tf: str) -> Optional[Dict]:
        """Симулирует жизненный цикл одной позиции до её закрытия"""
        base_df = data[base_tf]
        for j in range(start_idx + 1, len(base_df)):
            candle = base_df.row(j)
            current_price = candle["close"]
            high = candle["high"]
            low = candle["low"]
            ts = candle["open_time"]

            updates = self.tp_sl_manager.update_position(
                entry_order["id"],
                current_price=current_price,
                high=high,
                low=low,
                timestamp=ts
            )

            if updates and updates["closed"]:
                exit_price = updates["exit_price"]
                pnl = self.virtual_trader.calculate_pnl(
                    entry_order["price"],
                    exit_price,
                    entry_order["size"],
                    entry_order["direction"]
                )
                # FIX Фаза 3: учёт commission и slippage
                commission = self.config["trading"].get("commission", 0.0004)
                slippage = self.config["trading"].get("slippage_pct", 0.0005)
                pnl = pnl * (1 - commission * 2) - (slippage * abs(pnl))
                duration = ts - start_time
                return {
                    "exit_price": exit_price,
                    "pnl": pnl,
                    "pnl_pct": pnl / (entry_order["price"] * entry_order["size"]) * 100,
                    "duration": duration,
                    "reason": updates["reason"]
                }

        # Если дошли до конца сегмента — закрываем по рынку
        last_candle = base_df.row(-1)
        exit_price = last_candle["close"]
        pnl = self.virtual_trader.calculate_pnl(
            entry_order["price"], exit_price, entry_order["size"], entry_order["direction"]
        )
        # FIX Фаза 3: учёт commission и slippage
        commission = self.config["trading"].get("commission", 0.0004)
        slippage = self.config["trading"].get("slippage_pct", 0.0005)
        pnl = pnl * (1 - commission * 2) - (slippage * abs(pnl))
        return {
            "exit_price": exit_price,
            "pnl": pnl,
            "pnl_pct": pnl / (entry_order["price"] * entry_order["size"]) * 100 if entry_order["size"] else 0,
            "duration": last_candle["open_time"] - start_time,
            "reason": "end_of_segment"
        }

    def _calculate_metrics(self) -> Dict:
        """Расчёт итоговых метрик (PR, winrate, drawdown и т.д.)"""
        trades = self.results["trades"]
        if not trades:
            return {"total_trades": 0, "winrate": 0, "pr_ls": 0}

        df = pd.DataFrame(trades)
        wins = df[df["pnl"] > 0]
        losses = df[df["pnl"] < 0]

        total_trades = len(df)
        winrate = len(wins) / total_trades if total_trades else 0
        avg_win = wins["pnl"].mean() if not wins.empty else 0
        avg_loss = losses["pnl"].mean() if not losses.empty else 0

        pr_l = avg_win / abs(avg_loss) if avg_loss != 0 else float('inf')
        pr_s = pr_l

        equity = np.array(self.results["equity_curve"])
        drawdowns = (equity.cummax() - equity) / equity.cummax()
        max_dd = drawdowns.max() if len(drawdowns) > 0 else 0

        return {
            "total_trades": total_trades,
            "winrate": winrate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "pr_ls": pr_l,
            "max_drawdown": max_dd,
            "final_equity": equity[-1] if len(equity) > 0 else self.config["initial_balance"]
        }