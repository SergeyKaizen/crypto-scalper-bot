"""
src/backtest/engine.py

Движок бэктеста — симулирует торговлю на исторических данных.
Теперь поддерживает walk_forward_windows + hybrid backtest+live режим.

После аудита (Phase 5):
- Hybrid-режим по твоей логике: backtest-сигнал → виртуальная позиция + проверка маркировки → реальная live-позиция
- Все open/close — только через PositionManager
- PR рассчитывается через PRCalculator (expected length)
"""

import logging
import numpy as np
import pandas as pd
import polars as pl
from datetime import datetime
from typing import Dict, List, Tuple, Optional

from src.core.config import load_config
from src.core.enums import AnomalyType
from src.data.storage import Storage
from src.features.feature_engine import FeatureEngine
from src.model.inference import InferenceEngine
from src.model.scenario_tracker import ScenarioTracker
from src.trading.entry_manager import EntryManager
from src.trading.risk_manager import RiskManager
from src.trading.tp_sl_manager import TP_SL_Manager
from src.trading.virtual_trader import VirtualTrader
from src.trading.position_manager import PositionManager
from src.backtest.pr_calculator import PRCalculator
from src.utils.logger import setup_logger

logger = logging.getLogger(__name__)

class BacktestEngine:
    def __init__(self, config: dict, symbol: str):
        self.config = config
        self.symbol = symbol
        self.storage = Storage()
        self.feature_engine = FeatureEngine(config)
        self.inference = InferenceEngine(config)
        self.scenario_tracker = ScenarioTracker()
        self.entry_manager = EntryManager(self.scenario_tracker)
        self.risk_manager = RiskManager()
        self.tp_sl_manager = TP_SL_Manager()
        self.virtual_trader = VirtualTrader()
        self.position_manager = PositionManager()
        self.pr_calculator = PRCalculator()

        self.timeframes = config["timeframes"]
        self.seq_len = config["seq_len"]
        self.min_tf_consensus = config.get("min_tf_consensus", 2)
        self.walk_forward_windows = config["backtest"].get("walk_forward_windows", 5)

        self.results = {
            "trades": [],
            "equity_curve": [],
            "metrics": {}
        }

    def load_data(self) -> Dict[str, pl.DataFrame]:
        data = {}
        limit = self.config["data"]["backtest_candles_per_tf"] + self.seq_len + 50
        for tf in self.timeframes:
            df = self.storage.get_candles(self.symbol, tf, limit=limit)
            if df is None or len(df) < self.seq_len + 10:
                continue
            data[tf] = df
        return data

    def run_full_backtest(self) -> Dict:
        data = self.load_data()
        if not data:
            return {"error": "no data"}

        segments = self._split_into_walk_forward_segments(data)

        for segment_data in segments:
            segment_results = self._simulate_segment(segment_data)
            self.results["trades"].extend(segment_results["trades"])
            self.results["equity_curve"].extend(segment_results["equity_curve"])

        self.results["metrics"] = self._calculate_metrics()
        self.storage.save_backtest_results(self.symbol, self.results)

        return self.results["metrics"]

    def _split_into_walk_forward_segments(self, data: Dict[str, pl.DataFrame]) -> List[Dict[str, pl.DataFrame]]:
        min_len = min(len(df) for df in data.values())
        segment_size = int(min_len / self.walk_forward_windows)
        segments = []

        for w in range(self.walk_forward_windows):
            start = self.seq_len + w * segment_size
            end = min(start + segment_size, min_len)
            segment = {tf: df.slice(start - self.seq_len, end - start + self.seq_len) for tf, df in data.items()}
            segments.append(segment)

        return segments

    def _simulate_segment(self, data: Dict[str, pl.DataFrame]) -> Dict:
        trades = []
        equity = [self.config.get("initial_deposit", 10000.0)]
        base_tf = "1m"
        base_df = data.get(base_tf)
        if base_df is None:
            return {"trades": [], "equity_curve": []}

        for i in range(self.seq_len, len(base_df) - 1):
            current_candle = base_df.row(i)
            current_time = current_candle["open_time"]

            window = {}
            for tf, df in data.items():
                tf_window = df.filter(pl.col("open_time") <= current_time).tail(self.seq_len + 10)
                if len(tf_window) < self.seq_len:
                    continue
                window[tf] = tf_window

            if len(window) < self.min_tf_consensus:
                equity.append(equity[-1])
                continue

            features = self.feature_engine.build_features(window)
            prob_long, prob_short, uncertainty = self.inference.predict(features)

            if prob_long > self.config.get("trading", {}).get("min_prob", 0.65):
                direction = "L"
                anomaly_type = AnomalyType.C.value
            elif prob_short > self.config.get("trading", {}).get("min_prob", 0.65):
                direction = "S"
                anomaly_type = AnomalyType.C.value
            else:
                equity.append(equity[-1])
                continue

            # === HYBRID LOGIC (по твоему ТЗ) ===
            marking = f"{base_tf}_P{self.seq_len}_{anomaly_type}_{direction}"
            wl = self.storage.get_whitelist_settings(self.symbol)
            should_open_live = wl and wl.get('anomaly_type') == anomaly_type and wl.get('direction') == direction

            tp_sl = self.tp_sl_manager.calculate_tp_sl(features, anomaly_type)
            tp_price = tp_sl.get('tp', current_candle["close"] * (1.02 if direction == 'L' else 0.98))
            sl_price = tp_sl.get('sl', current_candle["close"] * (0.98 if direction == 'L' else 1.02))

            size = self.risk_manager.calculate_position_size(
                symbol=self.symbol,
                entry_price=current_candle["close"],
                tp_price=tp_price,
                sl_price=sl_price
            )

            if size <= 0:
                equity.append(equity[-1])
                continue

            position_data = {
                'pos_id': f"bt_{self.symbol}_{current_time}",
                'symbol': self.symbol,
                'anomaly_type': anomaly_type,
                'direction': direction,
                'entry_price': current_candle["close"],
                'size': size,
                'tp': tp_price,
                'sl': sl_price,
                'mode': 'virtual',
                'feats': features,
                'marking': marking,
                'is_backtest_signal': True
            }

            # Всегда открываем виртуальную для PR
            self.position_manager.open_position(position_data, is_backtest_signal=True)

            # Если маркировка совпадает — открываем реальную live
            if should_open_live:
                live_data = position_data.copy()
                live_data['mode'] = 'real'
                self.position_manager.open_position(live_data, is_backtest_signal=False)
                logger.info(f"Hybrid: открыта LIVE позиция {direction} {self.symbol} (marking совпала)")

            equity.append(equity[-1])

            # Проверка закрытия
            closed = self.position_manager.check_and_close(current_candle["close"], current_time)
            if closed:
                for c in closed:
                    pnl = c.get("net_pnl", 0)
                    trades.append({
                        "entry_time": current_time,
                        "direction": direction,
                        "entry_price": current_candle["close"],
                        "exit_price": current_candle["close"],
                        "size": size,
                        "pnl": pnl,
                        "pnl_pct": (pnl / size) * 100 if size else 0,
                        "reason": "tp" if c.get("hit_tp") else "sl"
                    })

        return {"trades": trades, "equity_curve": equity}

    def _calculate_metrics(self) -> Dict:
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

        pr_ls = avg_win / abs(avg_loss) if avg_loss != 0 else float('inf')

        equity = np.array(self.results["equity_curve"])
        drawdowns = (equity.cummax() - equity) / equity.cummax()
        max_dd = drawdowns.max() if len(drawdowns) > 0 else 0

        return {
            "total_trades": total_trades,
            "winrate": winrate,
            "pr_ls": pr_ls,
            "max_drawdown": max_dd,
            "final_equity": equity[-1] if len(equity) > 0 else self.config.get("initial_deposit", 10000.0)
        }