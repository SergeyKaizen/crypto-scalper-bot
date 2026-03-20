"""
src/backtest/pr_calculator.py
"""

from src.core.enums import AnomalyType
from src.data.storage import Storage

class PRCalculator:
    def __init__(self):
        self.storage = Storage()
        self.con = self.storage.con
        self._create_table()

    def _create_table(self):
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS pr_snapshots (
                symbol TEXT,
                anomaly_type TEXT,
                direction TEXT,
                tf TEXT,
                window INTEGER,
                tp_count INTEGER DEFAULT 0,
                sl_count INTEGER DEFAULT 0,
                expected_tp_length FLOAT DEFAULT 0,
                expected_sl_length FLOAT DEFAULT 0,
                last_update TIMESTAMP,
                PRIMARY KEY (symbol, anomaly_type, direction, tf, window)
            )
        """)

    def update_pr(self, symbol: str, anomaly_type: str, direction: str, tf: str, window: int, hit_tp: bool, tp_length: float, sl_length: float):
        """
        Обновление PR строго по ТЗ (утверждено):
        PR = (tp_count * tp_length) - (sl_count * sl_length)
        где tp_length / sl_length — ожидаемые значения НА МОМЕНТ СИГНАЛА (из calculate_tp_sl).
        Суммы по истории убраны полностью.
        """
        tp_count_delta = 1 if hit_tp else 0
        sl_count_delta = 0 if hit_tp else 1
        expected_tp = tp_length if hit_tp else 0
        expected_sl = sl_length if not hit_tp else 0

        self.con.execute("""
            INSERT INTO pr_snapshots 
            (symbol, anomaly_type, direction, tf, window, 
             tp_count, sl_count, expected_tp_length, expected_sl_length, last_update)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(symbol, anomaly_type, direction, tf, window) DO UPDATE SET
                tp_count = tp_count + excluded.tp_count,
                sl_count = sl_count + excluded.sl_count,
                expected_tp_length = COALESCE(expected_tp_length, excluded.expected_tp_length),
                expected_sl_length = COALESCE(expected_sl_length, excluded.expected_sl_length),
                last_update = CURRENT_TIMESTAMP
        """, (symbol, anomaly_type, direction, tf, window, 
              tp_count_delta, sl_count_delta, expected_tp, expected_sl))

        self.con.commit()

    def get_best_config(self, symbol: str):
        """Выбор лучшего условия по формуле ТЗ (PR = tp_count * expected_tp - sl_count * expected_sl)"""
        row = self.con.execute("""
            SELECT anomaly_type, direction, tf, window,
                   tp_count, sl_count, expected_tp_length, expected_sl_length,
                   (tp_count * expected_tp_length - sl_count * expected_sl_length) as pr_value
            FROM pr_snapshots 
            WHERE symbol = ?
            ORDER BY pr_value DESC LIMIT 1
        """, [symbol]).fetchone()

        if not row or row[4] == 0 and row[5] == 0:
            return None

        anomaly_type = row[0]
        direction = row[1]
        tf = row[2]
        window = row[3]
        pr_value = row[8] or 0

        setting = f"{tf}_P{window}_{anomaly_type}_{direction}"

        return {
            "anomaly_type": anomaly_type,
            "direction": direction,
            "tf": tf,
            "window": window,
            "pr_value": pr_value,
            "setting": setting,
            "expected_tp_length": row[6] or 0,
            "expected_sl_length": row[7] or 0
        }

    def get_stats(self, symbol: str):
        """Статистика для отладки (по ТЗ)"""
        row = self.con.execute("""
            SELECT SUM(tp_count), SUM(sl_count)
            FROM pr_snapshots 
            WHERE symbol = ?
        """, [symbol]).fetchone()
        if not row or row[1] == 0:
            return {"total_trades": 0, "pr_ls": 0, "win_rate": 0}
        tp = row[0] or 0
        sl = row[1] or 0
        total = tp + sl
        pr_ls = (tp * 1.0) / (sl + 1.0)
        win_rate = tp / total if total > 0 else 0
        return {"total_trades": total, "pr_ls": pr_ls, "win_rate": win_rate}