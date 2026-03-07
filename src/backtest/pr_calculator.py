"""
src/backtest/pr_calculator.py
"""

from src.core.enums import AnomalyType, Direction
from src.data.storage import Storage

class PRCalculator:
    def __init__(self):
        self.storage = Storage()
        self.con = self.storage.con  # FIX: совместимость с Storage из Группы 1

    def update_pr(self, symbol: str, anomaly_type: str, direction: str, hit_tp: bool, pl: float):
        commission = 0.0004
        pl_adjusted = pl * (1 - commission * 2)

        query = """
            INSERT INTO pr_snapshots 
            (symbol, anomaly_type, direction, pr_value, winrate, total_deals, tp_hits, sl_hits, last_update)
            VALUES (?, ?, ?, ?, ?, 1, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(symbol, anomaly_type, direction) DO UPDATE SET
                pr_value = pr_value + ?,
                total_deals = total_deals + 1,
                tp_hits = tp_hits + ?,
                sl_hits = sl_hits + ?,
                last_update = CURRENT_TIMESTAMP
        """
        self.con.execute(query, [
            symbol, anomaly_type, direction, pl_adjusted, 1 if hit_tp else 0,
            pl_adjusted, 1 if hit_tp else 0, 0 if hit_tp else 1
        ])
        self.con.commit()

    def get_best_config(self, symbol: str):
        row = self.con.execute("""
            SELECT anomaly_type, direction, pr_value 
            FROM pr_snapshots 
            WHERE symbol = ? 
            ORDER BY pr_value DESC LIMIT 1
        """, [symbol]).fetchone()
        if not row:
            return None
        return {
            "anomaly_type": row[0],
            "direction": row[1],
            "pr_value": row[2]
        }

    def get_stats(self, symbol: str):
        return {"total_trades": 0, "total_pr": 0, "win_rate": 0}