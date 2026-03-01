"""
src/backtest/pr_calculator.py

=== Основной принцип работы файла ===

Расчёт Profitable Rating по одной таблице pr_snapshots.

Ключевые функции:
- update_pr после каждой закрытой позиции
- get_best_config для выбора лучшей комбинации TF + anomaly + direction
- get_stats для отображения в UI и логировании
"""

from src.core.enums import AnomalyType, Direction
from src.data.storage import Storage

class PRCalculator:
    def __init__(self):
        self.storage = Storage()

    def update_pr(self, symbol: str, anomaly_type: str, direction: str, hit_tp: bool, pl: float):
        """Обновляет PR после каждой закрытой позиции"""
        # FIX Фаза 3: учёт комиссии (round-trip)
        commission = 0.0004
        pl_adjusted = pl * (1 - commission * 2)

        self.storage.execute("""
            INSERT INTO pr_snapshots 
            (symbol, anomaly_type, direction, pr_value, winrate, total_deals, tp_hits, sl_hits, last_update)
            VALUES (?, ?, ?, ?, ?, 1, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(symbol, anomaly_type, direction) DO UPDATE SET
                pr_value = pr_value + ?,
                total_deals = total_deals + 1,
                tp_hits = tp_hits + ?,
                sl_hits = sl_hits + ?,
                last_update = CURRENT_TIMESTAMP
        """, [
            symbol, anomaly_type, direction, pl_adjusted, 1 if hit_tp else 0,
            pl_adjusted, 1 if hit_tp else 0, 0 if hit_tp else 1
        ])

    def get_best_config(self, symbol: str):
        """Возвращает лучшую комбинацию для монеты"""
        row = self.storage.fetch_one("""
            SELECT anomaly_type, direction, pr_value 
            FROM pr_snapshots 
            WHERE symbol = ? 
            ORDER BY pr_value DESC LIMIT 1
        """, [symbol])
        if not row:
            return None
        return {
            "anomaly_type": row[0],
            "direction": row[1],
            "pr_value": row[2]
        }