# ================================================
# 6. src/backtest/pr_calculator.py (полная новая версия)
# ================================================
class PRCalculator:
    """Расчёт Profitable Rating по одной таблице pr_snapshots."""

    def __init__(self, storage):
        self.storage = storage

    def update_after_trade(self, trade: TradeResult):
        """Обновляет PR после каждой закрытой виртуальной сделки."""
        # trade содержит: coin, tf, period, anomaly_type, direction, is_tp, pr_contribution, winrate
        self.storage.execute("""
            INSERT INTO pr_snapshots 
            (coin, tf, period, anomaly_type, direction, pr_value, winrate, 
             total_deals, tp_hits, sl_hits, last_update)
            VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(coin, tf, period, anomaly_type, direction) DO UPDATE SET
                pr_value = excluded.pr_value,
                winrate = excluded.winrate,
                total_deals = pr_snapshots.total_deals + 1,
                tp_hits = pr_snapshots.tp_hits + ?,
                sl_hits = pr_snapshots.sl_hits + ?,
                last_update = CURRENT_TIMESTAMP
        """, [
            trade.coin, trade.tf, trade.period, trade.anomaly_type.value, trade.direction.value,
            trade.pr_contribution, trade.winrate,
            1 if trade.is_tp else 0,
            0 if trade.is_tp else 1,
            1 if trade.is_tp else 0,
            0 if trade.is_tp else 1
        ])

    def get_best_config(self, coin: str) -> dict:
        """Возвращает лучшую комбинацию для монеты."""
        row = self.storage.fetch_one("""
            SELECT tf, period, anomaly_type, direction, pr_value
            FROM pr_snapshots 
            WHERE coin = ? 
            ORDER BY pr_value DESC 
            LIMIT 1
        """, [coin])
        if not row:
            return None
        return {
            "tf": row[0],
            "period": row[1],
            "anomaly_type": AnomalyType(row[2]),
            "direction": Direction(row[3]),
            "pr_value": row[4]
        }