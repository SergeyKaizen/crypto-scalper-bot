# src/data/storage.py
"""
Хранилище данных: свечи, PR-снимки, whitelist монет.

Особенности реализации:
- На телефоне (phone_tiny): SQLite — лёгкий, не требует установки, маленький размер БД
- На Colab / сервере: DuckDB — очень быстрый, SQL + Polars-native, идеален для больших данных
- Таблицы:
  - candles (свечи по символам и TF, upsert по primary key)
  - pr_snapshots (снимки PR после каждой закрытой позиции или полного бэктеста)
  - whitelist (активные монеты после backtest_all.py, обновляется полностью)
- Upsert свечей (INSERT OR REPLACE) — избегаем дубликатов при докачке
- Автоматическое создание таблиц при инициализации
- Методы для работы с whitelist (обновление после бэктеста, получение списка)
- Сохранение PR-снимков с timestamp для истории
"""

import os
import time
import sqlite3
import duckdb
import polars as pl
from typing import List, Dict, Optional

from src.core.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

class Storage:
    def __init__(self, config: dict):
        self.config = config
        self.hardware = config["hardware_mode"]

        # Пути к БД
        base_dir = config.get("paths", {}).get("data_dir", "data")
        os.makedirs(base_dir, exist_ok=True)

        if self.hardware == "phone_tiny":
            self.db_path = os.path.join(base_dir, "storage_phone.db")  # SQLite
            self.conn = sqlite3.connect(self.db_path)
            self.is_duckdb = False
        else:
            self.db_path = os.path.join(base_dir, "storage.duckdb")    # DuckDB
            self.conn = duckdb.connect(self.db_path)
            self.is_duckdb = True

        self._create_tables()

    def _create_tables(self):
        """Создаёт все необходимые таблицы (если не существуют)"""
        if self.is_duckdb:
            # DuckDB
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS candles (
                    symbol VARCHAR,
                    tf VARCHAR,
                    open_time BIGINT,
                    open DOUBLE,
                    high DOUBLE,
                    low DOUBLE,
                    close DOUBLE,
                    volume DOUBLE,
                    buy_volume DOUBLE,
                    PRIMARY KEY (symbol, tf, open_time)
                )
            """)
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS pr_snapshots (
                    symbol VARCHAR,
                    timestamp BIGINT,
                    pr DOUBLE,
                    trades_count INTEGER,
                    winrate DOUBLE,
                    profit_factor DOUBLE,
                    max_drawdown DOUBLE,
                    PRIMARY KEY (symbol, timestamp)
                )
            """)
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS whitelist (
                    symbol VARCHAR PRIMARY KEY,
                    pr DOUBLE,
                    last_updated BIGINT
                )
            """)
        else:
            # SQLite (phone)
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS candles (
                    symbol TEXT,
                    tf TEXT,
                    open_time INTEGER,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume REAL,
                    buy_volume REAL,
                    PRIMARY KEY (symbol, tf, open_time)
                )
            """)
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS pr_snapshots (
                    symbol TEXT,
                    timestamp INTEGER,
                    pr REAL,
                    trades_count INTEGER,
                    winrate REAL,
                    profit_factor REAL,
                    max_drawdown REAL,
                    PRIMARY KEY (symbol, timestamp)
                )
            """)
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS whitelist (
                    symbol TEXT PRIMARY KEY,
                    pr REAL,
                    last_updated INTEGER
                )
            """)
        self.conn.commit()

    def save_candles(self, symbol: str, tf: str, df: pl.DataFrame):
        """
        Сохраняет новые свечи (upsert — вставляет или заменяет по primary key).
        """
        if df.is_empty():
            return

        df = df.with_columns([
            pl.lit(symbol).alias("symbol"),
            pl.lit(tf).alias("tf")
        ]).unique(subset=["symbol", "tf", "open_time"])

        if self.is_duckdb:
            self.conn.register("new_candles", df)
            self.conn.execute("""
                INSERT OR REPLACE INTO candles
                SELECT * FROM new_candles
            """)
        else:
            # SQLite — через pandas (простой и надёжный способ)
            df.to_pandas().to_sql("candles", self.conn, if_exists="append", index=False)

        logger.debug(f"Сохранено {len(df)} свечей {symbol} {tf}")

    def load_candles(self, symbol: str, tf: str, limit: int = 1000, min_timestamp: Optional[int] = None) -> Optional[pl.DataFrame]:
        """
        Загружает последние свечи (или с min_timestamp).
        """
        query = f"""
            SELECT * FROM candles
            WHERE symbol = '{symbol}' AND tf = '{tf}'
        """
        if min_timestamp:
            query += f" AND open_time >= {min_timestamp}"

        query += f" ORDER BY open_time DESC LIMIT {limit}"

        if self.is_duckdb:
            df = self.conn.execute(query).pl()
        else:
            import pandas as pd
            df = pl.from_pandas(pd.read_sql_query(query, self.conn))

        if df.is_empty():
            return None

        return df.sort("open_time")

    def save_pr_snapshot(self, symbol: str, pr_data: Dict):
        """
        Сохраняет снимок PR после закрытия позиции или полного бэктеста.
        """
        now = int(time.time())
        row = {
            "symbol": symbol,
            "timestamp": now,
            "pr": pr_data.get("profit_factor", 0.0),
            "trades_count": pr_data.get("trades_count", 0),
            "winrate": pr_data.get("winrate", 0.0),
            "profit_factor": pr_data.get("profit_factor", 0.0),
            "max_drawdown": pr_data.get("max_drawdown", 0.0)
        }

        df = pl.DataFrame([row])

        if self.is_duckdb:
            self.conn.register("new_pr", df)
            self.conn.execute("""
                INSERT OR REPLACE INTO pr_snapshots
                SELECT * FROM new_pr
            """)
        else:
            df.to_pandas().to_sql("pr_snapshots", self.conn, if_exists="append", index=False)

        logger.debug(f"Сохранён PR-снимок {symbol}: PR={row['pr']:.2f}")

    def update_whitelist(self, symbols_data: List[Dict]):
        """
        Обновляет таблицу whitelist после backtest_all.py.
        symbols_data = [{"symbol": str, "pr": float, ...}]
        """
        if not symbols_data:
            return

        now = int(time.time())
        df = pl.DataFrame({
            "symbol": [d["symbol"] for d in symbols_data],
            "pr": [d.get("pr", 0.0) for d in symbols_data],
            "last_updated": [now] * len(symbols_data)
        })

        if self.is_duckdb:
            self.conn.register("new_whitelist", df)
            self.conn.execute("DELETE FROM whitelist")
            self.conn.execute("INSERT INTO whitelist SELECT * FROM new_whitelist")
        else:
            df.to_pandas().to_sql("whitelist", self.conn, if_exists="replace", index=False)

        logger.info(f"Whitelist обновлён: {len(df)} монет")

    def get_whitelist(self) -> List[str]:
        """Возвращает список активных монет для торговли"""
        if self.is_duckdb:
            df = self.conn.execute("SELECT symbol FROM whitelist").pl()
        else:
            import pandas as pd
            df = pl.from_pandas(pd.read_sql_query("SELECT symbol FROM whitelist", self.conn))

        return df["symbol"].to_list() if not df.is_empty() else []

    def get_pr_history(self, symbol: str, limit: int = 100) -> pl.DataFrame:
        """История PR-снимков для одной монеты"""
        if self.is_duckdb:
            df = self.conn.execute(f"""
                SELECT * FROM pr_snapshots
                WHERE symbol = '{symbol}'
                ORDER BY timestamp DESC
                LIMIT {limit}
            """).pl()
        else:
            df = pl.from_pandas(pd.read_sql_query(f"""
                SELECT * FROM pr_snapshots
                WHERE symbol = '{symbol}'
                ORDER BY timestamp DESC
                LIMIT {limit}
            """, self.conn))

        return df

    def close(self):
        """Закрывает соединение с БД"""
        self.conn.close()
        logger.debug("Соединение с БД закрыто")


if __name__ == "__main__":
    config = load_config()
    storage = Storage(config)

    # Тест сохранения whitelist
    test_data = [
        {"symbol": "BTCUSDT", "pr": 1.85},
        {"symbol": "ETHUSDT", "pr": 1.62}
    ]
    storage.update_whitelist(test_data)

    print("Whitelist:", storage.get_whitelist())
    storage.close()