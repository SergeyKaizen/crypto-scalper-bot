# src/data/storage.py
"""
Абстракция базы данных.
Поддерживает DuckDB (сервер/Colab) и SQLite (телефон).
Все методы безопасны для обоих бэкендов.
"""

import os
from pathlib import Path
from typing import Optional, Tuple, List

import duckdb
import polars as pl
import sqlite3

from ..core.config import get_config
from ..utils.logger import logger


class Storage:
    """Универсальный класс работы с данными."""

    def __init__(self):
        self.config = get_config()
        self.backend = self.config["storage"]["backend"]  # "duckdb" или "sqlite"
        self.db_path = Path(self.config["storage"]["path"])

        if self.backend == "duckdb":
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self.con = duckdb.connect(str(self.db_path))
            logger.info("DuckDB подключён", path=str(self.db_path))
        else:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self.con = sqlite3.connect(str(self.db_path))
            logger.info("SQLite подключён", path=str(self.db_path))

        self.init_db()

    def init_db(self):
        """Создаёт все таблицы при первом запуске."""
        if self.backend == "duckdb":
            self._init_duckdb()
        else:
            self._init_sqlite()

    def _init_duckdb(self):
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS candles (
                symbol      VARCHAR,
                timeframe   VARCHAR,
                ts          BIGINT,
                open        DOUBLE,
                high        DOUBLE,
                low         DOUBLE,
                close       DOUBLE,
                volume      DOUBLE,
                buy_volume  DOUBLE,
                PRIMARY KEY (symbol, timeframe, ts)
            )
        """)

        self.con.execute("""
            CREATE TABLE IF NOT EXISTS pr_snapshots (
                symbol        VARCHAR NOT NULL,
                tf            VARCHAR NOT NULL,
                period        INTEGER NOT NULL,
                anomaly_type  VARCHAR NOT NULL,
                direction     VARCHAR NOT NULL,
                pr_value      DOUBLE,
                winrate       DOUBLE,
                total_deals   BIGINT DEFAULT 0,
                tp_hits       BIGINT DEFAULT 0,
                sl_hits       BIGINT DEFAULT 0,
                last_update   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (symbol, tf, period, anomaly_type, direction)
            )
        """)

        self.con.execute("CREATE INDEX IF NOT EXISTS idx_pr_symbol ON pr_snapshots(symbol);")
        self.con.execute("CREATE INDEX IF NOT EXISTS idx_pr_value ON pr_snapshots(pr_value DESC);")

    def _init_sqlite(self):
        cur = self.con.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS candles (
                symbol      TEXT,
                timeframe   TEXT,
                ts          INTEGER,
                open        REAL,
                high        REAL,
                low         REAL,
                close       REAL,
                volume      REAL,
                buy_volume  REAL,
                PRIMARY KEY (symbol, timeframe, ts)
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS pr_snapshots (
                symbol        TEXT    NOT NULL,
                tf            TEXT    NOT NULL,
                period        INTEGER NOT NULL,
                anomaly_type  TEXT    NOT NULL,
                direction     TEXT    NOT NULL,
                pr_value      REAL,
                winrate       REAL,
                total_deals   INTEGER DEFAULT 0,
                tp_hits       INTEGER DEFAULT 0,
                sl_hits       INTEGER DEFAULT 0,
                last_update   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (symbol, tf, period, anomaly_type, direction)
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_pr_symbol ON pr_snapshots(symbol);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_pr_value ON pr_snapshots(pr_value DESC);")
        self.con.commit()

    def execute(self, query: str, params: Optional[Tuple] = None) -> None:
        """INSERT / UPDATE / CREATE"""
        if self.backend == "duckdb":
            self.con.execute(query, params or ())
        else:
            cur = self.con.cursor()
            cur.execute(query, params or ())
            self.con.commit()

    def fetch_one(self, query: str, params: Optional[Tuple] = None):
        if self.backend == "duckdb":
            return self.con.execute(query, params or ()).fetchone()
        else:
            cur = self.con.cursor()
            cur.execute(query, params or ())
            return cur.fetchone()

    def fetch_all(self, query: str, params: Optional[Tuple] = None) -> List:
        if self.backend == "duckdb":
            return self.con.execute(query, params or ()).fetchall()
        else:
            cur = self.con.cursor()
            cur.execute(query, params or ())
            return cur.fetchall()

    def insert_df(self, table: str, df: pl.DataFrame):
        if self.backend == "duckdb":
            self.con.execute(f"INSERT INTO {table} SELECT * FROM df", [df])
        else:
            df.write_database(table, self.con, if_table_exists="append", engine="sqlalchemy")

    # ================================================
    # НОВЫЕ МЕТОДЫ ДЛЯ ПРОСМОТРА И ЭКСПОРТА PR
    # ================================================

    def get_pr_table(self, min_pr: float = 0.0, min_deals: int = 5) -> pl.DataFrame:
        """
        Возвращает Polars DataFrame с актуальной таблицей PR.
        Фильтрует монеты с низким PR и малым кол-вом сделок.
        Сортирует по убыванию pr_value.
        """
        query = """
            SELECT 
                symbol,
                tf,
                period,
                anomaly_type,
                direction,
                pr_value,
                winrate,
                total_deals,
                tp_hits,
                sl_hits,
                last_update
            FROM pr_snapshots
            WHERE pr_value >= ?
              AND total_deals >= ?
            ORDER BY pr_value DESC
        """
        params = (min_pr, min_deals)

        if self.backend == "duckdb":
            result = self.con.execute(query, params).pl()
        else:
            import pandas as pd
            df_pd = pd.read_sql_query(query, self.con, params=params)
            result = pl.from_pandas(df_pd)

        logger.debug("Таблица PR запрошена", 
                     rows=len(result), 
                     min_pr=min_pr, 
                     min_deals=min_deals)
        return result

    def export_pr_to_csv(self, df: pl.DataFrame, prefix: str = "pr_table") -> str:
        """
        Экспортирует DataFrame в CSV с временной меткой.
        Возвращает путь к файлу.
        """
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{timestamp}.csv"
        filepath = filename  # можно сделать ./logs/ или другую папку

        df.write_csv(filepath)
        logger.info("PR таблица экспортирована в CSV", 
                    filepath=filepath, 
                    rows=len(df))
        return filepath

    def close(self):
        self.con.close()
        logger.info("Соединение с БД закрыто")