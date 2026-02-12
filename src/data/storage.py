# src/data/storage.py
"""
Абстракция над базой данных.
Скрывает детали реализации (SQLite / DuckDB / TimescaleDB)
Предоставляет единый интерфейс для всех модулей.

Поддерживаемые бэкенды:
- SQLite (телефон — лёгкий, файловая БД)
- DuckDB (Colab — быстрый columnar, parquet-поддержка)
- TimescaleDB (сервер — time-series PostgreSQL, компрессия, aggregates)

Основные операции:
- save_ohlcv(symbol, timeframe, candles)
- get_last_timestamp(symbol, timeframe)
- get_candles(symbol, timeframe, since, limit)
- add_coins(new_coins)
- remove_coins(removed_coins)
- get_current_coins()
- cleanup_old_data(days=730) — очистка старше 2 лет
"""

import logging
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta

import polars as pl

from src.core.config import load_config

logger = logging.getLogger(__name__)

class Storage:
    """Абстракция БД — единый интерфейс для всех операций"""

    def __init__(self, config: Dict):
        self.config = config
        self.backend = config["storage"]["backend"]  # "sqlite" / "duckdb" / "timescaledb"

        if self.backend == "sqlite":
            self._init_sqlite()
        elif self.backend == "duckdb":
            self._init_duckdb()
        elif self.backend == "timescaledb":
            self._init_timescaledb()
        else:
            raise ValueError(f"Unknown storage backend: {self.backend}")

        logger.info("Storage initialized with backend: %s", self.backend)

    def _init_sqlite(self):
        """Инициализация SQLite (для телефона)"""
        import sqlite3
        self.conn = sqlite3.connect("data/bot.db")
        self.cursor = self.conn.cursor()
        self._create_tables_sqlite()

    def _init_duckdb(self):
        """Инициализация DuckDB (для Colab)"""
        import duckdb
        self.db = duckdb.connect("data/bot.duckdb")
        self._create_tables_duckdb()

    def _init_timescaledb(self):
        """Инициализация TimescaleDB (для сервера)"""
        import psycopg2
        self.conn = psycopg2.connect(
            dbname="bot_db",
            user="postgres",
            password="password",
            host="localhost",
            port=5432
        )
        self.cursor = self.conn.cursor()
        self._create_tables_timescaledb()

    def _create_tables_sqlite(self):
        """Создание таблиц в SQLite"""
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS candles (
                symbol TEXT,
                timeframe TEXT,
                timestamp INTEGER,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                bid_volume REAL,
                ask_volume REAL,
                PRIMARY KEY (symbol, timeframe, timestamp)
            )
        """)
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS coins (
                symbol TEXT PRIMARY KEY,
                added_at INTEGER,
                last_updated INTEGER
            )
        """)
        self.conn.commit()

    # Аналогично для DuckDB и TimescaleDB (создание таблиц + hypertable для Timescale)
    # ... (реализация сокращена для примера, но в реальном коде — все 3 метода)

    async def save_ohlcv(self, symbol: str, timeframe: str, ohlcv: List[List[float]]):
        """Сохранение списка свечей (OHLCV + bid/ask)"""
        if self.backend == "sqlite":
            data = [
                (symbol, timeframe, int(c[0]), c[1], c[2], c[3], c[4], c[5], c[6], c[7])
                for c in ohlcv
            ]
            self.cursor.executemany("""
                INSERT OR REPLACE INTO candles 
                (symbol, timeframe, timestamp, open, high, low, close, volume, bid_volume, ask_volume)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, data)
            self.conn.commit()

        elif self.backend == "duckdb":
            df = pl.DataFrame({
                "symbol": [symbol] * len(ohlcv),
                "timeframe": [timeframe] * len(ohlcv),
                "timestamp": [int(c[0]) for c in ohlcv],
                "open": [c[1] for c in ohlcv],
                "high": [c[2] for c in ohlcv],
                "low": [c[3] for c in ohlcv],
                "close": [c[4] for c in ohlcv],
                "volume": [c[5] for c in ohlcv],
                "bid_volume": [c[6] for c in ohlcv],
                "ask_volume": [c[7] for c in ohlcv],
            })
            self.db.execute("INSERT INTO candles SELECT * FROM df ON CONFLICT DO NOTHING")

        logger.info("Saved %d candles for %s (%s)", len(ohlcv), symbol, timeframe)

    async def get_last_timestamp(self, symbol: str, timeframe: str) -> Optional[int]:
        """Последний timestamp для монеты и TF"""
        if self.backend == "sqlite":
            self.cursor.execute("""
                SELECT MAX(timestamp) FROM candles 
                WHERE symbol = ? AND timeframe = ?
            """, (symbol, timeframe))
            result = self.cursor.fetchone()
            return result[0] if result and result[0] else None

        # Аналогично для DuckDB и TimescaleDB

    async def get_candles(self, symbol: str, timeframe: str, since: int, limit: int = 1000) -> List[Dict]:
        """Получение последних свечей (для live_loop)"""
        # Реализация зависит от backend (SQL-запросы или Polars/DuckDB query)
        pass  # Полная реализация в реальном коде

    async def add_coins(self, symbols: List[str]):
        """Добавление новых монет в БД"""
        now = int(datetime.now().timestamp())
        if self.backend == "sqlite":
            data = [(s, now, now) for s in symbols]
            self.cursor.executemany("INSERT OR IGNORE INTO coins VALUES (?, ?, ?)", data)
            self.conn.commit()

    async def remove_coins(self, symbols: List[str]):
        """Удаление delisted монет"""
        if self.backend == "sqlite":
            placeholders = ','.join('?' for _ in symbols)
            self.cursor.execute(f"DELETE FROM coins WHERE symbol IN ({placeholders})", symbols)
            self.cursor.execute(f"DELETE FROM candles WHERE symbol IN ({placeholders})", symbols)
            self.conn.commit()

    async def get_current_coins(self) -> List[str]:
        """Список всех монет в модуле"""
        if self.backend == "sqlite":
            self.cursor.execute("SELECT symbol FROM coins")
            return [row[0] for row in self.cursor.fetchall()]

    async def cleanup_old_data(self, days: int = 730):
        """Очистка данных старше N дней (для экономии места)"""
        cutoff = int((datetime.now() - timedelta(days=days)).timestamp())
        if self.backend == "sqlite":
            self.cursor.execute("DELETE FROM candles WHERE timestamp < ?", (cutoff,))
            self.conn.commit()
            logger.info("Cleaned old data older than %d days", days)

    async def close(self):
        """Закрытие соединения"""
        if self.backend == "sqlite":
            self.conn.close()
        # Аналогично для других backend
        logger.info("Storage closed")