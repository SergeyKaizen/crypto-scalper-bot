"""
src/data/storage.py

Хранилище всех свечных данных + метаданных монет.
Используем DuckDB как основное решение (быстрее SQLite на больших объёмах, columnar, легко аналитика).
Fallback — SQLite, если DuckDB по каким-то причинам недоступен.

Основные таблицы:
- candles_{timeframe}     — отдельная таблица на каждый таймфрейм (1m, 3m, 5m, 10m, 15m)
- symbols_meta            — метаданные монет (listed_at, delisted, last_update, etc.)
- whitelist               — текущий список торгуемых монет + их лучшие настройки (tf, window, anomaly_type, direction)

Ключевые изменения для delisted:
- remove_delisted(symbols: list[str]) теперь автоматически:
  - Удаляет свечи из всех candles_*
  - Помечает delisted=True в symbols_meta
  - Удаляет из whitelist
  - Удаляет связанные модели (файлы models/{symbol}_*.pt, если существуют — по ТЗ модели общие, но на случай per-монета)

Методы:
- save_candles(symbol, timeframe, df, append=True)
- get_last_timestamp(symbol, timeframe) → int или None
- get_candles(symbol, timeframe, start_ts=None, end_ts=None) → pd.DataFrame
- remove_delisted(symbols_to_remove: list[str])
- update_symbol_meta(symbol, listed_at=None, delisted=False)
- get_all_symbols() → list[str]
- get_whitelisted_symbols() → list[str]
- add_to_whitelist(symbol, tf, window, anomaly_type, direction, pr_value)
- remove_from_whitelist(symbols: list[str])
- clear_whitelist()

"""

import os
import logging
from datetime import datetime
import pandas as pd
try:
    import duckdb
    ENGINE = "duckdb"
except ImportError:
    import sqlite3
    ENGINE = "sqlite"

from src.core.config import load_config
from src.utils.logger import setup_logger

logger = setup_logger("storage", logging.INFO)

class Storage:
    def __init__(self):
        config = load_config()
        self.data_dir = config["paths"]["data_dir"]
        self.models_dir = config["paths"].get("models_dir", os.path.join(self.data_dir, "models"))  # Директория для моделей
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)

        if ENGINE == "duckdb":
            self.db_path = os.path.join(self.data_dir, "candles.duckdb")
            self.con = duckdb.connect(self.db_path)
            logger.info("Используется DuckDB как хранилище")
        else:
            self.db_path = os.path.join(self.data_dir, "candles.sqlite")
            self.con = sqlite3.connect(self.db_path)
            logger.warning("DuckDB не найден → используется SQLite (медленнее на больших данных)")

        self._create_tables()

    def _create_tables(self):
        """Создаёт необходимые таблицы, если их нет"""
        timeframes = load_config()["timeframes"]  # ['1m', '3m', '5m', '10m', '15m']

        for tf in timeframes:
            table_name = f"candles_{tf.replace('m', '')}m"
            if ENGINE == "duckdb":
                self.con.execute(f"""
                    CREATE TABLE IF NOT EXISTS {table_name} (
                        symbol      VARCHAR,
                        timestamp   TIMESTAMP,
                        open        DOUBLE,
                        high        DOUBLE,
                        low         DOUBLE,
                        close       DOUBLE,
                        volume      DOUBLE,
                        bid         DOUBLE,
                        ask         DOUBLE,
                        PRIMARY KEY (symbol, timestamp)
                    )
                """)
            else:
                # SQLite
                self.con.execute(f"""
                    CREATE TABLE IF NOT EXISTS {table_name} (
                        symbol      TEXT,
                        timestamp   TEXT,
                        open        REAL,
                        high        REAL,
                        low         REAL,
                        close       REAL,
                        volume      REAL,
                        bid         REAL,
                        ask         REAL,
                        PRIMARY KEY (symbol, timestamp)
                    )
                """)

        # Таблица метаданных символов
        if ENGINE == "duckdb":
            self.con.execute("""
                CREATE TABLE IF NOT EXISTS symbols_meta (
                    symbol      VARCHAR PRIMARY KEY,
                    listed_at   TIMESTAMP,
                    delisted    BOOLEAN DEFAULT FALSE,
                    last_update TIMESTAMP
                )
            """)
            self.con.execute("""
                CREATE TABLE IF NOT EXISTS whitelist (
                    symbol          VARCHAR PRIMARY KEY,
                    tf              VARCHAR,
                    window          INTEGER,
                    anomaly_type    VARCHAR,    -- 'C', 'V', 'CV'
                    direction       VARCHAR,    -- 'L', 'S', 'LS'
                    pr_value        DOUBLE,
                    updated_at      TIMESTAMP
                )
            """)
        else:
            # SQLite аналоги (без BOOLEAN, используем INTEGER 0/1)
            self.con.execute("""
                CREATE TABLE IF NOT EXISTS symbols_meta (
                    symbol      TEXT PRIMARY KEY,
                    listed_at   TEXT,
                    delisted    INTEGER DEFAULT 0,
                    last_update TEXT
                )
            """)
            self.con.execute("""
                CREATE TABLE IF NOT EXISTS whitelist (
                    symbol          TEXT PRIMARY KEY,
                    tf              TEXT,
                    window          INTEGER,
                    anomaly_type    TEXT,
                    direction       TEXT,
                    pr_value        REAL,
                    updated_at      TEXT
                )
            """)

        self.con.commit()

    def save_candles(self, symbol: str, timeframe: str, df: pd.DataFrame, append: bool = True):
        """Сохраняет или добавляет свечи в соответствующую таблицу"""
        if df.empty:
            return

        table_name = f"candles_{timeframe.replace('m', '')}m"

        # DuckDB любит uppercase колонки → приводим
        df.columns = [c.upper() for c in df.columns]

        if ENGINE == "duckdb":
            if append:
                self.con.execute(f"""
                    INSERT OR REPLACE INTO {table_name}
                    SELECT * FROM df
                """, {"df": df})
            else:
                self.con.execute(f"DELETE FROM {table_name} WHERE symbol = ?", [symbol])
                self.con.execute(f"INSERT INTO {table_name} SELECT * FROM df", {"df": df})
        else:
            # SQLite — чуть сложнее
            conn = sqlite3.connect(self.db_path)
            if not append:
                conn.execute(f"DELETE FROM {table_name} WHERE symbol = ?", (symbol,))
            df.to_sql(table_name, conn, if_exists="append", index=False)
            conn.close()

        self.con.commit()
        logger.debug(f"Сохранено {len(df)} свечей {symbol} {timeframe}")

    def get_last_timestamp(self, symbol: str, timeframe: str) -> int | None:
        """Возвращает timestamp последней свечи (в миллисекундах) или None"""
        table_name = f"candles_{timeframe.replace('m', '')}m"
        if ENGINE == "duckdb":
            row = self.con.execute(f"""
                SELECT MAX(timestamp) FROM {table_name} WHERE symbol = ?
            """, [symbol]).fetchone()
        else:
            cur = self.con.cursor()
            cur.execute(f"SELECT MAX(timestamp) FROM {table_name} WHERE symbol = ?", (symbol,))
            row = cur.fetchone()
            cur.close()

        if row and row[0] is not None:
            if isinstance(row[0], str):  # SQLite
                dt = datetime.fromisoformat(row[0])
            else:
                dt = row[0]
            return int(dt.timestamp() * 1000)
        return None

    def get_all_symbols(self) -> list[str]:
        """Возвращает список всех уникальных символов в candles_* таблицах"""
        timeframes = load_config()["timeframes"]
        symbols = set()
        for tf in timeframes:
            table = f"candles_{tf.replace('m', '')}m"
            if ENGINE == "duckdb":
                rows = self.con.execute(f"SELECT DISTINCT symbol FROM {table}").fetchall()
            else:
                cur = self.con.cursor()
                cur.execute(f"SELECT DISTINCT symbol FROM {table}")
                rows = cur.fetchall()
                cur.close()
            symbols.update([r[0] for r in rows if r[0]])
        return list(symbols)

    def remove_delisted(self, symbols: list[str]):
        """Автоматически удаляет все данные по delisted символам:
        - Свечи из всех candles_*
        - Записи из whitelist
        - Помечает delisted=True в symbols_meta
        - Удаляет связанные модели файлы (models/{symbol}_*.pt, если существуют)
        """
        if not symbols:
            return

        timeframes = load_config()["timeframes"]
        for symbol in symbols:
            # Удаление свечей
            for tf in timeframes:
                table = f"candles_{tf.replace('m', '')}m"
                if ENGINE == "duckdb":
                    self.con.execute(f"DELETE FROM {table} WHERE symbol = ?", [symbol])
                else:
                    self.con.execute(f"DELETE FROM {table} WHERE symbol = ?", (symbol,))

            # Помечаем delisted в meta
            now = datetime.utcnow()
            if ENGINE == "duckdb":
                self.con.execute("""
                    INSERT OR REPLACE INTO symbols_meta (symbol, delisted, last_update)
                    VALUES (?, TRUE, ?)
                """, [symbol, now])
            else:
                self.con.execute("""
                    INSERT OR REPLACE INTO symbols_meta (symbol, delisted, last_update)
                    VALUES (?, 1, ?)
                """, (symbol, now.isoformat()))

            # Удаление из whitelist
            if ENGINE == "duckdb":
                self.con.execute("DELETE FROM whitelist WHERE symbol = ?", [symbol])
            else:
                self.con.execute("DELETE FROM whitelist WHERE symbol = ?", (symbol,))

            # Удаление моделей (файлов, если per-монета; по ТЗ модели общие, но на всякий)
            for file in os.listdir(self.models_dir):
                if file.startswith(f"{symbol}_") and file.endswith('.pt'):
                    os.remove(os.path.join(self.models_dir, file))
                    logger.info(f"Удалена модель {file} для {symbol}")

        self.con.commit()
        logger.info(f"Удалены данные по {len(symbols)} delisted монетам (включая модели и whitelist)")

    def update_symbol_meta(self, symbol: str, listed_at: datetime = None, delisted: bool = False):
        """Обновляет метаданные символа"""
        now = datetime.utcnow()
        if ENGINE == "duckdb":
            self.con.execute("""
                INSERT OR REPLACE INTO symbols_meta (symbol, listed_at, delisted, last_update)
                VALUES (?, ?, ?, ?)
            """, [symbol, listed_at, delisted, now])
        else:
            self.con.execute("""
                INSERT OR REPLACE INTO symbols_meta (symbol, listed_at, delisted, last_update)
                VALUES (?, ?, ?, ?)
            """, (symbol, listed_at.isoformat() if listed_at else None, 1 if delisted else 0, now.isoformat()))
        self.con.commit()

    def get_whitelisted_symbols(self) -> list[str]:
        """Возвращает список торгуемых монет из whitelist"""
        if ENGINE == "duckdb":
            rows = self.con.execute("SELECT symbol FROM whitelist").fetchall()
        else:
            cur = self.con.cursor()
            cur.execute("SELECT symbol FROM whitelist")
            rows = cur.fetchall()
            cur.close()
        return [r[0] for r in rows]

    def add_to_whitelist(self, symbol: str, tf: str, window: int, anomaly_type: str, direction: str, pr_value: float):
        """Добавляет/обновляет запись в whitelist"""
        now = datetime.utcnow()
        if ENGINE == "duckdb":
            self.con.execute("""
                INSERT OR REPLACE INTO whitelist (symbol, tf, window, anomaly_type, direction, pr_value, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, [symbol, tf, window, anomaly_type, direction, pr_value, now])
        else:
            self.con.execute("""
                INSERT OR REPLACE INTO whitelist (symbol, tf, window, anomaly_type, direction, pr_value, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (symbol, tf, window, anomaly_type, direction, pr_value, now.isoformat()))
        self.con.commit()

    def clear_whitelist(self):
        """Очищает whitelist"""
        if ENGINE == "duckdb":
            self.con.execute("DELETE FROM whitelist")
        else:
            self.con.execute("DELETE FROM whitelist")
        self.con.commit()

    # ======================== ФИКСЫ ФАЗЫ 4 ========================
    def get_candles(self, symbol: str, timeframe: str, start_ts: int = None, end_ts: int = None) -> pd.DataFrame:
        """FIX Фаза 4: теперь реальный запрос (раньше пустой стаб)"""
        table_name = f"candles_{timeframe.replace('m', '')}m"
        query = f"SELECT * FROM {table_name} WHERE symbol = ?"
        params = [symbol]
        if start_ts:
            query += " AND timestamp >= ?"
            params.append(start_ts)
        if end_ts:
            query += " AND timestamp <= ?"
            params.append(end_ts)
        if ENGINE == "duckdb":
            df = self.con.execute(query, params).fetchdf()
        else:
            df = pd.read_sql_query(query, sqlite3.connect(self.db_path), params=params)
        return df

    def get_whitelist_settings(self, symbol: str) -> dict:
        """FIX Фаза 4: теперь реальный запрос (раньше пустой стаб)"""
        if ENGINE == "duckdb":
            row = self.con.execute("SELECT * FROM whitelist WHERE symbol = ?", [symbol]).fetchone()
        else:
            cur = self.con.cursor()
            cur.execute("SELECT * FROM whitelist WHERE symbol = ?", (symbol,))
            row = cur.fetchone()
            cur.close()
        if row:
            return {
                "tf": row[1],
                "window": row[2],
                "anomaly_type": row[3],
                "direction": row[4],
                "pr_value": row[5]
            }
        return {}

    def get_last_candle(self, symbol: str, timeframe: str) -> dict:
        """FIX Фаза 4: теперь реальный запрос (раньше пустой стаб)"""
        table_name = f"candles_{timeframe.replace('m', '')}m"
        if ENGINE == "duckdb":
            row = self.con.execute(f"SELECT * FROM {table_name} WHERE symbol = ? ORDER BY timestamp DESC LIMIT 1", [symbol]).fetchone()
        else:
            cur = self.con.cursor()
            cur.execute(f"SELECT * FROM {table_name} WHERE symbol = ? ORDER BY timestamp DESC LIMIT 1", (symbol,))
            row = cur.fetchone()
            cur.close()
        if row:
            return dict(zip(["symbol", "timestamp", "open", "high", "low", "close", "volume", "bid", "ask"], row))
        return {}

    def close(self):
        if hasattr(self, "con"):
            self.con.close()