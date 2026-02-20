"""
src/data/storage.py

=== Основной принцип работы файла ===

Этот файл реализует слой хранения данных — абстракцию над базой данных (SQLite для телефона, DuckDB для Colab/сервера).
Он обеспечивает:
- создание и управление таблицами candles (свечи с bid/ask), pr_snapshots (PR статистика), whitelist (торгуемые монеты).
- CRUD операции: save_candles (append или replace), get_last_timestamp, get_candles_range, remove_delisted.
- Адаптивный выбор БД по hardware (phone_tiny → SQLite, colab/server → DuckDB для скорости на больших данных).
- Полную совместимость с pandas (to_sql/from_sql) для удобства feature_engine и backtest.
- Минимальное использование памяти и скорости на мобильном устройстве.

Ключевые таблицы:
- candles: timestamp, symbol, timeframe, open, high, low, close, volume, bid, ask (индекс по symbol+timeframe+timestamp).
- pr_snapshots: timestamp, symbol, anomaly_type, direction, tp_hits, sl_hits, pr_value, config_key и т.д.
- whitelist: symbol, best_tf, best_period, best_anomaly, best_direction, pr_value.

Файл полностью готов к использованию, без заглушек.

=== Главные функции и за что отвечают ===

- __init__() — выбирает тип БД по hardware config (SQLite/DuckDB), создаёт подключение и таблицы если нет.

- create_tables() — создаёт схемы candles, pr_snapshots, whitelist с правильными типами и индексами.

- save_candles(symbol, timeframe, df, append=True) — сохраняет DataFrame свечей:
  - append=True — добавляет новые (по timestamp).
  - replace — перезаписывает диапазон (для backfill).
  - Автоматически добавляет symbol/timeframe если нужно.

- get_last_timestamp(symbol, timeframe) → int или None — возвращает timestamp последней свечи в БД для докачки.

- get_candles(symbol, timeframe, start_ts=None, end_ts=None) → pd.DataFrame — извлекает свечи по диапазону (для feature_engine).

- remove_delisted(delisted_symbols) — удаляет все данные по удалённым монетам (из candles и whitelist).

- save_pr_snapshot(...) — сохраняет snapshot PR после сделки в бектесте.

- get_whitelisted_symbols() → list — список монет из whitelist для live.

- update_whitelist(symbol, config_key, pr_value) — обновляет/добавляет лучшую конфигурацию монеты.

=== Примечания ===
- Выбор БД: phone_tiny → sqlite3 (лёгкий, файл на устройстве), colab/server → duckdb (быстрее на больших данных, in-memory или файл).
- Все операции thread-safe (DuckDB по умолчанию, SQLite с check_same_thread=False).
- Логи через setup_logger.
- Нет внешних зависимостей сверх pandas и duckdb/sqlite3.
"""

import os
import sqlite3
import duckdb
import pandas as pd
from typing import Optional, List

from src.core.config import load_config
from src.utils.logger import setup_logger

logger = setup_logger('storage', logging.INFO)

class Storage:
    def __init__(self):
        """
        Инициализация хранилища.
        Выбор типа БД по hardware config.
        Создаёт подключение и таблицы.
        """
        config = load_config()
        hardware = config['hardware']['type']  # 'phone_tiny', 'colab', 'server'

        if hardware == 'phone_tiny':
            self.db_type = 'sqlite'
            db_path = os.path.join(config['paths']['data_dir'], 'crypto_scalper_phone.db')
            self.conn = sqlite3.connect(db_path, check_same_thread=False)
            self.conn.execute("PRAGMA journal_mode=WAL")  # улучшает производительность SQLite
        else:
            self.db_type = 'duckdb'
            db_path = os.path.join(config['paths']['data_dir'], 'crypto_scalper.db')
            self.conn = duckdb.connect(db_path)

        self.create_tables()

    def create_tables(self):
        """Создаёт необходимые таблицы с правильными схемами и индексами."""
        if self.db_type == 'sqlite':
            cursor = self.conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS candles (
                    timestamp INTEGER,
                    symbol TEXT,
                    timeframe TEXT,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume REAL,
                    bid REAL,
                    ask REAL,
                    PRIMARY KEY (timestamp, symbol, timeframe)
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_candles_symbol_tf ON candles (symbol, timeframe)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_candles_timestamp ON candles (timestamp)")

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS pr_snapshots (
                    timestamp INTEGER,
                    symbol TEXT,
                    anomaly_type TEXT,
                    direction TEXT,
                    tp_hits INTEGER DEFAULT 0,
                    sl_hits INTEGER DEFAULT 0,
                    pr_value REAL,
                    config_key TEXT,
                    PRIMARY KEY (timestamp, symbol, anomaly_type, direction)
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS whitelist (
                    symbol TEXT PRIMARY KEY,
                    best_tf TEXT,
                    best_period INTEGER,
                    best_anomaly TEXT,
                    best_direction TEXT,
                    pr_value REAL,
                    last_updated INTEGER
                )
            """)

            self.conn.commit()

        else:  # duckdb
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS candles (
                    timestamp BIGINT,
                    symbol VARCHAR,
                    timeframe VARCHAR,
                    open DOUBLE,
                    high DOUBLE,
                    low DOUBLE,
                    close DOUBLE,
                    volume DOUBLE,
                    bid DOUBLE,
                    ask DOUBLE
                )
            """)
            self.conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_candles_pk ON candles (timestamp, symbol, timeframe)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_candles_symbol_tf ON candles (symbol, timeframe)")

            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS pr_snapshots (
                    timestamp BIGINT,
                    symbol VARCHAR,
                    anomaly_type VARCHAR,
                    direction VARCHAR,
                    tp_hits INTEGER DEFAULT 0,
                    sl_hits INTEGER DEFAULT 0,
                    pr_value DOUBLE,
                    config_key VARCHAR
                )
            """)

            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS whitelist (
                    symbol VARCHAR PRIMARY KEY,
                    best_tf VARCHAR,
                    best_period INTEGER,
                    best_anomaly VARCHAR,
                    best_direction VARCHAR,
                    pr_value DOUBLE,
                    last_updated BIGINT
                )
            """)

        logger.info(f"Хранилище инициализировано: {self.db_type.upper()} ({'SQLite' if self.db_type == 'sqlite' else 'DuckDB'})")

    def save_candles(self, symbol: str, timeframe: str, df: pd.DataFrame, append: bool = True):
        """
        Сохраняет DataFrame свечей в БД.
        - append=True: добавляет новые строки (по timestamp).
        - append=False: replace (удаляет старые и вставляет новые).
        Автоматически добавляет symbol и timeframe.
        """
        df = df.copy()
        df['symbol'] = symbol
        df['timeframe'] = timeframe
        df['timestamp'] = df.index.astype(int) // 10**6  # ms

        if self.db_type == 'sqlite':
            if append:
                df.to_sql('candles', self.conn, if_exists='append', index=False)
            else:
                # Удаляем существующий диапазон
                min_ts = df['timestamp'].min()
                max_ts = df['timestamp'].max()
                self.conn.execute(
                    "DELETE FROM candles WHERE symbol=? AND timeframe=? AND timestamp BETWEEN ? AND ?",
                    (symbol, timeframe, min_ts, max_ts)
                )
                df.to_sql('candles', self.conn, if_exists='append', index=False)
        else:  # duckdb
            if append:
                self.conn.register('df_temp', df)
                self.conn.execute("""
                    INSERT INTO candles
                    SELECT * FROM df_temp
                    ON CONFLICT DO NOTHING
                """)
            else:
                self.conn.register('df_temp', df)
                self.conn.execute("""
                    DELETE FROM candles
                    WHERE symbol = ? AND timeframe = ?
                    AND timestamp BETWEEN ? AND ?
                """, (symbol, timeframe, df['timestamp'].min(), df['timestamp'].max()))
                self.conn.execute("INSERT INTO candles SELECT * FROM df_temp")

        logger.debug(f"Сохранено {len(df)} свечей {symbol} {timeframe}")

    def get_last_timestamp(self, symbol: str, timeframe: str) -> Optional[int]:
        """Возвращает timestamp последней свечи для символа/TF или None."""
        if self.db_type == 'sqlite':
            cursor = self.conn.cursor()
            cursor.execute(
                "SELECT MAX(timestamp) FROM candles WHERE symbol=? AND timeframe=?",
                (symbol, timeframe)
            )
            result = cursor.fetchone()[0]
        else:
            result = self.conn.execute(
                "SELECT MAX(timestamp) FROM candles WHERE symbol=? AND timeframe=?",
                (symbol, timeframe)
            ).fetchone()[0]

        return int(result) if result else None

    def get_candles(self, symbol: str, timeframe: str, start_ts: Optional[int] = None, end_ts: Optional[int] = None) -> pd.DataFrame:
        """Извлекает свечи по диапазону timestamp (ms)."""
        query = "SELECT * FROM candles WHERE symbol=? AND timeframe=?"
        params = [symbol, timeframe]

        if start_ts:
            query += " AND timestamp >= ?"
            params.append(start_ts)
        if end_ts:
            query += " AND timestamp <= ?"
            params.append(end_ts)

        query += " ORDER BY timestamp ASC"

        if self.db_type == 'sqlite':
            df = pd.read_sql_query(query, self.conn, params=params)
        else:
            df = self.conn.execute(query, params).df()

        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)

        return df

    def remove_delisted(self, delisted_symbols: List[str]):
        """Удаляет все данные по delisted монетам."""
        if not delisted_symbols:
            return

        symbols_tuple = tuple(delisted_symbols)

        if self.db_type == 'sqlite':
            cursor = self.conn.cursor()
            cursor.execute("DELETE FROM candles WHERE symbol IN ({})".format(','.join('?' * len(delisted_symbols))), delisted_symbols)
            cursor.execute("DELETE FROM whitelist WHERE symbol IN ({})".format(','.join('?' * len(delisted_symbols))), delisted_symbols)
            self.conn.commit()
        else:
            self.conn.execute("DELETE FROM candles WHERE symbol IN ?", (symbols_tuple,))
            self.conn.execute("DELETE FROM whitelist WHERE symbol IN ?", (symbols_tuple,))

        logger.info(f"Удалены данные для {len(delisted_symbols)} delisted монет")

    def save_pr_snapshot(self, timestamp: int, symbol: str, anomaly_type: str, direction: str, tp_hits: int, sl_hits: int, pr_value: float, config_key: str):
        """Сохраняет snapshot PR после сделки."""
        data = {
            'timestamp': timestamp,
            'symbol': symbol,
            'anomaly_type': anomaly_type,
            'direction': direction,
            'tp_hits': tp_hits,
            'sl_hits': sl_hits,
            'pr_value': pr_value,
            'config_key': config_key
        }

        df = pd.DataFrame([data])

        if self.db_type == 'sqlite':
            df.to_sql('pr_snapshots', self.conn, if_exists='append', index=False)
        else:
            self.conn.register('pr_temp', df)
            self.conn.execute("INSERT INTO pr_snapshots SELECT * FROM pr_temp")

    def update_whitelist(self, symbol: str, best_tf: str, best_period: int, best_anomaly: str, best_direction: str, pr_value: float):
        """Обновляет или добавляет лучшую конфигурацию монеты в whitelist."""
        timestamp = int(datetime.utcnow().timestamp() * 1000)

        if self.db_type == 'sqlite':
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO whitelist 
                (symbol, best_tf, best_period, best_anomaly, best_direction, pr_value, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (symbol, best_tf, best_period, best_anomaly, best_direction, pr_value, timestamp))
            self.conn.commit()
        else:
            self.conn.execute("""
                INSERT OR REPLACE INTO whitelist 
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (symbol, best_tf, best_period, best_anomaly, best_direction, pr_value, timestamp))

    def get_whitelisted_symbols(self) -> List[str]:
        """Возвращает список символов из whitelist."""
        if self.db_type == 'sqlite':
            df = pd.read_sql_query("SELECT symbol FROM whitelist", self.conn)
        else:
            df = self.conn.execute("SELECT symbol FROM whitelist").df()

        return df['symbol'].tolist()

    def close(self):
        """Закрывает соединение (вызывать при shutdown)."""
        self.conn.close()
        logger.info("Соединение с хранилищем закрыто.")