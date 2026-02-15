# src/data/resampler.py
"""
Модуль инкрементального ресэмплинга 1-минутных свечей в старшие таймфреймы.

Ключевые требования из ТЗ:
- Обучение и предсказание на нескольких TF: 1m, 3m, 5m, 10m, 15m
- Модель видит 4 окна (24, 50, 74, 100 свечей) на каждом TF
- Данные скачиваются и хранятся в DuckDB / SQLite, но для скорости расчётов
  последние N свечей держим в памяти (особенно критично на телефоне)

Особенности реализации:
- Кэш в памяти ограничен: 500 свечей на телефоне (tiny), 2000–5000 на сервере/colab
- При добавлении новой 1m свечи — мгновенно обновляются все старшие TF
- Старые свечи автоматически вытесняются из кэша (FIFO)
- Всё на Polars — очень быстро даже на слабом железе
"""

from collections import defaultdict, deque
import polars as pl
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from src.core.config import Config  # предполагаем, что конфиг загружается через этот класс
from src.utils.logger import get_logger

logger = get_logger(__name__)

class Resampler:
    """
    Инкрементальный ресэмплер.
    Один экземпляр на монету (или на весь бот — зависит от реализации).
    """

    def __init__(self, config: dict):
        self.config = config
        
        # Список поддерживаемых таймфреймов (из ТЗ)
        self.timeframes = ["1m", "3m", "5m", "10m", "15m"]
        
        # Максимальное количество свечей в памяти для каждого TF
        hardware = config["hardware_mode"]  # "phone_tiny", "colab", "server"
        if hardware == "phone_tiny":
            self.max_cache_per_tf = 500   # ~8–12 часов на 1m, достаточно для всех окон
        elif hardware == "server":
            self.max_cache_per_tf = 5000  # ~3–4 дня на 1m — комфортно
        else:  # colab
            self.max_cache_per_tf = 2000
        
        # Кэш: {tf: deque[pl.DataFrame]} — последние свечи в памяти
        self.cache: Dict[str, deque[pl.DataFrame]] = {
            tf: deque(maxlen=self.max_cache_per_tf) for tf in self.timeframes
        }
        
        # Последний timestamp для каждого TF (чтобы знать, что уже обработано)
        self.last_ts: Dict[str, int] = {tf: 0 for tf in self.timeframes}

    def add_1m_candle(self, candle: dict) -> None:
        """
        Добавляем новую 1-минутную свечу (приходит из websocket или polling).
        
        candle = {
            "open_time": int (ms),
            "open": float,
            "high": float,
            "low": float,
            "close": float,
            "volume": float,
            "buy_volume": float,   # если есть, иначе 0
            ...
        }
        
        Автоматически обновляет все старшие TF в кэше.
        """
        df_1m = pl.DataFrame([candle]).with_columns(
            pl.col("open_time").cast(pl.Int64).alias("timestamp")
        ).sort("timestamp")
        
        # Добавляем в кэш 1m
        self._append_to_cache("1m", df_1m)
        
        # Обновляем старшие TF
        for tf in self.timeframes[1:]:  # пропускаем 1m
            self._resample_and_append(tf, df_1m)

    def get_window(self, tf: str, window_size: int) -> Optional[pl.DataFrame]:
        """
        Возвращает последние window_size свечей для указанного TF.
        Если в кэше недостаточно — возвращает None (нужно докачать из БД).
        
        Используется в feature_engine и inference.
        """
        if tf not in self.cache or len(self.cache[tf]) < window_size:
            return None
        
        # Собираем последние window_size свечей
        recent = list(self.cache[tf])[-window_size:]
        return pl.concat(recent)

    def get_all_windows(self, tf: str) -> Dict[int, pl.DataFrame]:
        """
        Удобный метод для модели: возвращает словарь всех нужных окон.
        {24: df_24, 50: df_50, 74: df_74, 100: df_100}
        """
        windows = {}
        for size in [24, 50, 74, 100]:
            df = self.get_window(tf, size)
            if df is not None:
                windows[size] = df
        return windows

    def _append_to_cache(self, tf: str, new_df: pl.DataFrame):
        """Добавляет новые свечи в deque и следит за лимитом."""
        if new_df.is_empty():
            return
        
        current_last_ts = self.last_ts[tf]
        new_df = new_df.filter(pl.col("timestamp") > current_last_ts)
        
        if new_df.is_empty():
            return
        
        self.cache[tf].append(new_df)
        self.last_ts[tf] = new_df["timestamp"].max()
        
        # Лог при очистке кэша (на телефоне полезно видеть)
        if len(self.cache[tf]) == self.max_cache_per_tf:
            logger.debug(f"[{tf}] Кэш достиг лимита {self.max_cache_per_tf} свечей — старые вытеснены")

    def _resample_and_append(self, target_tf: str, new_1m: pl.DataFrame):
        """
        Инкрементальный ресэмплинг 1m → target_tf (3m,5m,10m,15m).
        Только новые свечи обрабатываются.
        """
        if new_1m.is_empty():
            return
        
        # Определяем правило группировки
        rule_map = {
            "3m": "3min",
            "5m": "5min",
            "10m": "10min",
            "15m": "15min"
        }
        rule = rule_map.get(target_tf)
        if not rule:
            return
        
        # Ресэмплим только новые 1m свечи
        resampled = new_1m.group_by_dynamic(
            "timestamp",
            every=rule,
            closed="right",
            by_group_by=False
        ).agg(
            open=pl.col("open").first(),
            high=pl.col("high").max(),
            low=pl.col("low").min(),
            close=pl.col("close").last(),
            volume=pl.col("volume").sum(),
            buy_volume=pl.col("buy_volume").sum()
        ).sort("timestamp")
        
        if resampled.is_empty():
            return
        
        self._append_to_cache(target_tf, resampled)

    def clear_cache(self):
        """Очистка кэша (например при смене монеты или рестарте)"""
        for tf in self.cache:
            self.cache[tf].clear()
            self.last_ts[tf] = 0
        logger.info("Кэш ресэмплера очищен")

    def get_cache_stats(self) -> Dict[str, int]:
        """Для отладки: сколько свечей в памяти по каждому TF"""
        return {tf: len(self.cache[tf]) for tf in self.timeframes}


# Пример использования (тест)
if __name__ == "__main__":
    cfg = {"hardware_mode": "phone_tiny"}
    resampler = Resampler(cfg)
    
    # Симулируем добавление 1m свечей
    for i in range(600):
        candle = {
            "open_time": int(datetime.now().timestamp() * 1000) - i * 60000,
            "open": 60000 + i * 10,
            "high": 60100 + i * 10,
            "low": 59900 + i * 10,
            "close": 60050 + i * 10,
            "volume": 10.0,
            "buy_volume": 6.0
        }
        resampler.add_1m_candle(candle)
    
    print(resampler.get_cache_stats())
    # Ожидаем ~500 на каждом TF (лимит на телефоне)