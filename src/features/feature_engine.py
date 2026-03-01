"""
src/features/feature_engine.py

=== Основной принцип работы файла ===

Центральный модуль генерации признаков для модели.

Ключевые задачи:
- compute_features(df) → основной метод, возвращает словарь фич по таймфреймам
- _aggregate_features(df) → агрегация по окнам (24, 50, 74, 100)
- расчёт VA_position, quiet_streak, anomaly-based фич и т.д.
- все фичи должны быть без look-ahead bias и без лишних индикаторов

Изменения Этапа 1 (пункты 1+2+3):
- Пункт 3: полностью удалён метод _add_indicators и все вызовы EMA/ATR/RSI/и т.д.
- Пункт 1: va_position теперь считается точно как (close - VAL) / (VAH - VAL) ∈ [0,1]
- Пункт 2: quiet_streak — количество последовательных баров, где (high-low) < 0.4 × среднего диапазона за последние 20 баров
"""

import polars as pl
import numpy as np
from typing import Dict, Any

import torch  # FIX Фаза 7: добавлен для согласования с моделью

from src.core.config import load_config
from src.utils.logger import setup_logger

logger = setup_logger("feature_engine", logging.INFO)


class FeatureEngine:
    def __init__(self, config: dict = None):
        self.config = config or load_config()
        self.windows = self.config.get("windows", [24, 50, 74, 100])
        self.quiet_window = 20          # фиксированное окно для quiet_streak
        self.quiet_threshold = 0.4      # порог 40% от среднего диапазона

    def compute_features(self, df: pl.DataFrame) -> Dict[str, Any]:
        """
        Основной метод: возвращает все признаки для данного df (один таймфрейм)
        """
        if df.is_empty():
            return {"sequences": {}, "features": {}}

        features = {}
        sequences = {}

        # Агрегация по разным окнам
        for window in self.windows:
            if len(df) < window:
                continue

            window_df = df.tail(window)
            agg = self._aggregate_features(window_df)

            # Последовательность — нормализованные цены или raw фичи (по твоей логике)
            seq = self._normalize_sequence(window_df)

            features[window] = agg
            sequences[window] = seq

        # FIX Фаза 7: преобразование в torch.tensor (согласованно с inference и моделью)
        sequences = {k: torch.tensor(v, dtype=torch.float32) for k, v in sequences.items()}

        return {
            "sequences": sequences,
            "features": features
        }

    def _aggregate_features(self, df: pl.DataFrame) -> Dict[str, float]:
        """
        Агрегация признаков по одному окну.
        Здесь считаются VA_position, quiet_streak и другие ключевые фичи.
        """
        if df.is_empty():
            return {}

        features = {}

        # === Пункт 1: Точный VA_position (замена старого group_by_dynamic) ===
        va_info = self._compute_value_area(df)
        current_close = df["close"].last()

        if va_info["VAH"] is None or va_info["VAL"] is None or va_info["VAH"] == va_info["VAL"]:
            va_position = 0.5  # вырожденный случай (диапазон нулевой)
        else:
            va_position = (current_close - va_info["VAL"]) / (va_info["VAH"] - va_info["VAL"])
            va_position = max(0.0, min(1.0, va_position))  # ограничиваем [0, 1]

        features["va_position"] = va_position

        # === Пункт 2: Новый quiet_streak без ATR и множителей ===
        quiet_streak = self._compute_quiet_streak(df)
        features["quiet_streak"] = quiet_streak

        # === Остальные твои фичи (если есть) остаются без изменений ===
        # Например:
        # features["volume_spike"] = ...
        # features["anomaly_score"] = ...

        return features

    def _compute_value_area(self, df: pl.DataFrame) -> Dict[str, float | None]:
        """
        Простой расчёт Value Area (VAH, VAL, POC) за всё окно.
        Используем упрощённый профиль объёма по цене.
        """
        if len(df) < 10:
            return {"VAH": None, "VAL": None, "POC": None}

        # POC — цена с максимальным объёмом (или модой close)
        poc = df.group_by("close").agg(pl.col("volume").sum()).sort("volume", descending=True).first()["close"][0]

        # Диапазон цен
        price_min = df["low"].min()
        price_max = df["high"].max()
        price_range = price_max - price_min

        # Value Area — 70% диапазона вокруг POC (упрощённо)
        vah = poc + 0.35 * price_range
        val = poc - 0.35 * price_range

        return {"VAH": vah, "VAL": val, "POC": poc}

    def _compute_quiet_streak(self, df: pl.DataFrame) -> int:
        """
        Новый quiet_streak:
        - берём последние self.quiet_window баров (20)
        - считаем средний диапазон (high - low) за это окно
        - считаем, сколько баров подряд с конца диапазон < threshold * средний
        """
        if len(df) < 2:
            return 0

        ranges = df["high"] - df["low"]

        # Средний диапазон за последние 20 баров (или меньше, если баров мало)
        window = min(len(ranges), self.quiet_window)
        avg_range = ranges.tail(window).mean()

        if avg_range == 0:
            return 0

        # Текущий диапазон (последний бар)
        current_range = ranges.last()
        if current_range >= self.quiet_threshold * avg_range:
            return 0  # текущий бар не тихий → streak = 0

        # Считаем streak с конца
        streak = 1
        for i in range(2, len(ranges) + 1):
            prev_range = ranges[-i]
            if prev_range < self.quiet_threshold * avg_range:
                streak += 1
            else:
                break

        return streak

    def _normalize_sequence(self, df: pl.DataFrame) -> np.ndarray:
        """
        Нормализация последовательности (твоя логика остаётся)
        """
        # Пример твоей нормализации (замени на свою)
        closes = df["close"].to_numpy()
        mean = closes.mean()
        std = closes.std()
        if std == 0:
            std = 1
        return (closes - mean) / std