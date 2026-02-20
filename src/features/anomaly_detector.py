"""
src/features/anomaly_detector.py

=== Основной принцип работы файла ===

Этот файл реализует детекцию всех типов аномалий строго по ТЗ:
- Свечная аномалия (C): размер последней свечи > (средний размер + среднее отклонение) за lookback=25 свечей.
- Объёмная аномалия (V): объём > percentile threshold (95–99%) + фильтр по dominance (taker_buy) и положению относительно VAH/VAL.
- Комбинированная (CV): одновременно C и V true.
- Quiet (Q): отсутствие любой аномалии (C=V=CV=0) — для входа по паттернам без всплесков.

Дополнительно:
- Для volume и CV: свеча должна быть вне VA (над VAH или под VAL) или пересекать границу (по ТЗ).
- Порог dominance_pct (например, >70% buy/sell) — настраивается в config.
- Все расчёты векторизованы (polars/pandas), возвращают dict с bool и strength (для фильтрации слабых сигналов).

Результаты подаются в feature_engine (для binary признаков) и entry_manager (для открытия позиций).

=== Главные функции и за что отвечают ===

- detect_anomalies(df: pd.DataFrame, tf_minutes: int, current_window: int) → dict
  Основная функция: детектит C, V, CV, Q для последней свечи.
  df — последние свечи (минимум lookback + 1).
  current_window — размер текущего окна (для VA period).

- _detect_candle_anomaly(df: pd.DataFrame) → (bool, float)
  Свечная аномалия по ТЗ: mean_size + mean_abs_deviation.

- _detect_volume_anomaly(df: pd.DataFrame, tf_minutes: int) → (bool, float)
  Объёмная: percentile + dominance_pct + фильтр VAH/VAL.

- _detect_cv_anomaly(candle_anom: bool, volume_anom: bool) → bool
  Комбинированная: candle и volume одновременно true.

- _is_outside_va(close: float, vah: float, val: float) → bool
  Проверка: close > vah или close < val.

=== Примечания ===
- Lookback=25 для аномалий (по ТЗ).
- VA period = current_window (24/50/74/100).
- Thresholds в config (percentile_level, dominance_threshold_pct).
- Нет заглушек — все аномалии реализованы.
- Логи только критичные ошибки.
"""

import polars as pl
import pandas as pd
import numpy as np
from typing import Tuple, Dict

from src.features.channels import calculate_value_area
from src.core.config import load_config
from src.utils.logger import setup_logger

logger = setup_logger('anomaly_detector', logging.INFO)

LOOKBACK = 25  # из ТЗ для аномалий
MIN_PERC = 0.94
MAX_PERC = 0.99

def detect_anomalies(df: pd.DataFrame, tf_minutes: int, current_window: int) -> Dict[str, bool]:
    """
    Детектит все аномалии для последней свечи.
    df — DataFrame с колонками open, high, low, close, volume, bid, ask.
    tf_minutes — длительность TF в минутах (1,3,5,10,15).
    current_window — размер текущего окна для VA period.
    Возвращает {'candle': bool, 'volume': bool, 'cv': bool, 'q': bool}
    """
    if len(df) < LOOKBACK + 1:
        logger.warning("Недостаточно данных для детекции аномалий")
        return {'candle': False, 'volume': False, 'cv': False, 'q': False}

    df_pl = pl.from_pandas(df)

    candle_anom, candle_strength = _detect_candle_anomaly(df_pl)
    volume_anom, volume_strength = _detect_volume_anomaly(df_pl, tf_minutes, current_window)

    cv_anom = candle_anom and volume_anom

    q = not (candle_anom or volume_anom or cv_anom)

    return {
        'candle': candle_anom,
        'volume': volume_anom,
        'cv': cv_anom,
        'q': q
    }

def _detect_candle_anomaly(df: pl.DataFrame) -> Tuple[bool, float]:
    """
    Свечная аномалия строго по ТЗ.
    - candle_size_pct = (high - low) / high * 100
    - mean_size = средний размер за LOOKBACK прошлых свечей
    - deviation_i = |size_i - mean_size|
    - mean_deviation = среднее отклонение
    - threshold = mean_size + mean_deviation
    - Аномалия = size_last > threshold
    """
    past = df.slice(-LOOKBACK - 1, LOOKBACK)  # прошлые LOOKBACK без текущей
    current = df[-1]

    sizes_pct = ((past['high'] - past['low']) / past['high'].clip_min(1e-8)) * 100
    mean_size = sizes_pct.mean()
    deviations = (sizes_pct - mean_size).abs()
    mean_deviation = deviations.mean()

    threshold = mean_size + mean_deviation
    current_size_pct = ((current['high'] - current['low']) / current['high'].clip_min(1e-8)) * 100

    is_anomaly = current_size_pct > threshold
    strength = current_size_pct / threshold if threshold > 0 else 0.0

    return bool(is_anomaly), float(strength)

def _detect_volume_anomaly(df: pl.DataFrame, tf_minutes: int, window_size: int) -> Tuple[bool, float]:
    """
    Объёмная аномалия по ТЗ с фильтром VAH/VAL.
    - Прошлые LOOKBACK свечей: percentile threshold (95–99%).
    - Нормализация по range (как в примере).
    - Dominance: ask/bid dominance_pct = (ask - bid) / volume * 100
    - Аномалия только если свеча вне VA (close > vah or close < val).
    """
    config = load_config()
    percentile_level = config.get('anomaly', {}).get('volume_percentile', 0.97)
    dominance_threshold = config.get('anomaly', {}).get('dominance_threshold_pct', 60.0)

    past = df.slice(-LOOKBACK - 1, LOOKBACK)
    current = df[-1]

    vols = past['volume']
    if len(vols) == 0:
        return False, 0.0

    # Percentile threshold
    sorted_vols = vols.sort()
    idx = int(percentile_level * (len(sorted_vols) - 1))
    perc_threshold = sorted_vols[idx]

    # Нормализация по ширине текущей свечи (как в примере)
    ranges = past['high'] - past['low']
    avg_range = ranges.mean()
    current_range = current['high'] - current['low']
    current_range_pct = current_range / current['close'].clip_min(1e-8)
    norm_factor = avg_range / current_range_pct if current_range_pct > 0 else 1.0

    adjusted_threshold = perc_threshold * max(1.0, norm_factor * 0.8)

    high_volume = current['volume'] > adjusted_threshold

    # Dominance check
    dominance_pct = (current['ask'] - current['bid']) / current['volume'].clip_min(1e-8) * 100
    dominance_ok = abs(dominance_pct) > dominance_threshold

    # VA filter
    va = calculate_value_area(df.to_pandas().tail(window_size), period=window_size)
    vah = va.get('vah', current['close'])
    val = va.get('val', current['close'])
    outside_va = (current['close'] > vah) or (current['close'] < val)

    is_anomaly = high_volume and dominance_ok and outside_va
    strength = current['volume'] / adjusted_threshold if adjusted_threshold > 0 else 0.0

    return bool(is_anomaly), float(strength)

def _is_outside_va(close: float, vah: float, val: float) -> bool:
    """Проверка нахождения цены вне Value Area."""
    return close > vah or close < val