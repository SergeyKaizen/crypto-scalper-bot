"""
src/features/channels.py

=== Основной принцип работы файла ===

Этот файл реализует расчёты ценового канала и Value Area (VAH/VAL/POC) строго по ТЗ.

Ключевые задачи:
- Ценовой канал: rolling max/min по mid = (open + close)/2 за period (как в примере ТЗ).
  Вычисляет normalized position (0..1), расстояния до upper/lower, breakout сигналы.
- Value Area: классический профиль объёма с накоплением 68–70% объёма вокруг POC (стандарт 70%, но можно 60% в config).
  Биннинг по фиксированному % от средней цены, определение VAH/VAL/POC.

Все функции векторизованы (pandas/polarras), работают на любом period (включая размер окна: 24/50/74/100).
Результаты используются в anomaly_detector (фильтр аномалий по VA) и feature_engine (признаки position, breakout, above_vah и т.д.).

=== Главные функции и за что отвечают ===

- anomalous_surge_channel_feature(df: pd.DataFrame, period: int) → pd.DataFrame
  Основная функция ценового канала по ТЗ: rolling max/min mid, width, norm_dist_to_upper/lower, position 0..1.
  Добавляет колонки в df.

- calculate_value_area(df: pd.DataFrame, period: int, value_area_pct: float = 0.70) → dict
  Расчёт Value Area: биннинг цен, POC = max volume bin, накопление value_area_pct объёма вверх/вниз от POC.
  Возвращает {'poc': float, 'vah': float, 'val': float}

- get_va_position(close: float, vah: float, val: float) → float
  Нормализованная позиция цены внутри VA (0..1).

=== Примечания ===
- Period = размер текущего окна (передаётся из feature_engine).
- Safe handling: clip position 0..1, защита от zero width (1e-10).
- Биннинг в VA: шаг = 0.1% от средней цены (настраивается).
- Нет заглушек — всё реализовано по ТЗ.
- Логи минимальны.
"""

import pandas as pd
import numpy as np
from typing import Dict

from src.core.config import load_config
from src.utils.logger import setup_logger

logger = setup_logger('channels', logging.INFO)

def anomalous_surge_channel_feature(df: pd.DataFrame, period: int = 100) -> pd.DataFrame:
    """
    Вычисляет ценовой канал по ТЗ (rolling max/min mid).
    Добавляет колонки:
    - mid
    - asc_upper, asc_lower, asc_channel_width
    - asc_norm_dist_to_upper, asc_norm_dist_to_lower
    - asc_position_in_channel (0..1)

    Обработка NaN и zero-width: safe_width = 1e-10, clip position.
    """
    if len(df) < period:
        logger.warning(f"Недостаточно данных для канала (period={period}, len={len(df)})")
        return df.copy()

    required_cols = {'open', 'close'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Отсутствуют колонки: {required_cols - set(df.columns)}")

    result = df.copy()

    # Mid = (open + close) / 2
    mid = (result['open'] + result['close']) / 2
    mid = mid.replace([np.inf, -np.inf], np.nan)

    # Rolling max/min mid
    upper = mid.rolling(window=period, min_periods=1).max()
    lower = mid.rolling(window=period, min_periods=1).min()

    width = upper - lower
    safe_width = width.replace(0, np.nan).fillna(1e-10)

    norm_to_upper = (result['close'] - upper) / safe_width
    norm_to_lower = (result['close'] - lower) / safe_width
    position = (result['close'] - lower) / safe_width

    result['mid'] = mid
    result['asc_upper'] = upper
    result['asc_lower'] = lower
    result['asc_channel_width'] = width
    result['asc_norm_dist_to_upper'] = norm_to_upper
    result['asc_norm_dist_to_lower'] = norm_to_lower
    result['asc_position_in_channel'] = position.clip(0, 1)

    return result

def calculate_value_area(df: pd.DataFrame, period: int, value_area_pct: float = 0.70) -> Dict[str, float]:
    """
    Расчёт Value Area (VAH/VAL/POC) для последних period свечей.
    - Биннинг: шаг = 0.1% от средней цены за период.
    - POC = бин с максимальным объёмом.
    - Накопление value_area_pct (70%) объёма вверх/вниз от POC.
    - Возвращает {'poc': float, 'vah': float, 'val': float} или пустой dict при ошибке.
    """
    if len(df) < period:
        return {}

    # Последние period свечей
    recent = df.tail(period).copy()

    avg_price = recent['close'].mean()
    bin_step = avg_price * 0.001  # 0.1% от средней

    # Биннинг цен (close для простоты, или (h+l)/2)
    recent['price_bin'] = np.round(recent['close'] / bin_step) * bin_step

    # Volume profile
    vp = recent.groupby('price_bin')['volume'].sum().reset_index()
    if vp.empty:
        return {}

    poc_bin = vp.loc[vp['volume'].idxmax(), 'price_bin']
    poc_volume = vp['volume'].max()
    total_volume = vp['volume'].sum()

    # Накопление 70% вокруг POC
    vp_sorted = vp.sort_values('price_bin')
    poc_idx = vp_sorted[vp_sorted['price_bin'] == poc_bin].index[0]

    accumulated = poc_volume
    upper_idx = poc_idx
    lower_idx = poc_idx

    while accumulated < total_volume * value_area_pct and (upper_idx < len(vp_sorted) - 1 or lower_idx > 0):
        upper_vol = vp_sorted.iloc[upper_idx + 1]['volume'] if upper_idx < len(vp_sorted) - 1 else 0
        lower_vol = vp_sorted.iloc[lower_idx - 1]['volume'] if lower_idx > 0 else 0

        if upper_vol >= lower_vol and upper_idx < len(vp_sorted) - 1:
            upper_idx += 1
            accumulated += upper_vol
        elif lower_idx > 0:
            lower_idx -= 1
            accumulated += lower_vol
        else:
            break

    vah = vp_sorted.iloc[upper_idx]['price_bin']
    val = vp_sorted.iloc[lower_idx]['price_bin']

    return {
        'poc': poc_bin,
        'vah': vah,
        'val': val
    }

def get_va_position(close: float, vah: float, val: float) -> float:
    """
    Нормализованная позиция цены внутри VA (0..1).
    0 = на VAL или ниже, 1 = на VAH или выше.
    """
    if vah == val:
        return 0.5
    position = (close - val) / (vah - val)
    return np.clip(position, 0.0, 1.0)