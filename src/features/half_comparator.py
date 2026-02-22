"""
src/features/half_comparator.py

=== Основной принцип работы файла ===

Файл отвечает за сравнение левой (старой) и правой (новой) половины окна свечей.
Это ключевая часть ТЗ: модель видит переход между половинами, выявляет изменения признаков.

Для каждого окна (24/50/74/100) делим данные на две равные части:
- left: первые half свечей
- right: вторые half свечей

Расчёт:
- Количественные изменения (pct, absolute) для всех 12 базовых признаков из ТЗ
- Полное сравнение delta VA (poc_shift, vah/val shifts, width change, expanded)
- Binary флаги: increased/decreased, positive/negative, crossed_delta_vah и т.д.
- Всё возвращается в dict для feature_engine

=== Главные функции и за что отвечают ===

1. compare_halves(window_df: pd.DataFrame, window_size: int, va_std: dict, va_delta: dict)
   → Основная функция: возвращает dict с изменениями между половинами

2. _compute_base_changes(left, right)
   → Сравнение 12 базовых признаков из ТЗ (volume, bid, ask, delta и т.д.)

3. _compute_delta_va_changes(va_left, va_right, current_price)
   → Сравнение delta VA (poc_shift, vah/val shift, width_change, expanded, crossed)

=== Примечания ===
- Все % изменения считаются относительно left (старой половины)
- Защита от деления на 0 (fillna 0 или small epsilon)
- Binary флаги используются в sequential паттернах и сценариях
- Количественные изменения clipping'уются в feature_engine
"""

import numpy as np
import pandas as pd

def compare_halves(
    window_df: pd.DataFrame,
    window_size: int,
    va_std: dict,
    va_delta: dict,
    current_price: float
) -> dict:
    """
    Сравнивает левую и правую половину окна.

    Параметры:
    ----------
    window_df : pd.DataFrame
        Свечи окна (24/50/74/100)
    window_size : int
        Размер окна
    va_std : dict
        Результат calculate_volume_profile_va(use_delta=False)
    va_delta : dict
        Результат calculate_volume_profile_va(use_delta=True)
    current_price : float
        Текущая цена закрытия

    Возвращает:
    ----------
    dict с количественными и бинарными изменениями
    """
    half = window_size // 2
    left = window_df.iloc[:half]
    right = window_df.iloc[half:]

    changes = {}

    # 1. Базовые признаки (из ТЗ)
    changes.update(_compute_base_changes(left, right))

    # 2. Delta VA изменения
    va_left = calculate_volume_profile_va(left, window=half, use_delta=True)
    va_right = calculate_volume_profile_va(right, window=half, use_delta=True)

    changes.update(_compute_delta_va_changes(va_left, va_right, current_price, va_delta))

    return changes


def _compute_base_changes(left: pd.DataFrame, right: pd.DataFrame) -> dict:
    """
    Сравнение 12 базовых признаков из ТЗ
    """
    changes = {}

    # Volume, Bid, Ask, Delta (средние по половине)
    for col in ['volume', 'bid', 'ask', 'delta']:
        left_mean = left[col].mean()
        right_mean = right[col].mean()
        if left_mean != 0:
            pct_change = (right_mean - left_mean) / abs(left_mean) * 100
        else:
            pct_change = 0.0
        changes[f'{col}_change_pct'] = pct_change
        changes[f'{col}_increased'] = 1 if pct_change > 0 else 0

    # Средние цены половин относительно общей средней
    total_mid = (left['close'].mean() + right['close'].mean()) / 2
    left_mid = left['close'].mean()
    right_mid = right['close'].mean()
    changes['left_mid_dist_pct'] = (left_mid - total_mid) / total_mid * 100 if total_mid != 0 else 0
    changes['right_mid_dist_pct'] = (right_mid - total_mid) / total_mid * 100 if total_mid != 0 else 0
    changes['mid_dist_delta_pct'] = changes['right_mid_dist_pct'] - changes['left_mid_dist_pct']

    # Price change % в половине
    left_price_change = (left['close'].iloc[-1] - left['close'].iloc[0]) / left['close'].iloc[0] * 100
    right_price_change = (right['close'].iloc[-1] - right['close'].iloc[0]) / right['close'].iloc[0] * 100
    changes['price_change_left_pct'] = left_price_change
    changes['price_change_right_pct'] = right_price_change
    changes['price_change_diff_pct'] = right_price_change - left_price_change
    changes['price_change_increased'] = 1 if changes['price_change_diff_pct'] > 0 else 0

    # Volatility (средний range %)
    left_vol = ((left['high'] - left['low']) / left['close'] * 100).mean()
    right_vol = ((right['high'] - right['low']) / right['close'] * 100).mean()
    changes['volatility_change_pct'] = (right_vol - left_vol) / left_vol * 100 if left_vol != 0 else 0
    changes['volatility_increased'] = 1 if changes['volatility_change_pct'] > 0 else 0

    return changes


def _compute_delta_va_changes(va_left: dict, va_right: dict, current_price: float, va_delta: dict) -> dict:
    """
    Сравнение delta VA между половинами
    """
    changes = {}

    if np.isnan(va_left['poc_price']) or np.isnan(va_right['poc_price']):
        return changes

    # POC shift
    poc_shift = va_right['poc_price'] - va_left['poc_price']
    mid_price = (va_left['poc_price'] + va_right['poc_price']) / 2
    changes['delta_poc_shift_pct'] = poc_shift / mid_price * 100 if mid_price != 0 else 0
    changes['delta_poc_shift_positive'] = 1 if changes['delta_poc_shift_pct'] > 0 else 0

    # VAH / VAL shift
    vah_shift = va_right['vah'] - va_left['vah']
    val_shift = va_right['val'] - va_left['val']
    changes['delta_vah_shift_pct'] = vah_shift / va_left['vah'] * 100 if va_left['vah'] != 0 else 0
    changes['delta_val_shift_pct'] = val_shift / va_left['val'] * 100 if va_left['val'] != 0 else 0

    # Width change (расширение/сужение delta VA)
    left_width = va_left['vah'] - va_left['val']
    right_width = va_right['vah'] - va_right['val']
    changes['delta_va_width_change_pct'] = (right_width - left_width) / left_width * 100 if left_width != 0 else 0
    changes['delta_va_expanded'] = 1 if changes['delta_va_width_change_pct'] > 0 else 0

    # Пересечения в правой половине
    changes['current_crossed_delta_vah'] = 1 if current_price > va_right['vah'] else 0
    changes['current_crossed_delta_val'] = 1 if current_price < va_right['val'] else 0

    # Общее расстояние до delta VAH/VAL в правой половине
    changes['norm_dist_to_delta_vah_right'] = (current_price - va_right['vah']) / va_right['vah'] * 100 if va_right['vah'] != 0 else 0
    changes['norm_dist_to_delta_val_right'] = (current_price - va_right['val']) / va_right['val'] * 100 if va_right['val'] != 0 else 0

    return changes