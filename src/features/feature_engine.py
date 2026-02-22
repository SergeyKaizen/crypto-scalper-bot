"""
src/features/feature_engine.py

=== Основной принцип работы файла ===

Этот файл отвечает за полный расчёт всех признаков (features) для подачи в модель.
Он работает на последовательностях свечей (time series sequences) для каждого таймфрейма и каждого окна (24, 50, 74, 100 свечей).

Основные задачи:
- Вычисление базовых признаков: volume, bid, ask, delta, price change, volatility, средние цены половин
- Расчёт ценового канала (anomalous_surge_channel_feature)
- Расчёт Value Area (VA) и delta Value Area (POC, VAH, VAL) для каждого окна
- Генерация сравнений между половинами окна (left vs right)
- Генерация новых delta-признаков: binary (delta_positive, delta_increased), количественных (delta_change_pct, norm_dist_to_delta_vah и др.)
- Формирование последовательности фич (shape: [seq_len, n_features]) для Conv1D+GRU

Признаки используются в inference, trainer и scenario_tracker.
Все расчёты производятся для последних N свечей (seq_len=100 по умолчанию).

=== Главные функции и за что отвечают ===

1. compute_features(df: pd.DataFrame, windows: list[int] = [24,50,74,100], seq_len: int = 100)
   → Основная функция: возвращает dict[tf][window] → pd.DataFrame или np.array последовательностей фич

2. _compute_half_comparison(features_left, features_right)
   → Сравнение левой и правой половины окна (включая delta)

3. _add_delta_features(va_dict, current_price, half_delta_change)
   → Генерация дополнительных delta-признаков

=== Примечания ===
- Для каждого окна считается VA и delta VA отдельно → позволяет модели видеть multi-scale imbalance
- Binary признаки (0/1) удобны для сценариев и бинарной статистики
- Количественные признаки нормализованы (где возможно) для стабильности обучения
- Если окно слишком маленькое (<30 свечей) — delta VA может быть шумным → можно добавить фильтр в будущем
"""

import numpy as np
import pandas as pd

from src.features.channels import anomalous_surge_channel_feature, calculate_volume_profile_va
from src.core.config import load_config

def compute_features(
    df: pd.DataFrame,
    windows: list[int] = [24, 50, 74, 100],
    seq_len: int = 100
) -> dict:
    """
    Основная функция расчёта всех признаков для одного таймфрейма.

    Параметры:
    ----------
    df : pd.DataFrame
        Свечи с колонками: open, high, low, close, volume, bid, ask
    windows : list[int]
        Окна для multi-window анализа
    seq_len : int
        Длина последовательности для модели (обычно 100)

    Возвращает:
    ----------
    dict[window_size] → pd.DataFrame или np.array последовательностей фич
    """
    config = load_config()
    features_by_window = {}

    # Базовая защита
    if len(df) < seq_len:
        return {}

    recent = df.tail(seq_len).copy()

    # 1. Базовые признаки (по всей последовательности)
    recent['mid_price'] = (recent['open'] + recent['close']) / 2
    recent['range_pct'] = (recent['high'] - recent['low']) / recent['close'] * 100
    recent['delta'] = recent['ask'] - recent['bid']
    recent['delta_pct'] = recent['delta'] / recent['volume'].replace(0, np.nan) * 100

    # 2. Ценовой канал (на весь seq_len или отдельно по окнам — здесь на весь)
    recent = anomalous_surge_channel_feature(recent, period=seq_len)

    # 3. Для каждого окна — полный расчёт VA и delta VA
    for w in windows:
        if len(recent) < w:
            continue

        window_df = recent.tail(w).copy()

        # Делим окно на две половины
        half = w // 2
        left = window_df.iloc[:half]
        right = window_df.iloc[half:]

        # --- Стандартный VA ---
        va_std = calculate_volume_profile_va(
            window_df,
            window=w,
            va_percentage=config.get('va_percentage', 0.70),
            price_bin_step_pct=config.get('price_bin_step_pct', 0.002),
            use_delta=False
        )

        # --- Delta VA ---
        va_delta = calculate_volume_profile_va(
            window_df,
            window=w,
            va_percentage=config.get('va_percentage', 0.70),
            price_bin_step_pct=config.get('price_bin_step_pct', 0.002),
            use_delta=True
        )

        # --- Сравнение половин (базовые + delta) ---
        left_va_delta = calculate_volume_profile_va(left, window=half, use_delta=True)
        right_va_delta = calculate_volume_profile_va(right, window=half, use_delta=True)

        delta_change_pct = 0.0
        if not np.isnan(left_va_delta['total_volume']) and left_va_delta['total_volume'] > 0:
            delta_change_pct = (
                (right_va_delta['total_volume'] - left_va_delta['total_volume'])
                / left_va_delta['total_volume'] * 100
            ) if not np.isnan(right_va_delta['total_volume']) else 0.0

        current_price = window_df['close'].iloc[-1]

        # --- Базовые признаки для окна ---
        window_features = {
            'volume_mean': window_df['volume'].mean(),
            'delta_mean': window_df['delta'].mean(),
            'volatility_mean': window_df['range_pct'].mean(),
            'price_change_pct': (window_df['close'].iloc[-1] - window_df['close'].iloc[0]) / window_df['close'].iloc[0] * 100,
            # Ценовой канал (последние значения)
            'asc_position': window_df['asc_position_in_channel'].iloc[-1],
            # Стандартный VA
            'poc_dist_norm': (current_price - va_std['poc_price']) / (va_std['vah'] - va_std['val']) if va_std['vah'] != va_std['val'] else 0,
            'in_va': 1 if va_std['val'] <= current_price <= va_std['vah'] else 0,
            'above_vah': 1 if current_price > va_std['vah'] else 0,
            'below_val': 1 if current_price < va_std['val'] else 0,
            # Delta VA признаки
            'delta_poc_dist_norm': (current_price - va_delta['poc_price']) / (va_delta['vah'] - va_delta['val']) if va_delta['vah'] != va_delta['val'] else 0,
            'delta_positive': 1 if va_delta['total_volume'] > 0 else 0,          # общая delta > 0
            'delta_increased': 1 if delta_change_pct > 0 else 0,                 # выросла во второй половине
            'delta_change_pct': delta_change_pct,
            'norm_dist_to_delta_vah': (current_price - va_delta['vah']) / va_delta['vah'] * 100 if va_delta['vah'] != 0 else 0,
            'norm_dist_to_delta_val': (current_price - va_delta['val']) / va_delta['val'] * 100 if va_delta['val'] != 0 else 0,
            # Можно добавить ещё: delta_poc_shift, delta_vah_shift и т.д.
        }

        # Сохраняем для окна
        features_by_window[w] = window_features

    return features_by_window


def prepare_sequence_features(
    df: pd.DataFrame,
    seq_len: int = 100,
    windows: list[int] = [24, 50, 74, 100]
) -> np.ndarray:
    """
    Подготавливает последовательность фич для подачи в модель (Conv1D input).
    Для каждой свечи в seq_len собирает признаки из всех окон.
    """
    features_list = []

    for i in range(len(df) - seq_len + 1):
        window_df = df.iloc[i:i+seq_len]
        feats = compute_features(window_df, windows=windows, seq_len=seq_len)

        # Преобразуем в плоский вектор (можно оптимизировать)
        flat_feats = []
        for w in windows:
            if w in feats:
                flat_feats.extend(list(feats[w].values()))

        features_list.append(flat_feats)

    return np.array(features_list)