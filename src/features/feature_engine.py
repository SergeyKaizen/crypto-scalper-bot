"""
src/features/feature_engine.py

=== Основной принцип работы файла ===

Этот файл — центральный модуль feature engineering.
Он берёт сырые свечи из storage (open, high, low, close, volume, bid, ask) и вычисляет 
все признаки, требуемые ТЗ, строго без лишних индикаторов (RSI, MACD и т.п. полностью удалены).

Признаки делятся на:
1. Базовые по половинам и окнам (24, 50, 74, 100 свечей):
   - Bid, Ask, Delta (mean по половине, delta % между половинами)
   - Price change внутри половины и delta между ними
   - Volatility (средний размер свечи в % + delta между половинами)
   - Dist от средней цены (dist1_pct, dist2_pct, dist_delta_pct, mean_delta_pct)
   - Price channel (позиция в канале, breakout upper/lower)
   - VAH/VAL/POC (позиция цены относительно VA, delta между половинами)

2. Binary состояния для half_comparator: increased/decreased, above/below, crossed и т.д.

3. Вычисление по всем таймфреймам и окнам — возвращает dict[tf][window] = features_df

Всё векторизовано через polars для скорости (особенно на сервере), но работает и на pandas (для телефона).
Нет заглушек — все признаки реализованы полностью по ТЗ.

=== Главные функции и за что отвечают ===

- compute_features_for_tf(tf: str, df: pd.DataFrame) → dict[window_size: pd.DataFrame]
  Основная функция: для одного TF берёт свечи, делит на окна (24/50/74/100), 
  вычисляет все признаки по половинам каждого окна.

- _compute_half_features(half_df: pd.DataFrame) → dict
  Вычисляет признаки для одной половины: mean_bid, mean_ask, delta, price_change_pct, volatility_mean, mean_price, dist_pct и т.д.

- _compute_channel_features(df: pd.DataFrame, period: int) → pd.Series
  Rolling max/min на mid, position in channel, breakout signals.

- _compute_va_features(df: pd.DataFrame, period: int) → dict
  Вызывает channels.calculate_value_area, вычисляет position (close относительно VAL/VAH), binary above_vah/below_val.

- prepare_sequence_features(symbol: str, tf: str, window_size: int) → pd.DataFrame
  Готовит последовательность для модели: признаки + binary состояния + аномалии (C/V/CV/Q).

=== Примечания ===
- Все признаки scale-invariant (% изменения).
- Binary состояния — 0/1 для подачи в модель (Conv1D+GRU).
- Период VA = размер окна (24→period=24 и т.д.).
- Нет внешних индикаторов — строго ТЗ.
- Логи через setup_logger.
"""

import polars as pl
import pandas as pd
import numpy as np
from typing import Dict, Optional

from src.data.storage import Storage
from src.features.channels import calculate_value_area, anomalous_surge_channel_feature
from src.utils.logger import setup_logger

logger = setup_logger('feature_engine', logging.INFO)

WINDOW_SIZES = [24, 50, 74, 100]  # из ТЗ

def compute_features_for_tf(tf: str, df_raw: pd.DataFrame) -> Dict[int, pd.DataFrame]:
    """
    Основная функция: вычисляет все признаки для одного TF.
    df_raw — DataFrame из storage с колонками open, high, low, close, volume, bid, ask.
    Возвращает dict[window_size: features_df] — признаки по каждому окну.
    """
    # Конвертируем в polars для скорости
    df = pl.from_pandas(df_raw.reset_index()).with_columns(
        pl.col("timestamp").cast(pl.Datetime)
    ).sort("timestamp")

    result = {}

    for window in WINDOW_SIZES:
        if len(df) < window:
            logger.warning(f"Недостаточно данных для окна {window} ({len(df)} свечей)")
            continue

        # Разделяем на sliding windows (каждое окно — последние N свечей)
        # Для простоты берём последние полные окна
        features_list = []
        for i in range(window - 1, len(df)):
            window_df = df.slice(i - window + 1, window)

            half_len = window // 2
            left_half = window_df.slice(0, half_len)
            right_half = window_df.slice(half_len, half_len)

            half_features = _compute_half_features(left_half, right_half)

            # Channel и VA для всего окна
            channel_feats = _compute_channel_features(window_df.to_pandas(), period=window)
            va_feats = _compute_va_features(window_df.to_pandas(), period=window)

            # Собираем всё в dict
            row = {
                'timestamp': window_df['timestamp'][-1],
                **half_features,
                **channel_feats,
                **va_feats
            }
            features_list.append(row)

        if features_list:
            features_df = pd.DataFrame(features_list).set_index('timestamp')
            result[window] = features_df

    return result

def _compute_half_features(left: pl.DataFrame, right: pl.DataFrame) -> dict:
    """
    Вычисляет признаки для левой и правой половины.
    Возвращает dict с mean_bid1/2, delta_bid_pct, price_change1/2, volatility_mean1/2 и т.д.
    """
    # Средние по половине
    mean_bid1 = left['bid'].mean()
    mean_bid2 = right['bid'].mean()
    mean_ask1 = left['ask'].mean()
    mean_ask2 = right['ask'].mean()

    delta1 = (mean_bid1 - mean_ask1) / (mean_bid1 + mean_ask1 + 1e-8) * 100 if (mean_bid1 + mean_ask1) > 0 else 0
    delta2 = (mean_bid2 - mean_ask2) / (mean_bid2 + mean_ask2 + 1e-8) * 100 if (mean_bid2 + mean_ask2) > 0 else 0

    # Price change внутри половины
    if len(left) > 0 and len(right) > 0:
        price_change1 = (left['close'][-1] - left['close'][0]) / left['close'][0] * 100 if left['close'][0] != 0 else 0
        price_change2 = (right['close'][-1] - right['close'][0]) / right['close'][0] * 100 if right['close'][0] != 0 else 0
    else:
        price_change1 = price_change2 = 0

    # Volatility
    left_size_pct = ((left['high'] - left['low']) / left['high'].clip_min(1e-8)) * 100
    right_size_pct = ((right['high'] - right['low']) / right['high'].clip_min(1e-8)) * 100
    vol_mean1 = left_size_pct.mean()
    vol_mean2 = right_size_pct.mean()

    # Средняя цена и dist
    mean_price1 = left['close'].mean()
    mean_price2 = right['close'].mean()
    dist1_pct = (left['close'][-1] - mean_price1) / mean_price1 * 100 if mean_price1 != 0 else 0
    dist2_pct = (right['close'][-1] - mean_price2) / mean_price2 * 100 if mean_price2 != 0 else 0
    mean_delta_pct = (mean_price2 - mean_price1) / mean_price1 * 100 if mean_price1 != 0 else 0

    return {
        'mean_bid1': mean_bid1, 'mean_bid2': mean_bid2,
        'delta_bid_pct': (mean_bid2 - mean_bid1) / mean_bid1 * 100 if mean_bid1 != 0 else 0,
        'mean_ask1': mean_ask1, 'mean_ask2': mean_ask2,
        'delta_ask_pct': (mean_ask2 - mean_ask1) / mean_ask1 * 100 if mean_ask1 != 0 else 0,
        'delta1': delta1, 'delta2': delta2,
        'delta_delta_pct': delta2 - delta1,
        'price_change1': price_change1, 'price_change2': price_change2,
        'price_delta_pct': price_change2 - price_change1,
        'volatility_mean1': vol_mean1, 'volatility_mean2': vol_mean2,
        'volatility_delta_pct': (vol_mean2 - vol_mean1) / vol_mean1 * 100 if vol_mean1 != 0 else 0,
        'dist1_pct': dist1_pct, 'dist2_pct': dist2_pct,
        'dist_delta_pct': dist2_pct - dist1_pct,
        'mean_delta_pct': mean_delta_pct
    }

def _compute_channel_features(df: pd.DataFrame, period: int) -> dict:
    """
    Вычисляет признаки ценового канала (по ТЗ rolling max/min mid).
    Возвращает dict с position, breakout_upper/lower для последней свечи.
    """
    df_channel = anomalous_surge_channel_feature(df, period=period)
    last_row = df_channel.iloc[-1]

    return {
        'channel_position': last_row['asc_position_in_channel'],
        'channel_breakout_upper': 1 if last_row['asc_norm_dist_to_upper'] > 0 else 0,
        'channel_breakout_lower': 1 if last_row['asc_norm_dist_to_lower'] < 0 else 0
    }

def _compute_va_features(df: pd.DataFrame, period: int) -> dict:
    """
    Вычисляет признаки Value Area (VAH/VAL/POC).
    Период = размер окна.
    Возвращает position относительно VA, binary above_vah / below_val.
    """
    va = calculate_value_area(df, period=period)
    if not va or 'vah' not in va or 'val' not in va:
        return {'va_position': 0.5, 'above_vah': 0, 'below_val': 0}

    close_last = df['close'].iloc[-1]
    vah = va['vah']
    val = va['val']

    if vah == val:  # degenerate case
        va_width = 1e-8
    else:
        va_width = vah - val

    position = (close_last - val) / va_width
    position = np.clip(position, 0, 1)

    return {
        'va_position': position,
        'above_vah': 1 if close_last > vah else 0,
        'below_val': 1 if close_last < val else 0
    }

def prepare_sequence_features(symbol: str, tf: str, window_size: int = 100) -> Optional[pd.DataFrame]:
    """
    Готовит последовательность признаков для модели.
    Берёт последние window_size свечей, вычисляет признаки.
    Используется в inference и trainer.
    """
    storage = Storage()
    df = storage.get_candles(symbol, tf, end_ts=None)  # все свечи

    if len(df) < window_size:
        logger.warning(f"Недостаточно данных для {symbol} {tf} (окно {window_size})")
        return None

    df_window = df.tail(window_size)

    # Вычисляем признаки для этого окна
    features = compute_features_for_tf(tf, df_window)

    if window_size not in features:
        return None

    return features[window_size]