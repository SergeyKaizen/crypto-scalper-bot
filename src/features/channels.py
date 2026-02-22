"""
src/features/channels.py

=== Основной принцип работы файла ===

Файл содержит функции для расчёта ключевых зон и каналов, используемых в feature engineering:
- Ценовой канал (Anomalous Surge Channel) — оригинальная реализация из ТЗ для выявления аномальных всплесков цен
- Volume Profile + Value Area (VAH/VAL/POC) — классический расчёт профиля объёма за окно свечей
- Delta Volume Profile + Delta Value Area — расширение на основе bid/ask volume для выявления directional imbalance

Все функции работают на pd.DataFrame со свечами (open, high, low, close, volume, bid, ask).
Расчёты оптимизированы для скорости (numpy векторизация где возможно), подходят для интрадей (окна 24–100 свечей).

Ключевые возвращаемые значения используются в feature_engine для создания фич:
- расстояния до POC/VAH/VAL (нормализованные и абсолютные)
- binary признаки: внутри VA / за VAH / под VAL
- изменения между половинами окна / между окнами

=== Главные функции и за что отвечают ===

1. anomalous_surge_channel_feature(df, period=100)
   → Оригинальный ценовой канал (mid body rolling max/min) — сохранён без изменений

2. calculate_volume_profile_va(df, window=100, va_percentage=0.70, price_bin_step_pct=0.002, use_delta=False)
   → Основная функция: строит Volume Profile (или Delta Profile если use_delta=True)
   → Возвращает dict с POC, VAH, VAL, total_volume и дополнительными метриками

=== Примечания ===
- price_bin_step_pct — шаг бина в % от средней цены окна (0.002 = 0.2%)
- use_delta=True → строит профиль по delta = ask - bid вместо total volume
- Для delta-версии total_volume → total_abs_delta (сумма абсолютных delta по бинам)
- Функция устойчива к малым окнам и нулевому объёму (возвращает NaN)
- Логика VA итеративная (от POC добавляем по максимальной стороне) — стандарт TPO/Volume Profile
"""

import numpy as np
import pandas as pd

def anomalous_surge_channel_feature(df: pd.DataFrame, period: int = 100) -> pd.DataFrame:
    """
    Вычисляет признак Anomalous Price Channel для использования в машинном обучении / нейронной сети.
    
    Параметры:
    ----------
    df : pd.DataFrame
        Должен содержать хотя бы колонки 'open' и 'close'.
        Индекс может быть любым (timestamp, integer и т.д.).
    
    period : int, default=100
        Размер окна для поиска максимума и минимума середины тела свечи.
    
    Возвращает:
    ----------
    pd.DataFrame
        Исходный датафрейм с добавленными колонками-признаками:
        
        'mid'                        → (open + close) / 2
        'asc_upper'                  → rolling max(mid) за period баров
        'asc_lower'                  → rolling min(mid) за period баров
        'asc_channel_width'          → asc_upper - asc_lower
        'asc_norm_dist_to_upper'     → (close - asc_upper) / asc_channel_width
        'asc_norm_dist_to_lower'     → (close - asc_lower) / asc_channel_width
        'asc_position_in_channel'    → (close - asc_lower) / asc_channel_width   (0..1)
    
    Особенности:
    • Для первых period-1 строк значения будут NaN (стандартное поведение rolling)
    • NaN в исходных данных обрабатываются корректно (rolling использует nanmax/nanmin)
    • Все вычисления векторизованы — очень быстро даже на миллионах строк
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Ожидается pandas DataFrame")

    required = {'open', 'close'}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f"Отсутствуют необходимые колонки: {missing}")

    if period < 1:
        raise ValueError("period должен быть >= 1")

    if len(df) < period and len(df) > 0:
        print(f"Предупреждение: длина данных ({len(df)}) меньше периода ({period}). "
              f"Все значения будут NaN до конца данных.")

    # Середина тела свечи
    mid = (df['open'] + df['close']) / 2.0

    # Защита от inf (маловероятно в реальных данных, но на всякий случай)
    mid = mid.replace([np.inf, -np.inf], np.nan)

    # Rolling-окна (векторизованно, очень эффективно)
    upper = mid.rolling(window=period, min_periods=1).max()
    lower = mid.rolling(window=period, min_periods=1).min()

    # Полезные производные признаки
    width = upper - lower
    # Защита от деления на ноль (ширина = 0 крайне редко, но бывает на плоских рынках)
    safe_width = width.replace(0, np.nan).fillna(1e-10)  # очень маленькое значение вместо нуля

    norm_to_upper = (df['close'] - upper) / safe_width
    norm_to_lower = (df['close'] - lower) / safe_width
    position = (df['close'] - lower) / safe_width   # классическая нормализация 0..1

    # Собираем результат
    result = df.copy()
    result['mid']                     = mid
    result['asc_upper']               = upper
    result['asc_lower']               = lower
    result['asc_channel_width']       = width
    result['asc_norm_dist_to_upper']  = norm_to_upper
    result['asc_norm_dist_to_lower']  = norm_to_lower
    result['asc_position_in_channel'] = position.clip(0, 1)  # ограничиваем 0–1 для стабильности

    return result


def calculate_volume_profile_va(
    df: pd.DataFrame,
    window: int = 100,
    va_percentage: float = 0.70,
    price_bin_step_pct: float = 0.002,
    use_delta: bool = False
) -> dict:
    """
    Вычисляет Volume Profile (или Delta Volume Profile) и Value Area за последние window свечей.

    Параметры:
    ----------
    df : pd.DataFrame
        Свечи с колонками: open, high, low, close, volume, bid, ask
    window : int
        Количество последних свечей для анализа (24, 50, 74, 100)
    va_percentage : float
        Доля объёма для Value Area (обычно 0.68–0.70)
    price_bin_step_pct : float
        Шаг бина в % от средней цены окна (0.002 = 0.2%)
    use_delta : bool
        Если True — строит профиль по delta = ask - bid вместо total volume

    Возвращает:
    ----------
    dict с ключами:
        'poc_price', 'poc_volume', 'vah', 'val',
        'total_volume' (или total_abs_delta при use_delta=True),
        'va_volume', 'price_bins', 'volume_profile' (опционально)
    """
    if len(df) < window:
        return {
            'poc_price': np.nan, 'poc_volume': np.nan,
            'vah': np.nan, 'val': np.nan,
            'total_volume': 0.0, 'va_volume': 0.0
        }

    recent = df.tail(window).copy()

    # Средняя цена для расчёта относительного шага бина
    mid_price = recent['close'].mean()
    bin_step = mid_price * price_bin_step_pct

    # Определяем границы бинов
    price_min = recent['low'].min()
    price_max = recent['high'].max()
    bins = np.arange(price_min - bin_step, price_max + 2 * bin_step, bin_step)

    # Профиль объёма / дельты
    profile = np.zeros(len(bins) - 1)

    for _, row in recent.iterrows():
        low_idx = np.searchsorted(bins, row['low'], side='right') - 1
        high_idx = np.searchsorted(bins, row['high'], side='right') - 1

        if low_idx < 0 or high_idx >= len(profile):
            continue  # свеча за пределами — редкий случай

        if low_idx == high_idx:
            vol = row['volume']
            if use_delta:
                vol = row['ask'] - row['bid']
            profile[low_idx] += vol
        else:
            num_bins = high_idx - low_idx + 1
            vol_per_bin = (row['volume'] if not use_delta else (row['ask'] - row['bid'])) / num_bins
            profile[low_idx : high_idx + 1] += vol_per_bin

    # Общий объём (или сумма абсолютных дельт)
    total = np.sum(np.abs(profile)) if use_delta else np.sum(profile)
    if total == 0:
        return {
            'poc_price': np.nan, 'poc_volume': np.nan,
            'vah': np.nan, 'val': np.nan,
            'total_volume': 0.0, 'va_volume': 0.0
        }

    # POC — максимум профиля
    poc_idx = np.argmax(np.abs(profile) if use_delta else profile)
    poc_price = (bins[poc_idx] + bins[poc_idx + 1]) / 2
    poc_vol = profile[poc_idx]

    # Value Area — итеративное расширение от POC
    target = total * va_percentage
    current_vol = poc_vol
    upper = poc_idx
    lower = poc_idx

    while current_vol < target and (upper < len(profile) - 1 or lower > 0):
        upper_vol = profile[upper + 1] if upper < len(profile) - 1 else -np.inf
        lower_vol = profile[lower - 1] if lower > 0 else -np.inf

        if upper_vol >= lower_vol and upper < len(profile) - 1:
            upper += 1
            current_vol += upper_vol
        elif lower > 0:
            lower -= 1
            current_vol += lower_vol
        else:
            break

    vah = (bins[upper + 1] + bins[upper]) / 2 if upper < len(bins) - 1 else bins[-1]
    val = (bins[lower] + bins[lower + 1]) / 2 if lower >= 0 else bins[0]

    return {
        'poc_price': poc_price,
        'poc_volume': poc_vol,
        'vah': vah,
        'val': val,
        'total_volume': total,
        'va_volume': current_vol,
        # Опционально для отладки / визуализации
        # 'price_bins': bins,
        # 'volume_profile': profile
    }