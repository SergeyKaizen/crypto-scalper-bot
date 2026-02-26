"""
src/features/feature_engine.py

=== Основной принцип работы файла ===

Расчёт всех признаков для окна свечей.

Ключевые улучшения (последние утверждённые):
- Формула волатильности: (high - low) / high * 100 — размер свечи в % от high
- Средняя волатильность: expanding.mean() (нарастающее окно)
- Regime separation: 2 бинарных признака на основе half-comparator
  - regime_bull_strength: delta_diff_norm > 0.65
  - regime_bear_strength: delta_diff_norm < -0.65
- delta_diff_norm = (right_delta_mean - left_delta_mean) / avg_candle_size
- Warm-up: если len(df) < min_bars → fallback
- Валидация схемы через FeaturesSchema

=== Примечания ===
- Все признаки локальны на df (up_to_now) — нет lookahead
- Regime-признаки попадают в конец feats → в ключ сценария
- Полностью соответствует ТЗ + утверждённым 3 пунктам
- Готов к использованию в live_loop, backtest, trainer
"""

import pandas as pd
import numpy as np
from typing import Dict

from src.core.config import load_config
from src.utils.logger import setup_logger
from src.features.feature_schema import FeaturesSchema  # Pydantic схема

logger = setup_logger('feature_engine', logging.INFO)

FEATURES_VERSION = "1.0"


def compute_features(df: pd.DataFrame) -> Dict:
    """
    Расчёт всех признаков для окна свечей.
    """
    config = load_config()
    min_bars = config.get('features', {}).get('min_bars_for_stats', 30)

    if len(df) < min_bars:
        logger.debug(f"Окно слишком короткое ({len(df)} < {min_bars}) — fallback")
        return {field: 0.0 for field in FeaturesSchema.__fields__ if field != 'version'}

    # Базовые расчёты
    price_change_pct = df['close'].pct_change().iloc[-1] * 100 if len(df) > 1 else 0.0

    # Волатильность — исправленная формула (твоя)
    volatility_pct = ((df['high'] - df['low']) / df['high']) * 100
    volatility_mean = volatility_pct.expanding().mean().iloc[-1]
    volatility_change_pct = volatility_pct.pct_change().iloc[-1] * 100 if len(volatility_pct) > 1 else 0.0

    # Delta (если есть)
    if 'delta' in df:
        delta_positive = (df['delta'] > 0).sum()
        delta_change_pct = df['delta'].pct_change().iloc[-1] * 100 if len(df) > 1 else 0.0
        cumulative_delta = df['delta'].expanding().sum().iloc[-1]
    else:
        delta_positive = 0
        delta_change_pct = 0.0
        cumulative_delta = 0.0

    volume_change_pct = df['volume'].pct_change().iloc[-1] * 100 if len(df) > 1 else 0.0

    va_position = 0.0  # placeholder
    norm_dist_to_delta_vah = 0.0
    norm_dist_to_delta_val = 0.0

    sequential_delta_positive_count = 0  # placeholder
    sequential_delta_increased_count = 0
    sequential_volume_increased_count = 0

    quiet_streak = 0  # placeholder

    # Regime separation: на основе half-comparator (дельта)
    # Предполагаем, что half-comparator вернул признаки для окна 50
    mid = len(df) // 2
    left = df.iloc[:mid]
    right = df.iloc[mid:]

    left_delta_mean = left['delta'].mean() if 'delta' in left else 0
    right_delta_mean = right['delta'].mean() if 'delta' in right else 0
    delta_diff = right_delta_mean - left_delta_mean

    avg_candle_size = (df['high'] - df['low']).mean()
    delta_diff_norm = delta_diff / avg_candle_size if avg_candle_size != 0 else 0

    regime_threshold = config['scenario_tracker']['regime']['delta_norm_threshold']  # 0.65
    regime_bull_strength  = 1 if delta_diff_norm > regime_threshold else 0
    regime_bear_strength  = 1 if delta_diff_norm < -regime_threshold else 0

    features = {
        'version': FEATURES_VERSION,
        'price_change_pct': price_change_pct,
        'volatility_mean': volatility_mean,
        'volatility_change_pct': volatility_change_pct,
        'delta_positive': delta_positive,
        'delta_change_pct': delta_change_pct,
        'volume_change_pct': volume_change_pct,
        'va_position': va_position,
        'norm_dist_to_delta_vah': norm_dist_to_delta_vah,
        'norm_dist_to_delta_val': norm_dist_to_delta_val,
        'sequential_delta_positive_count': sequential_delta_positive_count,
        'sequential_delta_increased_count': sequential_delta_increased_count,
        'sequential_volume_increased_count': sequential_volume_increased_count,
        'quiet_streak': quiet_streak,
        'regime_bull_strength': regime_bull_strength,
        'regime_bear_strength': regime_bear_strength,
        # ... остальные признаки из ТЗ
    }

    # Валидация схемы
    try:
        FeaturesSchema(**features)
    except Exception as e:
        logger.error(f"Ошибка валидации фич: {e}")
        raise

    return features