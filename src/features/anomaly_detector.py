"""
src/features/anomaly_detector.py

=== Основной принцип работы файла ===

Детекция аномалий по ТЗ:
- C — свечная (размер свечи > средний + отклонение, но < 2× среднего)
- V — объёмная (объём > средний × 1.5)
- CV — оба условия
- Q — отсутствие аномалий (quiet)

Ключевые особенности (после всех уточнений):
- Нет жёсткого требования пересечения VA / delta VA (модель сама учится)
- Мульти-TF consensus: аномалия на младшем TF подтверждается минимум на min_tf_consensus старших TF
- Quiet streak count — количество последовательных Q (сбрасывается при любой аномалии)
- delta VA используется только как optional фича для усиления (не блокирует сигнал)

Возвращает dict[tf][window] → {'type', 'confirmed', 'consensus_count', 'quiet_streak', ...}

=== Главные функции ===

- detect_anomalies(features_by_tf: dict, current_tf: str) → dict
- _detect_single_tf(features, timeframe)
- _get_consensus_score(anomaly_type, tf_anomalies, current_tf)
- _check_candle_anomaly, _check_volume_anomaly
- _check_delta_va_confirm (опционально)

=== Примечания ===
- Consensus только для младших TF (1m, 3m, 5m) — старшие (10m, 15m) подтверждают сами себя
- Quiet streak пока глобальный — в live_loop лучше сделать dict[symbol][tf]
- Config: va_confirm_enabled (по умолчанию false), min_tf_consensus (по умолчанию 2)
"""

import numpy as np
from src.core.config import load_config

# Глобальный quiet streak (в реальности per-symbol/per-tf в live_loop)
quiet_streak = 0

TF_ORDER = ['1m', '3m', '5m', '10m', '15m']  # от младшего к старшему

def detect_anomalies(features_by_tf: dict, current_tf: str) -> dict:
    """
    Главная функция: детекция по всем TF + consensus + quiet streak
    """
    config = load_config()
    min_consensus = config.get('min_tf_consensus', 2)
    va_confirm_enabled = config.get('va_confirm_enabled', False)

    # Детекция по каждому TF отдельно
    tf_anomalies = {}
    for tf in TF_ORDER:
        if tf in features_by_tf:
            tf_anomalies[tf] = _detect_single_tf(features_by_tf[tf], tf)

    # Результат только для текущего TF
    result = {}
    current_anomalies = tf_anomalies.get(current_tf, {})

    for window, single_anom in current_anomalies.items():
        anomaly_type = single_anom['type']

        # Quiet streak update
        global quiet_streak
        if anomaly_type == 'Q':
            quiet_streak += 1
        else:
            quiet_streak = 0

        # Consensus
        consensus_count = _get_consensus_score(anomaly_type, tf_anomalies, current_tf)

        confirmed = (anomaly_type != 'Q') and (consensus_count >= min_consensus)

        # Опциональное VA/delta VA усиление (не блокирующее)
        va_confirm = True
        if va_confirm_enabled:
            va_confirm = _check_delta_va_confirm(single_anom['details'], single_anom.get('half_changes', {}))

        result[window] = {
            'type': anomaly_type,
            'confirmed': confirmed and va_confirm,
            'consensus_count': consensus_count,
            'quiet_streak': quiet_streak,
            'delta_confirm': va_confirm,
            'details': single_anom.get('details', {})
        }

    return {current_tf: result}


def _detect_single_tf(features: dict, timeframe: str) -> dict:
    """Детекция аномалий на одном TF по всем окнам"""
    anomalies = {}
    for window, w_feats in features.items():
        half_feats = w_feats.get('half_changes', {})

        candle_ok = _check_candle_anomaly(w_feats, half_feats)
        volume_ok = _check_volume_anomaly(w_feats, half_feats)

        if candle_ok and volume_ok:
            anomaly_type = 'CV'
        elif candle_ok:
            anomaly_type = 'C'
        elif volume_ok:
            anomaly_type = 'V'
        else:
            anomaly_type = 'Q'

        anomalies[window] = {
            'type': anomaly_type,
            'details': {
                'volatility_anom': w_feats.get('volatility_increased', 0),
                'delta_positive': w_feats.get('delta_positive', 0),
                'delta_increased': w_feats.get('delta_increased', 0),
            },
            'half_changes': half_feats
        }

    return anomalies


def _check_candle_anomaly(w_feats: dict, half_feats: dict) -> bool:
    """Свечная аномалия по ТЗ (без жёсткого VA)"""
    volatility_mean = w_feats.get('volatility_mean', 0)
    candle_size_pct = (w_feats['high'].iloc[-1] - w_feats['low'].iloc[-1]) / w_feats['close'].iloc[-1] * 100

    is_anom_size = candle_size_pct > volatility_mean * 1.0

    # Игнор сверхбольших свечей (по ТЗ)
    if candle_size_pct > volatility_mean * 2.0:
        return False

    return is_anom_size


def _check_volume_anomaly(w_feats: dict, half_feats: dict) -> bool:
    """Объёмная аномалия по ТЗ"""
    volume_mean = w_feats.get('volume_mean', 0)
    last_volume = w_feats['volume'].iloc[-1]

    return last_volume > volume_mean * 1.5


def _get_consensus_score(anomaly_type: str, tf_anomalies: dict, current_tf: str) -> int:
    """
    Подсчёт подтверждений на старших TF
    """
    if anomaly_type == 'Q':
        return 0

    current_idx = TF_ORDER.index(current_tf)
    consensus = 1  # сам себя

    for i in range(current_idx + 1, len(TF_ORDER)):
        other_tf = TF_ORDER[i]
        if other_tf not in tf_anomalies:
            continue

        for other_anom in tf_anomalies[other_tf].values():
            if other_anom['type'] == anomaly_type:
                consensus += 1
                break  # достаточно одной аномалии на TF

    return consensus


def _check_delta_va_confirm(w_feats: dict, half_feats: dict) -> bool:
    """Опциональное усиление delta VA (не блокирующее)"""
    delta_positive = w_feats.get('delta_positive', 0) == 1
    delta_increased = w_feats.get('delta_increased', 0) == 1
    vah_crossed = half_feats.get('current_crossed_delta_vah', 0) == 1
    val_crossed = half_feats.get('current_crossed_delta_val', 0) == 1

    return (delta_positive or delta_increased) and (vah_crossed or val_crossed)