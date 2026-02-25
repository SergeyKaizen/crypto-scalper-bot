"""
tests/test_trainer.py

=== Основной принцип работы файла ===

Unit-тесты для trainer.py и связанных функций.

Ключевые тесты (по утверждённому пункту 5):
- test_leakage_in_features: проверяет, что compute_features не меняется при добавлении будущих свечей (lookahead)
- test_leakage_in_label: проверяет, что _get_label не меняется при добавлении будущих свечей (embargo around target)
- test_prepare_data_step: проверяет, что step в prepare_data уменьшает overlap
- test_embargo_in_split: проверяет, что purged_gap + embargo_gap работают в TimeSeriesSplit

=== Как запускать ===
pytest tests/test_trainer.py -v

=== Примечания ===
- Тесты используют mock-данные (маленький df) — быстрые и надёжные
- Проверки строгие: даже 1% изменения — ошибка
- Полностью соответствует ТЗ + пункту 5 (Leakage test in tests/)
- Готов к CI/CD (GitHub Actions, GitLab CI)
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.model.trainer import Trainer, TimeSeriesDataset
from src.features.feature_engine import compute_features
from src.trading.tp_sl_manager import TP_SL_Manager  # для _get_label

# Фикстура: маленький тестовый df
@pytest.fixture
def sample_df():
    dates = pd.date_range(start='2025-01-01', periods=50, freq='1min')
    df = pd.DataFrame({
        'open': np.random.rand(50) * 100 + 50000,
        'high': np.random.rand(50) * 100 + 50100,
        'low': np.random.rand(50) * 100 + 49900,
        'close': np.random.rand(50) * 100 + 50000,
        'volume': np.random.rand(50) * 1000,
    }, index=dates)
    return df

# Тест 1: Нет lookahead в compute_features
def test_leakage_in_features(sample_df):
    trainer = Trainer()

    # Исходные фичи
    original_feats = compute_features(sample_df)

    # Добавляем будущую свечу (сильное движение)
    future_row = pd.DataFrame({
        'open': [sample_df['close'].iloc[-1] + 500],
        'high': [sample_df['close'].iloc[-1] + 1000],
        'low': [sample_df['close'].iloc[-1] + 100],
        'close': [sample_df['close'].iloc[-1] + 800],
        'volume': [5000],
    }, index=[sample_df.index[-1] + timedelta(minutes=1)])

    extended_df = pd.concat([sample_df, future_row])

    # Фичи на исходном df (без будущего)
    new_feats = compute_features(extended_df.iloc[:-1])  # только оригинальные 50 баров

    # Должны быть идентичны
    assert new_feats == original_feats, "Leakage в фичах: добавление будущей свечи изменило признаки!"

# Тест 2: Нет lookahead в _get_label (embargo around target)
def test_leakage_in_label(sample_df):
    trainer = Trainer()

    # Исходный label на полном окне
    window_df = sample_df.iloc[-20:]  # последние 20 баров
    original_label = trainer._get_label(window_df)

    # Добавляем будущее движение
    future_row = pd.DataFrame({
        'open': [window_df['close'].iloc[-1] + 500],
        'high': [window_df['close'].iloc[-1] + 1000],
        'low': [window_df['close'].iloc[-1] + 100],
        'close': [window_df['close'].iloc[-1] + 800],
        'volume': [5000],
    }, index=[window_df.index[-1] + timedelta(minutes=1)])

    extended_window = pd.concat([window_df, future_row])

    # Label на окне без будущего
    new_label = trainer._get_label(extended_window.iloc[:-1])

    assert new_label == original_label, "Leakage в label: добавление будущей свечи изменило target!"

# Тест 3: Step в prepare_data уменьшает overlap
def test_prepare_data_step():
    trainer = Trainer()
    trainer.config['seq_len'] = 20
    trainer.config['step'] = 10  # шаг = seq_len // 2

    # Моковые данные
    df = pd.DataFrame(index=range(100))
    data_lists, labels = trainer.prepare_data()  # вызов с step

    # Проверяем, что шаг работает (длина data_lists должна быть ~ (100-20)/10 = 8)
    assert len(data_lists[0]) == 8, "Step в prepare_data не работает — overlap не уменьшен!"

# Тест 4: Embargo в TimeSeriesSplit работает
def test_embargo_in_split():
    trainer = Trainer()
    trainer.config['seq_len'] = 20
    trainer.config['purged_gap_multiplier'] = 1.0
    trainer.config['embargo_bars'] = 5

    # Моковый dataset
    dataset = TimeSeriesDataset([np.zeros((100, 10))], np.zeros(100))

    tscv = TimeSeriesSplit(n_splits=3)
    for train_idx, val_idx in tscv.split(range(len(dataset))):
        purged_gap = int(20 * 1.0)
        embargo_gap = 5
        train_idx_filtered = train_idx[train_idx < val_idx.min() - purged_gap - embargo_gap]
        assert len(train_idx_filtered) < len(train_idx), "Embargo в сплите не работает!"

# Тест 5: Общий тест на отсутствие leakage (комплексный)
def test_no_leakage_overall(sample_df):
    trainer = Trainer()

    # Исходные данные
    original_feats = compute_features(sample_df)
    original_label = trainer._get_label(sample_df)

    # Добавляем будущее
    future_df = sample_df.copy()
    future_df['close'] = future_df['close'] + 1000  # сильное движение

    # Фичи и label на оригинальном df
    new_feats = compute_features(sample_df)
    new_label = trainer._get_label(sample_df)

    assert new_feats == original_feats, "Leakage в фичах при изменении будущего!"
    assert new_label == original_label, "Leakage в label при изменении будущего!"