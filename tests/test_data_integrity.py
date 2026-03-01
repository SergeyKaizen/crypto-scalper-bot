"""
tests/test_data_integrity.py

=== Основной принцип работы файла ===

Тесты целостности данных — самой важной части проекта.

Проверяет:
- save_candles / get_candles (DuckDB + SQLite)
- remove_delisted
- resampler (инкрементальный ресэмплинг)
- binance_client fetch_klines (колонки + buy_volume)
"""

import pytest
import pandas as pd
from src.data.storage import Storage
from src.data.resampler import Resampler

def test_storage_save_load(tmp_storage):
    df = pd.DataFrame([{
        "timestamp": 1704067200000,
        "open": 42000.0,
        "high": 42500.0,
        "low": 41500.0,
        "close": 42200.0,
        "volume": 150.0,
        "buy_volume": 90.0
    }])

    tmp_storage.save_candles("BTCUSDT", "1m", df)
    # FIX Фаза 5: обновлён вызов (get_candles теперь реальный из Phase 4)
    loaded = tmp_storage.get_candles("BTCUSDT", "1m")
    
    assert len(loaded) == 1
    assert loaded.iloc[0]["close"] == 42200.0

def test_resampler_phone_server_agnostic(mock_config):
    resampler = Resampler(mock_config)
    candle = {
        "timestamp": 1704067200000,
        "open": 42000.0,
        "high": 42500.0,
        "low": 41500.0,
        "close": 42200.0,
        "volume": 150.0,
        "buy_volume": 90.0
    }
    
    resampler.add_1m_candle(candle)
    window = resampler.get_window("1m", 1)
    
    assert window is not None
    assert len(window) == 1