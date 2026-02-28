"""
tests/conftest.py

=== Основной принцип работы файла ===

Центральный файл фикстур pytest для всех тестов проекта.
Содержит общие мок-объекты и настройки, которые используются во всех тестах:
- mock_config — загружает тестовый конфиг (balanced по умолчанию)
- mock_storage — временное хранилище в памяти (tmp_path)
- mock_binance_client — мок BinanceClient без реальных запросов
- mock_logger — отключает реальное логирование во время тестов

Это позволяет тестам работать быстро, без интернета и без изменения реальных данных.

=== Как работает ===
pytest автоматически подхватывает все фикстуры из conftest.py
и передаёт их в тестовые функции по имени параметра.
"""

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

@pytest.fixture
def mock_config():
    """Фикстура: возвращает минимальный конфиг для тестов"""
    return {
        "trading_mode": "balanced",
        "hardware": {"max_workers": 2},
        "timeframes": ["1m", "5m"],
        "seq_len": 100,
        "trading": {
            "risk_pct": 0.01,
            "min_prob": 0.65
        }
    }


@pytest.fixture
def tmp_storage(tmp_path):
    """Фикстура: временное хранилище в /tmp для тестов storage"""
    from src.data.storage import Storage
    storage = Storage()
    storage.data_dir = str(tmp_path)
    return storage


@pytest.fixture
def mock_binance_client():
    """Фикстура: мок BinanceClient без реальных API-запросов"""
    client = MagicMock()
    client.update_markets_list.return_value = ["BTCUSDT", "ETHUSDT"]
    client.fetch_klines.return_value = None  # будем мокать в конкретных тестах
    return client


@pytest.fixture(autouse=True)
def disable_logging():
    """Автоматически отключает реальное логирование во всех тестах"""
    with patch("src.utils.logger.setup_logger") as mock_logger:
        mock_logger.return_value = MagicMock()
        yield