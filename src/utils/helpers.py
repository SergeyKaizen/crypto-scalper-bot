# src/utils/helpers.py
"""
Сборник вспомогательных утилит, которые используются по всему проекту.

Функции:
- timestamp_to_datetime() — преобразование ms timestamp в datetime
- calculate_commission() — комиссия Binance (taker/maker)
- apply_slippage() — добавление slippage в симуляции
- check_phone_safety() — температура и CPU нагрузка (Android)
- safe_load_yaml() — безопасная загрузка yaml с обработкой ошибок
- get_leverage_for_symbol() — получение плеча монеты (кэшируется)
- dict_to_polars() — преобразование dict/list в Polars DataFrame
- round_to_precision() — округление цены/количества по правилам Binance
- log_memory_usage() — логирование использования памяти (особенно на Colab)

Все функции — чистые, без side-effects (кроме логгинга)
"""

import logging
import time
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Union, Optional

import yaml
import psutil
import platform

logger = logging.getLogger(__name__)


def timestamp_to_datetime(ts_ms: Union[int, float]) -> datetime:
    """Преобразование миллисекундного timestamp в datetime"""
    return datetime.fromtimestamp(ts_ms / 1000)


def calculate_commission(trade_value_usdt: float, is_maker: bool = False) -> float:
    """Расчёт комиссии Binance Futures (USDT-M)"""
    rate = BINANCE_FUTURES_MAKER_FEE if is_maker else BINANCE_FUTURES_TAKER_FEE
    return trade_value_usdt * float(rate)


def apply_slippage(price: float, side: str, slippage_pct: float = 0.10) -> float:
    """Добавление slippage (для виртуальной симуляции)"""
    slippage = price * (slippage_pct / 100)
    if side.lower() == "buy":
        return price + slippage  # Покупаем дороже
    else:
        return price - slippage  # Продаём дешевле


def check_phone_safety() -> Dict[str, Union[float, bool]]:
    """
    Проверка температуры и нагрузки (только для Android/телефона)
    Возвращает:
    {
        "cpu_percent": float,
        "temperature_c": float or None,
        "is_safe": bool
    }
    """
    if platform.system().lower() != "linux" or "android" not in platform.release().lower():
        return {"cpu_percent": 0.0, "temperature_c": None, "is_safe": True}

    try:
        cpu_percent = psutil.cpu_percent(interval=0.5)

        # Температура — Android (через sysfs, обычно thermal_zone*)
        temp_c = None
        thermal_paths = Path("/sys/class/thermal").glob("thermal_zone*/temp")
        for path in thermal_paths:
            try:
                with open(path, "r") as f:
                    temp = int(f.read().strip()) / 1000
                    temp_c = temp
                    break
            except:
                continue

        is_safe = cpu_percent < 70.0 and (temp_c is None or temp_c < 45.0)

        return {
            "cpu_percent": cpu_percent,
            "temperature_c": temp_c,
            "is_safe": is_safe
        }
    except Exception as e:
        logger.warning("Не удалось проверить температуру/нагрузку телефона: %s", e)
        return {"cpu_percent": 0.0, "temperature_c": None, "is_safe": True}


def safe_load_yaml(file_path: str) -> Dict:
    """Безопасная загрузка yaml-файла"""
    path = Path(file_path)
    if not path.exists():
        logger.warning("YAML-файл не найден: %s", file_path)
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data or {}
    except Exception as e:
        logger.error("Ошибка загрузки YAML %s: %s", file_path, e)
        return {}


def round_to_precision(value: float, precision: int = 8) -> float:
    """Округление до нужной точности (для цены/количества)"""
    return round(value, precision)


def get_leverage_for_symbol(symbol: str, client) -> int:
    """Получение максимального плеча для монеты (кэшируется)"""
    # В реальном коде — fetch_markets() и кэш
    # Здесь — упрощённо
    return 20  # Placeholder — в реальном коде 5–125x в зависимости от монеты


def log_memory_usage():
    """Логирование использования памяти (особенно полезно на Colab)"""
    process = psutil.Process()
    mem = process.memory_info().rss / (1024 ** 2)  # в МБ
    logger.debug("Memory usage: %.1f MB", mem)