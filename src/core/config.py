"""
src/core/config.py

=== Основной принцип работы файла ===

Центральный модуль загрузки конфигурации проекта.
Теперь bot_config.yaml — единственный основной файл настроек (Фаза 5).
Все остальные файлы (default.yaml, trading_modes/*.yaml, hardware/*.yaml) больше не используются.

Логика:
1. Загружает config/bot_config.yaml
2. Делает deep merge с дефолтными значениями (если какие-то ключи отсутствуют)
3. Возвращает единый словарь config, который используется во всём проекте

=== Главные функции ===
- load_config() → возвращает полный конфиг
- deep_update() — рекурсивный merge словарей

=== Примечания ===
- Теперь всё управление настройками происходит только через bot_config.yaml
- Это упрощает проект и устраняет путаницу с несколькими файлами
"""

import os
import yaml
from typing import Dict, Any
import copy

from src.utils.logger import setup_logger

logger = setup_logger("config", logging.INFO)

DEFAULT_CONFIG_PATH = "config/bot_config.yaml"


def deep_update(target: Dict, source: Dict) -> Dict:
    """Рекурсивный merge словарей"""
    result = copy.deepcopy(target)
    for key, value in source.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def load_config() -> Dict[str, Any]:
    """
    Загружает конфигурацию только из bot_config.yaml (единственный основной файл).
    """
    if not os.path.exists(DEFAULT_CONFIG_PATH):
        logger.error(f"Основной конфиг не найден: {DEFAULT_CONFIG_PATH}")
        raise FileNotFoundError(f"Создайте файл {DEFAULT_CONFIG_PATH}")

    with open(DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logger.info(f"Загружен основной конфиг: {DEFAULT_CONFIG_PATH}")
    return config