"""
src/core/config.py

=== Основной принцип работы файла ===

Центральный модуль загрузки конфигурации проекта.
Теперь bot_config.yaml — единственный основной файл настроек.
Все остальные файлы (default.yaml, trading_modes/*.yaml, hardware/*.yaml) больше не используются и игнорируются.

Логика:
1. Загружает только config/bot_config.yaml
2. Возвращает единый словарь config, который используется во всём проекте

=== Примечания ===
- Это единственный источник правды для всех настроек проекта
- Упрощает поддержку и устраняет путаницу
"""

import os
import yaml
from typing import Dict, Any

from src.utils.logger import setup_logger

logger = setup_logger("config", logging.INFO)

DEFAULT_CONFIG_PATH = "config/bot_config.yaml"


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