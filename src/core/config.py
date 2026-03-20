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
- После аудита: обновлена валидация под реальные ключи yaml (Phase 1)
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
    Обновлённая валидация после аудита (убраны несуществующие ключи, добавлены реальные).
    """
    if not os.path.exists(DEFAULT_CONFIG_PATH):
        logger.error(f"Основной конфиг не найден: {DEFAULT_CONFIG_PATH}")
        raise FileNotFoundError(f"Создайте файл {DEFAULT_CONFIG_PATH}")

    with open(DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # === Валидация параметров (обновлено по аудиту) ===
    trading = config.get("trading", {})
    scenario = config.get("scenario_tracker", {})

    # Обязательные параметры (должны быть > 0)
    required_positive = [
        ("risk_pct", 0.001),
        ("daily_loss_limit", 0.01),
        ("max_open_positions", 1),
        ("max_leverage", 1),
        ("min_position_size_usdt", 1.0),
        ("min_prob", 0.01),
        ("min_prob_q", 0.01),
        ("min_confidence", 0.01),
        ("tp_multiplier", 0.01),      # ← добавлено по твоему требованию
        ("sl_multiplier", 0.01),      # ← добавлено по твоему требованию
    ]

    for key, min_value in required_positive:
        value = trading.get(key, 0)
        if value <= min_value:
            raise ValueError(f"trading.{key} должен быть > {min_value} (сейчас {value})")

    # Optional параметры
    if "max_exposure_pct" in trading and trading["max_exposure_pct"] <= 0:
        raise ValueError("trading.max_exposure_pct должен быть > 0, если указан")

    # VA-confirm (default false)
    if "va_confirm_enabled" not in trading:
        trading["va_confirm_enabled"] = False

    # Scenario tracker
    if scenario.get("save_every_trades", 0) <= 0:
        raise ValueError("scenario_tracker.save_every_trades должен быть > 0")

    logger.info(f"Загружен основной конфиг: {DEFAULT_CONFIG_PATH} (валидация пройдена)")
    return config