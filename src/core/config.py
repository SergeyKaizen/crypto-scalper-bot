"""
src/core/config.py

Центральный модуль конфигурации проекта.
Загружает и объединяет настройки из нескольких источников в порядке приоритета:
1. override_path (если указан)
2. bot_config.yaml (личные переопределения)
3. trading_modes/{mode}.yaml (режим торговли)
4. hardware/{hardware_mode}.yaml (colab / server)
5. default.yaml (базовые значения)

Все параметры доступны через load_config() — единая точка входа.

=== Важные моменты ===
- hardware_mode переключается вручную в bot_config.yaml
- n_features — placeholder, перезаписывается trainer'ом
- retrain_interval_days = 7 — еженедельный переобуч per-TF
- deep_update — глубокое слияние словарей
- @lru_cache — кэширование для ускорения live_loop
"""

import os
import yaml
from functools import lru_cache
from pathlib import Path
from typing import Optional, Dict, Any

BASE_DIR = Path(__file__).parent.parent.parent
DEFAULT_CONFIG_PATH = BASE_DIR / 'config' / 'default.yaml'


@lru_cache(maxsize=1)
def load_config(
    hardware: Optional[str] = None,
    mode: Optional[str] = None,
    hardware_profile: Optional[str] = None,
    override_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Загрузка и слияние всей конфигурации.
    Теперь принимает явные параметры hardware, mode, hardware_profile из CLI/скриптов.
    """
    config: Dict[str, Any] = {}

    # 1. default.yaml — всегда базовый слой
    if DEFAULT_CONFIG_PATH.exists():
        with open(DEFAULT_CONFIG_PATH, 'r', encoding='utf-8') as f:
            config.update(yaml.safe_load(f) or {})

    # 2. hardware — приоритет: параметр > bot_config.yaml > default
    hardware_mode = (
        hardware_profile
        or hardware
        or config.get('hardware_mode', 'colab')
    )
    hardware_file = BASE_DIR / 'config' / 'hardware' / f'{hardware_mode}.yaml'
    if hardware_file.exists():
        with open(hardware_file, 'r', encoding='utf-8') as f:
            config['hardware'] = yaml.safe_load(f) or {}
    else:
        config.setdefault('hardware', {})

    # 3. режим торговли — приоритет: параметр > bot_config.yaml > default
    trading_mode = (
        mode
        or config.get('trading', {}).get('mode', 'balanced')
    )
    mode_file = BASE_DIR / 'config' / 'trading_modes' / f'{trading_mode}.yaml'
    if mode_file.exists():
        with open(mode_file, 'r', encoding='utf-8') as f:
            mode_cfg = yaml.safe_load(f) or {}
            config.setdefault('trading', {}).update(mode_cfg)

    # 4. bot_config.yaml — переопределяет trading и hardware_mode
    bot_config_path = BASE_DIR / 'config' / 'bot_config.yaml'
    if bot_config_path.exists():
        with open(bot_config_path, 'r', encoding='utf-8') as f:
            bot_cfg = yaml.safe_load(f) or {}
            deep_update(config, bot_cfg)

    # 5. override (для тестов/CLI) — самый высокий приоритет
    if override_path and Path(override_path).exists():
        with open(override_path, 'r', encoding='utf-8') as f:
            deep_update(config, yaml.safe_load(f) or {})

    # Динамические дефолты (оставляем как было)
    config.setdefault('n_features', 128)
    config.setdefault('retrain_interval_days', 7)
    config.setdefault('min_tf_consensus', 2)
    config.setdefault('va_confirm_enabled', False)

    # Параметры для scenario_tracker (новые)
    config.setdefault('scenario_tracker', {})
    config['scenario_tracker'].setdefault('bayesian_prior_wins', 1.0)
    config['scenario_tracker'].setdefault('bayesian_prior_losses', 3.0)
    config['scenario_tracker'].setdefault('decay_half_life_days', 90.0)
    config['scenario_tracker'].setdefault('decay_enabled', True)
    config['scenario_tracker'].setdefault('regime', {})
    config['scenario_tracker']['regime'].setdefault('delta_norm_threshold', 0.65)

    return config


def deep_update(target: dict, source: dict):
    """Глубокое рекурсивное обновление словарей"""
    for key, value in source.items():
        if key in target and isinstance(target[key], dict) and isinstance(value, dict):
            deep_update(target[key], value)
        else:
            target[key] = value