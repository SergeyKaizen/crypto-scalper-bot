"""
src/core/config.py

=== Основной принцип работы файла ===

Центральный файл конфигурации всего проекта.
Загружает дефолтные значения из default.yaml, переопределяет из bot_config.yaml,
trading_modes/*.yaml и hardware/*.yaml в зависимости от окружения.

Все параметры доступны через load_config() — единая точка входа.

=== Структура конфига (все ключевые разделы) ===

- paths: директории данных/моделей
- binance: API ключи
- timeframes: ['1m', '3m', '5m', '10m', '15m']
- seq_len: 100
- windows: [24, 50, 74, 100]
- hardware: max_workers, model_size ('medium'/'large')
- trading: mode ('real'/'virtual'), min_prob, risk_pct, dropout и т.д.
- model: hidden_size, retrain_interval_days: 7, n_features (рассчитывается)
- filter: min_age_months, min_pr, min_trades
- va: va_percentage: 0.70, price_bin_step_pct: 0.002
- consensus: min_tf_consensus: 2, va_confirm_enabled: false
- clipping_bounds: dict с порогами для % изменений
- logging: level и т.д.

=== Примечания ===
- load_config() кэширует конфиг после первого вызова
- Поддержка override_path для тестов
- Все новые параметры из обсуждений добавлены
"""

import os
import yaml
from functools import lru_cache
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.parent
DEFAULT_CONFIG_PATH = BASE_DIR / 'config' / 'default.yaml'

@lru_cache(maxsize=1)
def load_config(override_path=None):
    """
    Загрузка и слияние конфигурации.
    Приоритет: override_path > bot_config.yaml > trading_modes/*.yaml > hardware/*.yaml > default.yaml
    """
    config = {}

    # 1. Default
    if DEFAULT_CONFIG_PATH.exists():
        with open(DEFAULT_CONFIG_PATH, 'r', encoding='utf-8') as f:
            config.update(yaml.safe_load(f) or {})

    # 2. Hardware (Colab/server)
    hardware_file = override_path or (BASE_DIR / 'config' / 'hardware' / 'colab.yaml')  # по умолчанию Colab
    if hardware_file.exists():
        with open(hardware_file, 'r', encoding='utf-8') as f:
            config['hardware'] = yaml.safe_load(f)

    # 3. Trading mode (balanced/aggressive/custom и т.д.)
    mode = config.get('trading', {}).get('mode', 'balanced')
    mode_file = BASE_DIR / 'config' / 'trading_modes' / f'{mode}.yaml'
    if mode_file.exists():
        with open(mode_file, 'r', encoding='utf-8') as f:
            mode_cfg = yaml.safe_load(f)
            if 'trading' not in config:
                config['trading'] = {}
            config['trading'].update(mode_cfg)

    # 4. Bot config (переопределение всего)
    bot_config_path = BASE_DIR / 'config' / 'bot_config.yaml'
    if bot_config_path.exists():
        with open(bot_config_path, 'r', encoding='utf-8') as f:
            bot_cfg = yaml.safe_load(f)
            deep_update(config, bot_cfg)

    # 5. Override path (для тестов или CLI)
    if override_path and Path(override_path).exists():
        with open(override_path, 'r', encoding='utf-8') as f:
            override_cfg = yaml.safe_load(f)
            deep_update(config, override_cfg)

    # 6. Динамические/расчётные параметры
    config['n_features'] = calculate_n_features(config)  # placeholder — реализуем ниже
    config['retrain_interval_days'] = config.get('retrain_interval_days', 7)
    config['min_tf_consensus'] = config.get('min_tf_consensus', 2)
    config['va_confirm_enabled'] = config.get('va_confirm_enabled', False)

    return config


def deep_update(target: dict, source: dict):
    """Глубокое слияние словарей"""
    for key, value in source.items():
        if isinstance(value, dict) and key in target and isinstance(target[key], dict):
            deep_update(target[key], value)
        else:
            target[key] = value


def calculate_n_features(config):
    """
    Динамический подсчёт количества фич (пример)
    В реальности — смотреть на compute_features и считать колонки
    """
    # Базовые 12 + half comparison (~10) + delta VA (~10) + sequential (~25) + quiet_streak (1)
    return 12 + 10 + 10 + 25 + 1  # ~58
    # Лучше реализовать как:
    # dummy_feats = compute_features(pd.DataFrame())
    # return sum(len(f) for f in dummy_feats.values()) + 1  # + quiet_streak