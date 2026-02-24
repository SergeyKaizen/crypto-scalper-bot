"""
src/core/config.py

=== Основной принцип работы файла ===

Центральный модуль конфигурации всего проекта.
Загружает и объединяет настройки из нескольких источников в следующем порядке приоритета:
1. override_path (если указан)
2. bot_config.yaml (личные переопределения пользователя)
3. trading_modes/{mode}.yaml (режим торговли: aggressive/balanced/conservative/custom)
4. hardware/{hardware_mode}.yaml (colab / server — переключается вручную)
5. default.yaml (базовые значения по умолчанию)

Все параметры доступны через load_config() — единая точка входа для всего кода.

=== Ключевые разделы конфигурации ===
- paths — пути к данным, моделям, логам, экспорту
- binance — API ключи (никогда не коммитить!)
- timeframes, seq_len, windows — таймфреймы и окна анализа
- hardware — параметры железа (max_workers, model_size)
- trading — режим торговли, вероятности, риск на сделку, комиссия
- model — параметры нейросети (hidden_size, dropout, retrain_interval_days, n_features, num_clusters, cluster_embedding_dim)
- filter — фильтры монет для whitelist и бэктеста
- va — параметры Value Area (профиль объёма)
- consensus — мульти-TF консенсус и VA-подтверждение
- clipping_bounds — ограничения на выбросы в фичах
- logging — уровень логов и файл
- backtest — параметры бэктеста и walk-forward

=== Важные моменты ===
- hardware_mode переключается вручную в bot_config.yaml ("colab" или "server")
- n_features — placeholder, перезаписывается trainer'ом после первого запуска
- retrain_interval_days = 7 — еженедельный переобуч per-TF
- va_confirm_enabled = false — модель учится сама, без жёсткого фильтра VA
- walk_forward: false — по умолчанию выключен (включается для валидации)
- num_clusters / cluster_embedding_dim — для интеграции кластеров в предикт
- deep_update — глубокое слияние словарей (поддерживает вложенные структуры)
- @lru_cache — конфиг кэшируется после первого вызова (ускоряет live_loop)
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
    Загрузка и слияние всей конфигурации.
    
    Приоритет источников (от высшего к низшему):
    1. override_path (если передан)
    2. bot_config.yaml (личные настройки пользователя)
    3. trading_modes/{mode}.yaml (режим торговли)
    4. hardware/{hardware_mode}.yaml (железо: colab/server)
    5. default.yaml (базовые значения)
    
    Возвращает единый словарь config.
    """
    config = {}

    # 1. Базовые значения из default.yaml
    if DEFAULT_CONFIG_PATH.exists():
        with open(DEFAULT_CONFIG_PATH, 'r', encoding='utf-8') as f:
            config.update(yaml.safe_load(f) or {})

    # 2. Hardware — переключается вручную через hardware_mode в bot_config
    hardware_mode = config.get('hardware_mode', 'colab')  # дефолт — colab
    hardware_file = BASE_DIR / 'config' / 'hardware' / f'{hardware_mode}.yaml'
    if hardware_file.exists():
        with open(hardware_file, 'r', encoding='utf-8') as f:
            config['hardware'] = yaml.safe_load(f)
    else:
        logger.warning(f"Файл hardware/{hardware_mode}.yaml не найден — используются дефолтные значения")

    # 3. Режим торговли (aggressive / balanced / conservative / custom)
    mode = config.get('trading', {}).get('mode', 'balanced')
    mode_file = BASE_DIR / 'config' / 'trading_modes' / f'{mode}.yaml'
    if mode_file.exists():
        with open(mode_file, 'r', encoding='utf-8') as f:
            mode_cfg = yaml.safe_load(f)
            config.setdefault('trading', {}).update(mode_cfg)

    # 4. Личные переопределения (самый высокий приоритет)
    bot_config_path = BASE_DIR / 'config' / 'bot_config.yaml'
    if bot_config_path.exists():
        with open(bot_config_path, 'r', encoding='utf-8') as f:
            bot_cfg = yaml.safe_load(f)
            deep_update(config, bot_cfg)

    # 5. Override path — для тестов, CLI или экспериментов
    if override_path and Path(override_path).exists():
        with open(override_path, 'r', encoding='utf-8') as f:
            deep_update(config, yaml.safe_load(f))

    # 6. Динамические / дефолтные значения (могут перезаписываться в trainer)
    config.setdefault('n_features', 128)                # будет перезаписано
    config.setdefault('retrain_interval_days', 7)
    config.setdefault('min_tf_consensus', 2)
    config.setdefault('va_confirm_enabled', False)

    # Параметры для walk-forward
    config.setdefault('walk_forward', False)                    # включить режим WFO
    config.setdefault('walk_forward_periods', 8)                # количество сегментов
    config.setdefault('walk_forward_segment_months', 3)         # длина одного сегмента в месяцах
    config.setdefault('walk_forward_retrain_frequency', 'weekly')  # частота retrain в in-sample

    # Параметры для сценариев и кластеризации
    config.setdefault('num_clusters', 50)                       # максимальное количество кластеров в HDBSCAN
    config.setdefault('cluster_embedding_dim', 16)              # размерность эмбеддинга кластера в модели

    return config


def deep_update(target: dict, source: dict):
    """
    Глубокое рекурсивное обновление словарей.
    Позволяет переопределять вложенные структуры без потери данных.
    """
    for key, value in source.items():
        if key in target and isinstance(target[key], dict) and isinstance(value, dict):
            deep_update(target[key], value)
        else:
            target[key] = value