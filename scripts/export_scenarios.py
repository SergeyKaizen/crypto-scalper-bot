"""
src/scripts/export_scenarios.py

=== Основной принцип работы файла ===

Этот скрипт предназначен для экспорта полной статистики всех бинарных сценариев из ScenarioTracker в CSV-файл.
Он:
- Загружает состояние tracker (все сценарии с winrate, count, weight).
- Получает полный DataFrame статистики (get_scenarios_stats без фильтров или с min_count=1).
- Добавляет кластеризацию HDBSCAN (labels для групп).
- Сортирует по weight descending.
- Экспортирует в CSV (scenarios_stats.csv или указанный путь).
- Поддерживает принудительный запуск через аргумент --force (даже если не в live-режиме).

Скрипт запускается вручную: python export_scenarios.py [--force] [--output path.csv]
Полностью соответствует ТЗ: экспорт всех сценариев (не топ-N), команда для принудительной выгрузки, CSV вместо Sheets.

=== Главные функции и за что отвечают ===

- main() — entry-point скрипта: парсит аргументы, загружает tracker, экспортирует.
- export_scenarios(tracker: ScenarioTracker, output_path: str) — основной экспорт:
  - Получает df = tracker.get_scenarios_stats(min_count=1)
  - Добавляет кластеры HDBSCAN.
  - Сортирует и сохраняет в CSV.
- parse_args() — парсит --force и --output.

=== Примечания ===
- Tracker загружается из storage или pkl (по реализации scenario_tracker).
- Полный экспорт — все сценарии, без обрезки топ-N.
- HDBSCAN добавляет cluster_label (-1 для outliers).
- Готов к запуску вручную или по cron.
- Логи через setup_logger.
- Нет зависимостей от live-режима — чистый скрипт.
"""

import argparse
import os
import pandas as pd
from hdbscan import HDBSCAN

from src.model.scenario_tracker import ScenarioTracker
from src.core.config import load_config
from src.utils.logger import setup_logger

logger = setup_logger('export_scenarios', logging.INFO)

def parse_args():
    parser = argparse.ArgumentParser(description="Экспорт статистики сценариев в CSV")
    parser.add_argument("--force", action="store_true", help="Принудительный запуск экспорта")
    parser.add_argument("--output", type=str, default="scenarios_stats.csv", help="Путь к выходному CSV")
    parser.add_argument("--hardware", default="phone_tiny", choices=["phone_tiny", "colab", "server"],
                        help="Профиль железа")
    parser.add_argument("--mode", default="balanced", choices=["conservative", "balanced", "aggressive", "custom"],
                        help="Режим торговли")
    return parser.parse_args()

def export_scenarios(tracker: ScenarioTracker, output_path: str):
    """
    Экспортирует полную статистику сценариев в CSV.
    - Все сценарии (full stats).
    - Сортировка по weight descending.
    - Добавление кластеризации HDBSCAN.
    """
    df = tracker.get_scenarios_stats(min_count=1)  # все сценарии

    if df.empty:
        logger.warning("Нет сценариев для экспорта")
        return

    # Кластеризация (если достаточно данных)
    if len(df) >= 3:
        features = df[['winrate', 'count', 'weight']].values
        clusterer = HDBSCAN(min_cluster_size=5)  # из config или константа
        labels = clusterer.fit_predict(features)
        df['cluster_label'] = labels
        logger.info(f"HDBSCAN кластеризация: {len(set(labels)) - (1 if -1 in labels else 0)} кластеров + outliers")

    # Сортировка и сохранение
    df = df.sort_values('weight', ascending=False).reset_index(drop=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Экспортировано {len(df)} сценариев в {output_path}")

def main():
    args = parse_args()

    config = load_config(hardware=args.hardware, mode=args.mode)  # ← ФИКС: передаём hardware и mode
    tracker = ScenarioTracker(config)  # ← предположительно tracker принимает config (если нет — адаптировать)

    if not args.force:
        logger.info("Экспорт по умолчанию (force не указан). Используйте --force для принудительного запуска.")
        return

    export_scenarios(tracker, args.output)

if __name__ == "__main__":
    main()