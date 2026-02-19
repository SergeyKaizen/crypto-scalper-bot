# scripts/export_scenarios.py
"""
Скрипт для принудительной выгрузки статистики бинарных сценариев (по ТЗ).

Что делает:
- Загружает текущую статистику сценариев из ScenarioTracker
- Формирует таблицу: scenario_key, count, wins, winrate, weight, cluster_id
- Сохраняет в CSV (по умолчанию scenario_stats_YYYYMMDD_HHMMSS.csv)
- Готов к импорту в Google Sheets (Ctrl+V → Импорт → CSV)
- Можно указать произвольное имя файла через аргумент --output
- Выводит топ-10 сценариев в консоль для быстрого просмотра

Запуск:
python scripts/export_scenarios.py
python scripts/export_scenarios.py --output my_scenarios_2026.csv
"""

import argparse
import os
from datetime import datetime

from src.core.config import load_config
from src.model.scenario_tracker import ScenarioTracker
from src.utils.logger import get_logger

logger = get_logger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Выгрузка статистики бинарных сценариев в CSV")
    parser.add_argument("--output", type=str, default=None,
                        help="Путь к CSV-файлу (по умолчанию: scenario_stats_YYYYMMDD_HHMMSS.csv)")
    args = parser.parse_args()

    config = load_config()
    tracker = ScenarioTracker(config)

    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"scenario_stats_{timestamp}.csv"

    # Экспорт
    logger.info(f"Запуск экспорта статистики сценариев → {args.output}")
    tracker.export_statistics(args.output)
    logger.info(f"Выгрузка завершена. Файл готов к загрузке в Google Sheets: {os.path.abspath(args.output)}")

    # Дополнительно: вывод топ-10 сценариев в консоль
    top = tracker.get_top_scenarios(top_n=10)
    if not top.is_empty():
        print("\nТоп-10 сценариев по весу:")
        print(top.to_pandas().to_string(index=False))


if __name__ == "__main__":
    main()