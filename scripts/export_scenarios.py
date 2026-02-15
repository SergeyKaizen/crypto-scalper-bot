# scripts/export_scenarios.py
"""
Принудительная выгрузка статистики сценариев (по ТЗ).

Запуск:
python scripts/export_scenarios.py [--output путь_к_csv]

Создаёт файл scenario_stats.csv (или указанный) с колонками:
- scenario_key
- count
- wins
- winrate
- weight
- cluster_id (если используется кластеризация)

Файл готов к импорту в Google Sheets.
"""

import argparse
from src.core.config import load_config
from src.model.scenario_tracker import ScenarioTracker
from src.utils.logger import get_logger

logger = get_logger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Выгрузка статистики сценариев")
    parser.add_argument("--output", type=str, default="scenario_stats.csv",
                        help="Путь к выходному CSV-файлу")
    args = parser.parse_args()

    config = load_config()
    tracker = ScenarioTracker(config)
    
    logger.info(f"Запуск экспорта статистики сценариев → {args.output}")
    tracker.export_statistics(args.output)
    logger.info("Выгрузка завершена. Файл готов к импорту в Google Sheets.")

if __name__ == "__main__":
    main()