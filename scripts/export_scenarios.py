# scripts/export_scenarios.py
"""
Скрипт выгрузки топ-сценариев после кластеризации (HDBSCAN).

Что делает:
1. Загружает конфиг
2. Инициализирует ScenarioTracker (загружает сохранённые сценарии)
3. Запускает кластеризацию (run_clustering)
4. Выбирает топ-N сценариев по весу (weight = winrate × log10(count + 1))
5. Формирует таблицу: cluster_id, winrate, count, weight, representative_vector
6. Сохраняет в CSV (scenarios_top.csv)
7. (опционально) выгружает в Google Sheets (требует gspread + credentials)

Запуск:
    python scripts/export_scenarios.py --top 20 --hardware server
    python scripts/export_scenarios.py --export-to-sheets

Аргументы:
    --top: сколько сценариев выгрузить (default 20)
    --hardware: phone_tiny / colab / server (для конфига)
    --export-to-sheets: выгрузить в Google Sheets (нужны credentials)
"""

import argparse
import logging
from pathlib import Path

import polars as pl

from src.core.config import load_config
from src.model.scenario_tracker import ScenarioTracker
from src.utils.logger import setup_logger

setup_logger()
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Выгрузка топ-сценариев после кластеризации")
    parser.add_argument("--top", type=int, default=20, help="Сколько топ-сценариев выгрузить")
    parser.add_argument("--hardware", default="phone_tiny", choices=["phone_tiny", "colab", "server"],
                        help="Профиль железа (для конфига)")
    parser.add_argument("--export-to-sheets", action="store_true", 
                        help="Выгрузить в Google Sheets (требует gspread + credentials.json)")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.hardware)

    tracker = ScenarioTracker(config)

    logger.info("Запуск кластеризации сценариев...")
    clustering_result = tracker.run_clustering()

    if not clustering_result or not clustering_result["clusters"]:
        logger.warning("Нет кластеров для экспорта")
        return

    # Топ-N по весу
    top_clusters = clustering_result["clusters"][:args.top]

    # Формируем таблицу
    data = []
    for cluster_id, stats in top_clusters:
        row = {
            "cluster_id": cluster_id,
            "winrate": round(stats["winrate"], 4),
            "count": stats["count"],
            "weight": round(stats["weight"], 4),
            "representative_vector": ",".join(f"{x:.3f}" for x in stats["representative"])
        }
        data.append(row)

    df = pl.DataFrame(data)

    # Сохранение в CSV
    output_path = Path("scenarios_top.csv")
    df.write_csv(output_path)
    logger.info("Топ-%d сценариев сохранено в %s", args.top, output_path)

    # Опционально — Google Sheets
    if args.export_to_sheets:
        try:
            import gspread
            from oauth2client.service_account import ServiceAccountCredentials

            scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
            creds = ServiceAccountCredentials.from_json_keyfile_name("credentials.json", scope)
            client = gspread.authorize(creds)

            sheet = client.open("CryptoScalper_Scenarios").sheet1
            sheet.clear()

            # Заголовки
            headers = list(data[0].keys())
            sheet.append_row(headers)

            # Данные
            rows = [list(row.values()) for row in data]
            sheet.append_rows(rows)

            logger.info("Топ-сценарии выгружены в Google Sheets: CryptoScalper_Scenarios")
        except Exception as e:
            logger.error("Ошибка выгрузки в Google Sheets: %s", e)
            logger.info("Убедитесь, что credentials.json существует и gspread установлен")


if __name__ == "__main__":
    main()