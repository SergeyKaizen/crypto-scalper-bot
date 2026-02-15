# src/model/scenario_tracker.py
"""
Отслеживание сценариев (бинарная статистика признаков) и их весов.

По ТЗ:
- Собирается статистика бинарных сценариев (каждый признак увеличился/уменьшился, true/false)
- Для каждого сценария ведётся счётчик wins/total
- Вес сценария = winrate × log(count + 1)  (логарифмический рост для редких, но надёжных сценариев)
- После обучения выводится статистика сценариев с винрейтом (для экспорта в Google Sheets)
- Автоматическое обновление каждые 10 000 свечей
- Есть команда на принудительную выгрузку (через export_scenarios.py)

Реализация:
- Хэш-сценария (tuple или frozenset бинарных флагов)
- HDBSCAN для кластеризации (если нужно — пока оставляем, как было)
- Экспорт в CSV + готовый шаблон для Google Sheets
"""

import numpy as np
import polars as pl
import os
from collections import defaultdict
from typing import Dict, Tuple, List
from src.utils.logger import get_logger

logger = get_logger(__name__)

class ScenarioTracker:
    def __init__(self, config: dict):
        self.config = config
        self.scenarios: Dict[Tuple[int, ...], Dict[str, Any]] = defaultdict(lambda: {
            "count": 0,
            "wins": 0,
            "cluster_id": -1
        })
        self.last_update_candles = 0
        self.export_dir = config.get("paths", {}).get("export_dir", "exports")
        os.makedirs(self.export_dir, exist_ok=True)

    def update_scenario(self, scenario_key: Tuple[int, ...], is_win: bool):
        """
        Обновляет статистику сценария после закрытия сделки.
        scenario_key — tuple бинарных значений (например (1,0,1,0) для candle=1, volume=0, cv=1, q=0)
        is_win — True если сделка закрыта в плюс
        """
        stats = self.scenarios[scenario_key]
        stats["count"] += 1
        if is_win:
            stats["wins"] += 1

        # Авто-экспорт каждые 500 обновлений (можно настроить)
        if stats["count"] % 500 == 0:
            self.export_statistics()

    def get_scenario_weight(self, scenario_key: Tuple[int, ...]) -> float:
        """
        Возвращает вес сценария для корректировки вероятности.
        Вес = winrate × log(count + 1)
        Если сценарий новый — возвращает 1.0 (нейтрально)
        """
        stats = self.scenarios.get(scenario_key, {"count": 0, "wins": 0})
        count = stats["count"]
        wins = stats["wins"]
        winrate = wins / count if count > 0 else 0.5  # 0.5 для новых сценариев
        weight = winrate * np.log1p(count) if count > 0 else 1.0
        return max(0.5, min(2.0, weight))  # ограничиваем диапазон 0.5–2.0

    def get_scenario_key(self, features: Dict[str, float]) -> Tuple[int, ...]:
        """
        Формирует уникальный ключ сценария из последних бинарных признаков.
        Порядок важен! Должен совпадать с порядком в feature_engine.py
        """
        binary_keys = [
            "candle_anomaly",
            "volume_anomaly",
            "cv_anomaly",
            "q_condition"
        ]
        key = tuple(1 if features.get(k, 0) > 0.5 else 0 for k in binary_keys)
        return key

    def export_statistics(self, filename: str = "scenario_stats.csv"):
        """
        Экспорт статистики всех сценариев в CSV.
        Колонки: scenario_key, count, wins, winrate, weight, cluster_id
        Готов к импорту в Google Sheets.
        """
        data = []
        for key, stats in self.scenarios.items():
            count = stats["count"]
            wins = stats["wins"]
            winrate = wins / count if count > 0 else 0.0
            weight = winrate * np.log1p(count)
            cluster = stats.get("cluster_id", -1)

            data.append({
                "scenario_key": str(key),  # tuple → строка
                "count": count,
                "wins": wins,
                "winrate": round(winrate, 4),
                "weight": round(weight, 4),
                "cluster_id": cluster
            })

        if not data:
            logger.warning("Нет сценариев для экспорта")
            return

        df = pl.DataFrame(data).sort("weight", descending=True)
        path = os.path.join(self.export_dir, filename)
        df.write_csv(path)
        logger.info(f"Экспортировано {len(data)} сценариев → {path}")

        # Дополнительно: можно добавить timestamp в имя файла
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        df.write_csv(os.path.join(self.export_dir, f"scenario_stats_{timestamp}.csv"))

    def get_top_scenarios(self, top_n: int = 20) -> pl.DataFrame:
        """Топ-N сценариев по весу"""
        stats = []
        for key, s in self.scenarios.items():
            count = s["count"]
            wins = s["wins"]
            winrate = wins / count if count > 0 else 0.0
            weight = winrate * np.log1p(count)
            stats.append({
                "scenario_key": str(key),
                "weight": weight,
                "winrate": winrate,
                "count": count
            })
        df = pl.DataFrame(stats).sort("weight", descending=True).head(top_n)
        return df


if __name__ == "__main__":
    config = load_config()
    tracker = ScenarioTracker(config)
    tracker.export_statistics()