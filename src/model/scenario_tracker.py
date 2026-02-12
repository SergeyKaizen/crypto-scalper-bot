# src/model/scenario_tracker.py
"""
Модуль отслеживания и кластеризации бинарных сценариев.

Основные функции:
- collect_scenario() — добавление нового сценария после каждой закрытой сделки
- run_clustering() — HDBSCAN кластеризация + расчёт весов
- export_top_scenarios() — выгрузка топ-сценариев в CSV/таблицу
- get_scenario_weight() — вес сценария = winrate × log(кол-во + 1)

Логика:
- Сценарий — бинарный вектор 16 бит (12 признаков + C/V/CV/Q)
- После каждой закрытой виртуальной/реальной сделки — добавляем сценарий + исход (1/0)
- Каждые 10 000 свечей (или по команде) — запускаем HDBSCAN
- HDBSCAN лучше K-Means: не нужно задавать число кластеров, находит шум
- Вес сценария = winrate × log10(кол-во + 1) — штрафует редкие сценарии
- Экспорт — CSV или Google Sheets (через gspread — опционально)

Зависимости:
- hdbscan
- numpy / polars
- config["scenario"]["export_path"] — путь для выгрузки
"""

import logging
from typing import Dict, List, Tuple
from collections import defaultdict
import numpy as np
import hdbscan
import polars as pl

from src.core.config import load_config

logger = logging.getLogger(__name__)


class ScenarioTracker:
    """Отслеживание и кластеризация сценариев"""

    def __init__(self, config: Dict):
        self.config = config
        self.scenarios = []  # List[Tuple[vector: np.array(16,), outcome: int]]
        self.min_count_for_cluster = config["scenario"].get("min_count_for_cluster", 5)
        self.export_path = config["scenario"].get("export_path", "scenarios_top.csv")

        logger.info("ScenarioTracker initialized")

    def collect_scenario(self, agg_features: Dict[str, float], outcome: int):
        """
        Добавление нового сценария после закрытия сделки

        Args:
            agg_features: Dict[str, float] — 12 признаков + 4 условия (C/V/CV/Q)
            outcome: 1 (профит) или 0 (убыток)
        """
        # Формируем бинарный вектор (16 элементов)
        vector = []
        # 12 признаков — бинаризуем (up/down)
        for key in sorted(agg_features.keys()):
            if key in ["C", "V", "CV", "Q"]:
                vector.append(agg_features[key])  # уже 0/1
            else:
                # Признаки — >0 → 1, <=0 → 0 (можно улучшить)
                vector.append(1 if agg_features[key] > 0 else 0)

        if len(vector) != 16:
            logger.error("Неверный размер вектора сценария: %d", len(vector))
            return

        self.scenarios.append((np.array(vector), outcome))
        logger.debug("Collected scenario: %s → outcome=%d (total: %d)", vector, outcome, len(self.scenarios))

    def run_clustering(self) -> Dict:
        """
        Запуск HDBSCAN кластеризации + расчёт весов

        Returns:
            Dict[int, Dict] — кластер → {winrate, count, weight, representative_vector}
        """
        if len(self.scenarios) < self.min_count_for_cluster * 5:
            logger.info("Слишком мало сценариев для кластеризации (%d < %d)", len(self.scenarios), self.min_count_for_cluster * 5)
            return {}

        # Формируем матрицу
        X = np.array([s[0] for s in self.scenarios])
        outcomes = np.array([s[1] for s in self.scenarios])

        # HDBSCAN — density-based clustering
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_count_for_cluster,
            min_samples=3,
            metric='hamming'  # Для бинарных векторов — Hamming distance
        )
        labels = clusterer.fit_predict(X)

        # Статистика по кластерам
        stats = defaultdict(lambda: {"count": 0, "wins": 0, "vectors": []})
        for label, outcome, vector in zip(labels, outcomes, X):
            if label == -1:  # Шум — пропускаем
                continue
            stats[label]["count"] += 1
            stats[label]["wins"] += outcome
            stats[label]["vectors"].append(vector)

        result = {}
        for label, data in stats.items():
            winrate = data["wins"] / data["count"] if data["count"] > 0 else 0.0
            weight = winrate * np.log10(data["count"] + 1)  # log10(n+1) — штраф за редкость
            representative = np.mean(data["vectors"], axis=0)  # средний вектор кластера

            result[label] = {
                "winrate": winrate,
                "count": data["count"],
                "weight": weight,
                "representative": representative.tolist()
            }

        # Сортировка по весу
        sorted_clusters = sorted(result.items(), key=lambda x: x[1]["weight"], reverse=True)

        logger.info("HDBSCAN clustering completed: %d clusters, total scenarios: %d", len(result), len(self.scenarios))

        return {"clusters": sorted_clusters, "noise_count": sum(1 for l in labels if l == -1)}

    def export_top_scenarios(self, top_n: int = 20):
        """Выгрузка топ-сценариев в CSV (для Google Sheets)"""
        clustering = self.run_clustering()
        if not clustering["clusters"]:
            logger.warning("No clusters to export")
            return

        data = []
        for label, stats in clustering["clusters"][:top_n]:
            row = {
                "cluster_id": label,
                "winrate": stats["winrate"],
                "count": stats["count"],
                "weight": stats["weight"],
                "representative_vector": ",".join(f"{x:.2f}" for x in stats["representative"])
            }
            data.append(row)

        df = pl.DataFrame(data)
        df.write_csv(self.export_path)
        logger.info("Exported top %d scenarios to %s", top_n, self.export_path)

    def get_scenario_weight(self, vector: np.ndarray) -> float:
        """Вес конкретного сценария (для фильтрации в live)"""
        # Упрощённо — можно улучшить поиском ближайшего кластера
        return 1.0  # Placeholder — в реальном коде поиск по HDBSCAN