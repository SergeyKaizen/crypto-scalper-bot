# src/model/scenario_tracker.py
"""
Отслеживание бинарных сценариев и их весов (по ТЗ).

Что делает:
- Каждый сценарий — это комбинация 4 бинарных условий (candle_anomaly, volume_anomaly, cv_anomaly, q_condition)
- Всего 16 возможных сценариев (2^4)
- Для каждого сценария ведётся статистика:
  - count — сколько раз сценарий встретился
  - wins — сколько раз сценарий принёс профит (сделка закрыта по TP)
- Вес сценария = winrate × log(count + 1) — используется для корректировки вероятности в inference
- Кластеризация сценариев HDBSCAN (после накопления ≥50 сценариев)
- Автоматический экспорт статистики в CSV (раз в 500 обновлений + по команде export_scenarios.py)
- Готов к импорту в Google Sheets (Ctrl+V → Импорт)

Логика:
- update_scenario() — вызывается после закрытия сделки
- get_scenario_weight() — корректирует вероятность в inference
- export_statistics() — выгрузка в CSV
- _recluster_if_needed() — HDBSCAN кластеризация (периодически)
"""

import numpy as np
import polars as pl
from collections import defaultdict
from typing import Tuple, Dict
import hdbscan
import os
from datetime import datetime

from src.utils.logger import get_logger

logger = get_logger(__name__)

class ScenarioTracker:
    def __init__(self, config: dict):
        self.config = config
        
        # Хранилище сценариев: ключ (tuple из 4 битов) → статистика
        self.scenarios: Dict[Tuple[int, int, int, int], Dict[str, Any]] = defaultdict(lambda: {
            "count": 0,
            "wins": 0,
            "cluster_id": -1  # -1 = не кластеризован или шум
        })
        
        # HDBSCAN для кластеризации бинарных векторов
        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=5,
            min_samples=3,
            metric='hamming'  # расстояние Хэмминга для бинарных векторов
        )
        self.need_recluster = True  # флаг: нужно ли перекластеризовать
        
        self.export_dir = config.get("paths", {}).get("export_dir", "exports")
        os.makedirs(self.export_dir, exist_ok=True)

    def update_scenario(self, scenario_key: Tuple[int, int, int, int], is_win: bool):
        """
        Обновляет статистику сценария после закрытия сделки.
        scenario_key — tuple из 4 битов (candle, volume, cv, q)
        is_win — True если сделка закрыта по TP (профит)
        """
        stats = self.scenarios[scenario_key]
        stats["count"] += 1
        if is_win:
            stats["wins"] += 1
        
        # После каждых 500 обновлений — перекластеризация и экспорт
        if stats["count"] % 500 == 0:
            self._recluster_if_needed()
            self.export_statistics()

    def get_scenario_weight(self, scenario_key: Tuple[int, int, int, int]) -> float:
        """
        Возвращает вес сценария для корректировки вероятности в inference.
        Вес = winrate × log(count + 1)
        Диапазон ограничен 0.5–2.0 для стабильности.
        """
        stats = self.scenarios.get(scenario_key, {"count": 0, "wins": 0})
        count = stats["count"]
        wins = stats["wins"]
        
        if count == 0:
            return 1.0  # нейтральный вес для нового сценария
        
        winrate = wins / count
        weight = winrate * np.log1p(count)
        return max(0.5, min(2.0, weight))

    def _recluster_if_needed(self):
        """
        Перекластеризация сценариев с помощью HDBSCAN.
        Выполняется периодически или при большом приросте данных.
        """
        if not self.need_recluster or len(self.scenarios) < 50:
            return
        
        logger.info(f"Перекластеризация сценариев (HDBSCAN) — {len(self.scenarios)} сценариев")
        
        keys = list(self.scenarios.keys())
        vectors = np.array(keys)  # tuple → np.array (бинарные векторы)
        
        if len(vectors) < 10:
            return
        
        labels = self.clusterer.fit_predict(vectors)
        
        for i, key in enumerate(keys):
            self.scenarios[key]["cluster_id"] = int(labels[i])
        
        self.need_recluster = False
        logger.info(f"HDBSCAN завершён: {len(set(labels)) - 1} кластеров (без шума)")

    def export_statistics(self, filename: str = None):
        """
        Экспорт статистики всех сценариев в CSV.
        Колонки: scenario_key, count, wins, winrate, weight, cluster_id
        Готов к импорту в Google Sheets.
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"scenario_stats_{timestamp}.csv"
        
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
        
        # Принудительная перекластеризация после экспорта
        self.need_recluster = True

    def get_top_scenarios(self, top_n: int = 20) -> pl.DataFrame:
        """Топ-N сценариев по весу (для логов или мониторинга)"""
        data = []
        for key, stats in self.scenarios.items():
            count = stats["count"]
            wins = stats["wins"]
            winrate = wins / count if count > 0 else 0.0
            weight = winrate * np.log1p(count)
            data.append({
                "scenario_key": str(key),
                "weight": round(weight, 4),
                "winrate": round(winrate, 4),
                "count": count,
                "cluster_id": stats.get("cluster_id", -1)
            })
        df = pl.DataFrame(data).sort("weight", descending=True).head(top_n)
        return df


if __name__ == "__main__":
    config = load_config()
    tracker = ScenarioTracker(config)
    
    # Тест обновления
    key = (1, 0, 1, 0)  # candle=yes, volume=no, cv=yes, q=no
    tracker.update_scenario(key, is_win=True)
    tracker.update_scenario(key, is_win=False)
    
    print("Вес сценария:", tracker.get_scenario_weight(key))
    tracker.export_statistics("test_scenario_stats.csv")