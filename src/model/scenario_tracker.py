"""
src/model/scenario_tracker.py

=== Основной принцип работы файла ===

Этот файл реализует отслеживание и анализ всех бинарных сценариев (комбинаций состояний признаков) после каждой закрытой сделки.

Ключевые задачи:
- После закрытия позиции (TP/SL) обновляет статистику сценария (winrate, count, weight).
- Weight = winrate * log(count + 1) — учитывает и вероятность успеха, и надёжность (мало сделок — низкий вес).
- Хранит все сценарии (full stats, без обрезки топ-N).
- Кластеризация — HDBSCAN (density-based, лучше K-Means для sparse бинарных данных по здравой логике).
- Экспорт полной статистики в CSV (все сценарии, sorted by weight descending).
- Поддержка принудительной выгрузки через scripts/export_scenarios.py --force.

Сценарий — строка бинарных состояний (например, "delta_bid_inc=1,dist_delta_pos=0,C=1,Q=0").
Хранится в dict или pd.DataFrame в памяти, сохраняется в storage при retrain/shutdown.

=== Главные функции и за что отвечают ===

- ScenarioTracker() — инициализация, загрузка из storage если есть.
- update_scenario(scenario_key: str, is_win: bool) — обновляет статистику после сделки.
- get_scenarios_stats(min_count: int = 1) → pd.DataFrame — возвращает все сценарии (full).
- cluster_scenarios() → добавляет HDBSCAN labels к df.
- export_to_csv(path: str) — экспорт full stats в CSV.
- save_state() / load_state() — сохранение/загрузка в storage (для retrain).

=== Примечания ===
- HDBSCAN выбран по логике (лучше для редких высокоприбыльных outliers).
- min_cluster_size=3–5 (в config).
- Weight формула по ТЗ: winrate * log(count + 1).
- Full stats — все сценарии, без топ-N.
- Готов к интеграции в entry_manager (после close) и export_scenarios.py.
- Логи через setup_logger.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from hdbscan import HDBSCAN
import os
import pickle

from src.core.config import load_config
from src.data.storage import Storage
from src.utils.logger import setup_logger

logger = setup_logger('scenario_tracker', logging.INFO)

class ScenarioTracker:
    """
    Отслеживание бинарных сценариев и их статистики.
    """
    def __init__(self):
        self.config = load_config()
        self.storage = Storage()
        self.scenarios = {}  # key: str сценария → {'win': int, 'loss': int, 'count': int, 'weight': float}
        self.load_state()

    def update_scenario(self, scenario_key: str, is_win: bool):
        """
        Обновляет статистику сценария после закрытия позиции.
        scenario_key — строка вида "delta_bid_inc=1,dist_delta_pos=0,C=1,Q=0"
        is_win — True если закрытие по TP, False по SL.
        """
        if scenario_key not in self.scenarios:
            self.scenarios[scenario_key] = {'win': 0, 'loss': 0, 'count': 0}

        entry = self.scenarios[scenario_key]
        if is_win:
            entry['win'] += 1
        else:
            entry['loss'] += 1
        entry['count'] += 1

        winrate = entry['win'] / entry['count'] if entry['count'] > 0 else 0.0
        entry['weight'] = winrate * np.log(entry['count'] + 1)  # формула по ТЗ

        logger.debug(f"Обновлён сценарий '{scenario_key}': winrate={winrate:.2f}, weight={entry['weight']:.4f}, count={entry['count']}")

        # Автосохранение каждые 100 обновлений
        if entry['count'] % 100 == 0:
            self.save_state()

    def get_scenarios_stats(self, min_count: int = 1) -> pd.DataFrame:
        """
        Возвращает DataFrame со всеми сценариями (full stats).
        Фильтр min_count для исключения шума (опционально).
        """
        data = []
        for key, stats in self.scenarios.items():
            if stats['count'] < min_count:
                continue
            winrate = stats['win'] / stats['count'] if stats['count'] > 0 else 0.0
            data.append({
                'scenario_key': key,
                'win': stats['win'],
                'loss': stats['loss'],
                'count': stats['count'],
                'winrate': winrate,
                'weight': stats['weight']
            })

        df = pd.DataFrame(data)
        df = df.sort_values('weight', ascending=False).reset_index(drop=True)

        # Кластеризация HDBSCAN
        if len(df) >= 3:  # минимум для кластеризации
            features = df[['winrate', 'count', 'weight']].values
            clusterer = HDBSCAN(min_cluster_size=self.config.get('hdbscan', {}).get('min_cluster_size', 5))
            labels = clusterer.fit_predict(features)
            df['cluster_label'] = labels

        logger.info(f"Статистика сценариев: {len(df)} записей (после фильтра min_count={min_count})")
        return df

    def export_to_csv(self, path: str = "scenarios_stats.csv"):
        """
        Экспорт полной статистики в CSV.
        """
        df = self.get_scenarios_stats(min_count=1)  # все
        df.to_csv(path, index=False)
        logger.info(f"Экспортировано {len(df)} сценариев в {path}")

    def save_state(self):
        """
        Сохраняет состояние tracker в storage или файл (pickle).
        """
        path = os.path.join(self.config['paths']['data_dir'], 'scenario_tracker.pkl')
        with open(path, 'wb') as f:
            pickle.dump(self.scenarios, f)
        logger.debug("Состояние ScenarioTracker сохранено")

    def load_state(self):
        """
        Загружает состояние из файла, если существует.
        """
        path = os.path.join(self.config['paths']['data_dir'], 'scenario_tracker.pkl')
        if os.path.exists(path):
            with open(path, 'rb') as f:
                self.scenarios = pickle.load(f)
            logger.info(f"Загружено {len(self.scenarios)} сценариев из {path}")
        else:
            logger.debug("Состояние ScenarioTracker не найдено — старт с нуля")