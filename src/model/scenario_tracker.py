"""
src/model/scenario_tracker.py

=== Основной принцип работы файла ===

Отслеживание и анализ бинарных сценариев для всех признаков (из ТЗ + delta VA + sequential паттерны).

Ключевые задачи:
- Хранение сценариев: каждый предикт + исход сделки → обновление статистики
- Бинарные состояния для 12 базовых признаков + delta VA + sequential_* + quiet_streak
- Вес сценария = винрейт × log(количество сделок + 1) — чтобы 1 сделка 100% не перевешивала 10 сделок 70%
- Кластеризация HDBSCAN для выявления топ-групп сценариев
- Вывод статистики в CSV (для Google Sheets) — по команде или каждые 1000 сделок
- Принудительная выгрузка по команде

=== Главные классы ===

- ScenarioTracker — основной класс
- add_scenario(pred_features, outcome) — добавление сценария
- get_top_scenarios(n=50) — топ по весу
- cluster_scenarios() — HDBSCAN кластеризация
- export_statistics() — выгрузка в CSV

=== Примечания ===
- Все признаки бинаризуются (increased/decreased, >0/<0, count > threshold)
- Sequential паттерны — count по окнам (0–4)
- Quiet streak — бинарный (quiet_streak > 3 → 1)
- Статистика хранится в dict[scenario_tuple] → {'wins': int, 'losses': int, 'count': int}
"""

import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict
from sklearn.cluster import HDBSCAN
import logging
import os

from src.core.config import load_config
from src.utils.logger import setup_logger

logger = setup_logger('scenario_tracker', logging.INFO)

class ScenarioTracker:
    def __init__(self):
        self.config = load_config()
        self.scenarios = defaultdict(lambda: {'wins': 0, 'losses': 0, 'count': 0, 'last_update': None})
        self.data_dir = self.config['paths']['data_dir']
        self.export_path = os.path.join(self.data_dir, 'scenario_stats.csv')
        self.min_count_weight = 5  # минимальное кол-во сделок для значимого веса

    def _binarize_features(self, features: dict) -> tuple:
        """
        Преобразование всех признаков в бинарные состояния (tuple для ключа)
        """
        binary_states = []

        # 12 базовых из ТЗ
        for key in ['volume', 'bid', 'ask', 'delta', 'mid_price_left', 'mid_price_right',
                    'price_change', 'volatility', 'price_channel_position', 'va_position',
                    'delta_mid_dist', 'delta_means']:
            val = features.get(key + '_change_pct', 0)
            binary_states.append(1 if val > 0 else 0)  # increased
            binary_states.append(1 if val > 5 else 0)  # strong increase (threshold)

        # Delta VA признаки
        binary_states.append(features.get('delta_positive', 0))
        binary_states.append(features.get('delta_increased', 0))
        binary_states.append(1 if abs(features.get('delta_change_pct', 0)) > 10 else 0)

        # Sequential паттерны (все, как утверждено)
        for seq_key in ['sequential_delta_positive_count', 'sequential_delta_increased_count',
                        'sequential_volume_increased_count', 'sequential_volatility_increased_count',
                        'sequential_price_change_positive_count', 'sequential_above_vah_count',
                        'sequential_below_val_count', 'accelerating_delta_imbalance',
                        # ... все остальные sequential_* из списка ~20–25
                        ]:
            count = features.get(seq_key, 0)
            binary_states.append(1 if count >= 2 else 0)  # минимум 2 окна подряд
            binary_states.append(1 if count >= 4 else 0)  # максимум

        # Quiet streak
        binary_states.append(1 if features.get('quiet_streak', 0) >= 3 else 0)

        return tuple(binary_states)

    def add_scenario(self, pred_features: dict, outcome: int):  # outcome: 1=win, 0=loss
        """
        Добавление сценария после предикта и исхода сделки
        """
        key = self._binarize_features(pred_features)
        entry = self.scenarios[key]

        entry['count'] += 1
        if outcome == 1:
            entry['wins'] += 1
        else:
            entry['losses'] += 1

        entry['last_update'] = datetime.utcnow()

        # Авто-экспорт каждые 1000 сделок
        total_count = sum(e['count'] for e in self.scenarios.values())
        if total_count % 1000 == 0:
            self.export_statistics()

    def get_weight(self, scenario):
        """Вес = винрейт × log(count + 1)"""
        entry = self.scenarios[scenario]
        if entry['count'] == 0:
            return 0.0
        winrate = entry['wins'] / entry['count']
        return winrate * np.log(entry['count'] + 1)

    def get_top_scenarios(self, n=50):
        """Топ сценариев по весу"""
        sorted_scen = sorted(self.scenarios.items(), key=lambda x: self.get_weight(x[0]), reverse=True)
        return sorted_scen[:n]

    def cluster_scenarios(self):
        """Кластеризация HDBSCAN для выявления групп"""
        if len(self.scenarios) < 10:
            return []

        # Преобразование в массив для кластеризации
        keys = list(self.scenarios.keys())
        X = np.array(keys)  # tuple → array

        hdb = HDBSCAN(min_cluster_size=5, min_samples=3)
        labels = hdb.fit_predict(X)

        clusters = defaultdict(list)
        for i, label in enumerate(labels):
            if label != -1:  # шум не учитываем
                clusters[label].append(keys[i])

        # Топ кластеры по среднему весу
        cluster_stats = []
        for label, scen_list in clusters.items():
            weights = [self.get_weight(s) for s in scen_list]
            cluster_stats.append({
                'cluster_id': label,
                'size': len(scen_list),
                'avg_weight': np.mean(weights),
                'scenarios': scen_list
            })

        return sorted(cluster_stats, key=lambda x: x['avg_weight'], reverse=True)

    def export_statistics(self):
        """Выгрузка в CSV для Google Sheets"""
        data = []
        for key, entry in self.scenarios.items():
            winrate = entry['wins'] / entry['count'] if entry['count'] > 0 else 0
            weight = self.get_weight(key)
            data.append({
                'scenario': str(key),
                'count': entry['count'],
                'wins': entry['wins'],
                'losses': entry['losses'],
                'winrate': winrate,
                'weight': weight,
                'last_update': entry['last_update']
            })

        df = pd.DataFrame(data)
        df.sort_values('weight', ascending=False, inplace=True)
        df.to_csv(self.export_path, index=False)
        logger.info(f"Статистика сценариев выгружена: {self.export_path}")

    def export_top_clusters(self):
        clusters = self.cluster_scenarios()
        if not clusters:
            return

        data = []
        for cl in clusters[:10]:
            data.append({
                'cluster_id': cl['cluster_id'],
                'size': cl['size'],
                'avg_weight': cl['avg_weight'],
                'scenarios_count': len(cl['scenarios'])
            })

        df = pd.DataFrame(data)
        df.to_csv(os.path.join(self.data_dir, 'top_clusters.csv'), index=False)
        logger.info("Топ кластеры выгружены")config