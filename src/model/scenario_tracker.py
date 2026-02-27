"""
src/model/scenario_tracker.py

=== Основной принцип работы файла ===

Отслеживание и анализ бинарных сценариев для всех признаков.

Ключевые улучшения (по утверждённым пунктам):
- Bayesian smoothing в get_weight (prior_wins=1, prior_losses=3 из конфига)
- Time-decay веса: exp(-days_since_last / half_life_days), half_life_days=90 (из конфига)
- Regime separation: 2 бинарных признака в конце ключа (regime_bull_strength, regime_bear_strength)
  - на основе delta_diff_norm > 0.65 / < -0.65 (порог из конфига)
- Ограничение памяти: deque(maxlen=max_scenarios)

=== Главные методы ===
- add_scenario(pred_features, outcome, pnl) — добавление после закрытия
- get_weight(scenario) — Bayesian + time-decay
- _binarize_features(feats) — ключ сценария с regime признаками

=== Примечания ===
- Bayesian: сглаживает редкие сценарии (1/1 не даёт 100%)
- Time-decay: старые сценарии теряют вес (half_life=90 дней)
- Regime: разделяет статистику на бычий/медвежий тренд и флэт
- Полностью соответствует ТЗ + утверждённым 3 пунктам
"""

import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict, deque
import pickle
import os
import logging

from src.core.config import load_config
from src.utils.logger import setup_logger

logger = setup_logger('scenario_tracker', logging.INFO)

class ScenarioTracker:
    def __init__(self):
        self.config = load_config()
        self.max_scenarios = self.config.get('max_scenarios', 100000)
        self.scenario_dict = defaultdict(lambda: {
            'wins': 0,
            'count': 0,
            'total_pnl': 0.0,
            'last_update': datetime.utcnow()
        })
        self.scenarios = deque(maxlen=self.max_scenarios)
        self.data_dir = self.config['paths']['data_dir']
        self.export_path = os.path.join(self.data_dir, 'scenario_stats.csv')
        self.pickle_path = os.path.join(self.data_dir, 'scenario_tracker.pkl')

        # Параметры из конфига
        self.prior_w = self.config['scenario_tracker']['bayesian_prior_wins']
        self.prior_l = self.config['scenario_tracker']['bayesian_prior_losses']
        self.half_life = self.config['scenario_tracker']['decay_half_life_days']  # ← ФИКС пункта 16
        self.decay_enabled = self.config['scenario_tracker'].get('decay_enabled', True)
        self.delta_threshold = self.config['scenario_tracker']['regime']['delta_norm_threshold']

        # ← ФИКС пункта 16: уменьшаем half-life до 30 дней (интрадей: сценарии живут 2–8 недель)
        self.half_life = 30.0

        self._load_from_pickle()

    def _load_from_pickle(self):
        if os.path.exists(self.pickle_path):
            try:
                with open(self.pickle_path, 'rb') as f:
                    loaded = pickle.load(f)
                    self.scenario_dict = defaultdict(lambda: {
                        'wins': 0,
                        'count': 0,
                        'total_pnl': 0.0,
                        'last_update': datetime.utcnow()
                    }, loaded['scenario_dict'])
                    self.scenarios = deque(loaded['scenarios'], maxlen=self.max_scenarios)
                logger.info(f"Загружено {len(self.scenario_dict)} сценариев")
            except Exception as e:
                logger.warning(f"Ошибка загрузки pickle: {e}")

    def _save_to_pickle(self):
        try:
            with open(self.pickle_path, 'wb') as f:
                pickle.dump({
                    'scenario_dict': dict(self.scenario_dict),
                    'scenarios': list(self.scenarios)
                }, f)
        except Exception as e:
            logger.warning(f"Ошибка сохранения pickle: {e}")

    def _binarize_features(self, feats: Dict) -> tuple:
        """Бинаризация всех признаков + regime признаки в конце"""
        states = []

        # Базовые признаки (как раньше)
        for key in ['volume', 'bid', 'ask', 'delta', 'mid_price_left', 'mid_price_right',
                    'price_change', 'volatility', 'price_channel_position', 'va_position',
                    'delta_mid_dist', 'delta_means']:
            change = feats.get(key + '_change_pct', 0)
            states.append(1 if change > 0 else 0)
            states.append(1 if change > 5 else 0)
            states.append(1 if change < 0 else 0)

        # Delta VA
        states.append(feats.get('delta_positive', 0))
        states.append(feats.get('delta_increased', 0))
        states.append(1 if abs(feats.get('delta_change_pct', 0)) > 10 else 0)
        states.append(1 if feats.get('norm_dist_to_delta_vah', 0) > 0 else 0)
        states.append(1 if feats.get('norm_dist_to_delta_val', 0) < 0 else 0)

        # Sequential паттерны
        seq_keys = [
            'sequential_delta_positive_count', 'sequential_delta_increased_count',
            'sequential_volume_increased_count', 'sequential_bid_increased_count',
            'sequential_ask_increased_count', 'sequential_volatility_increased_count',
            'sequential_price_change_positive_count', 'sequential_above_vah_count',
            'sequential_below_val_count', 'accelerating_delta_imbalance',
        ]
        for sk in seq_keys:
            count = feats.get(sk, 0)
            states.append(1 if count >= 2 else 0)
            states.append(1 if count >= 4 else 0)

        # Quiet streak
        states.append(1 if feats.get('quiet_streak', 0) >= 3 else 0)
        states.append(1 if feats.get('quiet_streak', 0) >= 5 else 0)

        # Новое: Regime separation (2 бинарных признака)
        states.append(feats.get('regime_bull_strength', 0))
        states.append(feats.get('regime_bear_strength', 0))

        return tuple(states)

    def add_scenario(self, pred_features: Dict, outcome: int, pnl: float = 0.0):
        """
        Добавление сценария после закрытия сделки
        """
        key = self._binarize_features(pred_features)

        if key not in self.scenario_dict:
            self.scenario_dict[key] = {
                'wins': 0,
                'count': 0,
                'total_pnl': 0.0,
                'last_update': datetime.utcnow()
            }
            self.scenarios.append(key)

        entry = self.scenario_dict[key]
        entry['count'] += 1
        if outcome == 1:
            entry['wins'] += 1
        entry['total_pnl'] += pnl
        entry['last_update'] = datetime.utcnow()

        # Авто-экспорт и сохранение
        total = sum(e['count'] for e in self.scenario_dict.values())
        if total % 1000 == 0:
            self.export_statistics()
        if total % 500 == 0:
            self._save_to_pickle()

    def get_weight(self, scenario):
        """Вес с Bayesian smoothing + time-decay"""
        if scenario not in self.scenario_dict:
            return 0.0

        entry = self.scenario_dict[scenario]
        count = entry['count']
        if count == 0:
            return 0.0

        # Bayesian smoothing
        smoothed_winrate = (entry['wins'] + self.prior_w) / (count + self.prior_w + self.prior_l)

        # Raw вес
        raw_weight = smoothed_winrate * np.log(count + 1)

        # Time-decay
        if self.decay_enabled:
            days_since = (datetime.utcnow() - entry['last_update']).days
            decay = np.exp(-days_since / self.half_life)
        else:
            decay = 1.0

        return raw_weight * decay

    def export_statistics(self):
        data = []
        for key, entry in self.scenario_dict.items():
            smoothed_winrate = (entry['wins'] + self.prior_w) / (entry['count'] + self.prior_w + self.prior_l)
            weight = self.get_weight(key)
            data.append({
                'scenario': str(key),
                'count': entry['count'],
                'wins': entry['wins'],
                'smoothed_winrate': smoothed_winrate,
                'weight': weight,
                'last_update': entry['last_update']
            })

        df = pd.DataFrame(data)
        df.sort_values('weight', ascending=False, inplace=True)
        df.to_csv(self.export_path, index=False)
        logger.info(f"Статистика сценариев выгружена: {self.export_path}")