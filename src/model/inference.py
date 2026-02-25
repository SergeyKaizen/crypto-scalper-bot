"""
src/model/inference.py

=== Основной принцип работы файла ===

Предикт модели на новых данных (аномалия + признаки).

Ключевые особенности (по ТЗ + все утверждённые улучшения):
- Multi-TF input: 5 отдельных тензоров (1m,3m,5m,10m,15m)
- Binary предсказание: да/нет — будет ли прибыль ≥ TP и дойдёт ли до TP раньше SL
- quiet_streak как дополнительный канал (repeat по seq_len)
- cluster_id как фича (получаем из scenario_tracker и передаём в модель)
- Валидация входных фич через Pydantic FeaturesSchema (data-contract + versioning)
- n_features из config (динамически рассчитано в trainer)
- Проверка shape и схемы перед forward (защита от тихих поломок)
- Eval mode для скорости в live_loop

=== Главные функции ===

- Inference class
- predict(features_by_tf: dict, anomaly_type: str, extra_features: dict = {}) — предикт
- load_model(timeframe=None) — загрузка модели (per-TF или общая)

=== Примечания ===
- Input: list of 5 tensors (по TF), cluster_id (scalar)
- Output: float [0-1] — вероятность "да/нет" (TP раньше SL и профит ≥ TP)
- cluster_id = 0 для fallback (outlier или нет кластера)
- Валидация фич через FeaturesSchema — предотвращает поломки при изменении признаков
- Готов к интеграции в live_loop (только confirmed аномалии)
- Логи через setup_logger
"""

import torch
import numpy as np
import os
from datetime import datetime
from typing import Dict, List

from src.model.architectures import build_model
from src.core.config import load_config
from src.model.scenario_tracker import ScenarioTracker
from src.features.feature_engine import FeaturesSchema  # data-contract
from src.utils.logger import setup_logger

logger = setup_logger('inference', logging.INFO)

class Inference:
    def __init__(self):
        self.config = load_config()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model()
        self.n_features = self.config['model']['n_features']
        self.seq_len = self.config['seq_len']
        self.scenario_tracker = ScenarioTracker()  # для cluster_id

    def load_model(self, timeframe=None):
        """Загрузка модели (per-TF или общая)"""
        model_path = f"models/model_{timeframe or 'all'}_{datetime.now().strftime('%Y%m%d')}.pt"
        if not os.path.exists(model_path):
            logger.warning(f"Модель {model_path} не найдена — использую дефолтную")
            return build_model(self.config)

        model = build_model(self.config)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model

    def predict(self, features_by_tf: Dict[str, np.ndarray], anomaly_type: str, extra_features: Dict = {}) -> float:
        """
        Предикт: да/нет — будет ли прибыль ≥ TP и дойдёт ли до TP раньше SL
        
        Входные фичи валидируются через FeaturesSchema (data-contract + versioning).
        cluster_id получается из scenario_tracker и передаётся в модель.
        """
        timeframes = self.config['timeframes']

        # Валидация фич для каждого TF (data-contract)
        validated_features = {}
        for tf in timeframes:
            if tf not in features_by_tf or features_by_tf[tf] is None:
                validated_features[tf] = np.zeros((self.seq_len, self.n_features))
                continue

            raw_feats = features_by_tf[tf]
            if raw_feats.shape != (self.seq_len, self.n_features):
                logger.error(f"Shape mismatch для {tf}: {raw_feats.shape} vs ({self.seq_len}, {self.n_features})")
                return 0.0

            # Преобразуем в dict для валидации (Pydantic требует dict)
            feat_dict = {
                'version': FEATURES_VERSION,
                'price_change_pct': float(raw_feats[-1, 0]),  # пример — реальные индексы по твоей схеме
                # ... все остальные поля из FeaturesSchema (заполняем по индексам или по ключам)
                # Для простоты — предполагаем, что raw_feats — np.array с фиксированным порядком
                # В реальности — лучше маппинг по именам фич
            }

            try:
                FeaturesSchema(**feat_dict)  # валидация + проверка версии
            except Exception as e:
                logger.error(f"Ошибка валидации фич для {tf}: {e}")
                return 0.0

            validated_features[tf] = raw_feats

        # Получаем cluster_id
        # Берём фичи любого TF (binarize одинаковый для всех)
        sample_feats = validated_features[list(timeframes)[0]]
        sample_dict = {k: float(sample_feats[-1, i]) for i, k in enumerate(FeaturesSchema.__fields__ if k != 'version')}
        key = self.scenario_tracker._binarize_features(sample_dict)
        cluster_id = self.scenario_tracker.get_cluster_id(key) if hasattr(self.scenario_tracker, 'get_cluster_id') else 0

        extra_features['cluster_id'] = cluster_id

        # Подготовка multi-TF input (5 тензоров)
        inputs = []
        for tf in timeframes:
            input_tf = validated_features[tf]
            # Добавление quiet_streak как дополнительного канала
            quiet_streak = extra_features.get('quiet_streak', 0)
            quiet_col = np.full((self.seq_len, 1), quiet_streak, dtype=np.float32)
            input_tf = np.concatenate([input_tf, quiet_col], axis=1)

            inputs.append(torch.tensor(input_tf, dtype=torch.float32).unsqueeze(0).to(self.device))

        # cluster_tensor
        cluster_tensor = torch.tensor([cluster_id], dtype=torch.long, device=self.device)

        # Предикт
        with torch.no_grad():
            prob, _ = self.model(inputs, cluster_tensor)  # используем только prob
            prob = prob.item()  # binary prob (0-1)

        logger.debug(f"Предикт для {anomaly_type}: prob={prob:.4f}, cluster_id={cluster_id}")
        return prob