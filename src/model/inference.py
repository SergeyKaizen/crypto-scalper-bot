"""
src/model/inference.py

=== Основной принцип работы файла ===

Предикт модели на новых данных (аномалия + признаки).

Ключевые особенности (по ТЗ + утверждённые улучшения):
- Multi-TF input: 5 отдельных тензоров (1m,3m,5m,10m,15m)
- Binary предсказание: да/нет — будет ли прибыль ≥ TP и дойдёт ли до TP раньше SL
- quiet_streak как дополнительный канал (repeat по seq_len)
- cluster_id как дополнительная фича (получаем из scenario_tracker и передаём в модель)
- MC Dropout: 5 passes с включённым dropout для uncertainty (variance) — фильтрует неуверенные сигналы
- n_features из config (динамически рассчитано в trainer)
- Проверка shape перед forward (защита от ошибок)
- Eval mode для скорости в live_loop

=== Главные функции ===
- Inference class
- predict(features_by_tf: dict, anomaly_type: str, extra_features: dict = {}) — предикт

=== Примечания ===
- MC Dropout: passes=5 — средняя prob + uncertainty (std) > threshold → пропуск
- Threshold uncertainty = 0.05 (можно в config)
- Полностью соответствует ТЗ + улучшениям (MC Dropout для uncertainty)
- Готов к интеграции в live_loop (только confirmed аномалии)
- Логи через setup_logger
"""

import torch
import numpy as np
import os
from datetime import datetime

from src.model.architectures import build_model
from src.core.config import load_config
from src.model.scenario_tracker import ScenarioTracker
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
        self.mc_passes = 5  # утверждённое значение
        self.uncertainty_threshold = 0.05  # порог для фильтра (можно в config)

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

    def predict(self, features_by_tf: dict, anomaly_type: str, extra_features: dict = {}):
        """
        Предикт с MC Dropout (passes=5) для uncertainty
        """
        timeframes = self.config['timeframes']

        # Получаем cluster_id
        sample_feats = features_by_tf[list(timeframes)[0]]
        key = self.scenario_tracker._binarize_features(sample_feats)
        cluster_id = self.scenario_tracker.get_cluster_id(key) if hasattr(self.scenario_tracker, 'get_cluster_id') else 0

        extra_features['cluster_id'] = cluster_id

        # Подготовка input (5 тензоров)
        inputs = []
        for tf in timeframes:
            input_tf = features_by_tf[tf]
            if input_tf.shape != (self.seq_len, self.n_features):
                logger.error(f"Shape mismatch для {tf}: {input_tf.shape} vs ({self.seq_len}, {self.n_features})")
                return 0.0

            quiet_streak = extra_features.get('quiet_streak', 0)
            quiet_col = np.full((self.seq_len, 1), quiet_streak)
            input_tf = np.concatenate([input_tf, quiet_col], axis=1)

            inputs.append(torch.tensor(input_tf, dtype=torch.float32).unsqueeze(0))

        cluster_tensor = torch.tensor([cluster_id], dtype=torch.long, device=self.device)

        # MC Dropout: 5 проходов с dropout включённым
        probs = []
        self.model.train()  # включаем dropout даже в предикте
        with torch.no_grad():
            for _ in range(self.mc_passes):
                prob, _ = self.model(inputs, cluster_tensor)
                probs.append(prob.item())

        mean_prob = np.mean(probs)
        uncertainty = np.std(probs)  # variance как мера uncertainty

        if uncertainty > self.uncertainty_threshold:
            logger.debug(f"Высокая неуверенность: {uncertainty:.4f} — пропускаем сигнал")
            return 0.0

        logger.debug(f"Предикт для {anomaly_type}: mean_prob={mean_prob:.4f}, uncertainty={uncertainty:.4f}, cluster_id={cluster_id}")
        return mean_prob