"""
src/model/inference.py

=== Основной принцип работы файла ===

Предикт модели на новых данных (аномалия + признаки).

Ключевые особенности (по ТЗ + последние уточнения):
- Multi-TF input: 5 отдельных тензоров (1m,3m,5m,10m,15m)
- Binary предсказание: да/нет — будет ли прибыль ≥ TP и дойдёт ли до TP раньше SL
- quiet_streak как дополнительный канал (repeat по seq_len)
- n_features из config (рассчитано в trainer)
- Проверка shape перед forward (защита от ошибок)
- Eval mode для скорости в live_loop

=== Главные функции ===

- Inference class
- predict(features_by_tf: dict, anomaly_type: str, extra_features: dict = {}) — предикт
- load_model(timeframe=None) — загрузка модели (per-TF или общая)

=== Примечания ===
- Input: list of 5 tensors (по TF), каждый (1, seq_len, n_features)
- Output: float [0-1] — вероятность "да/нет" (TP раньше SL и профит ≥ TP)
- Готов к интеграции в live_loop (только confirmed аномалии)
- Логи через setup_logger
"""

import torch
import numpy as np
import os
from datetime import datetime

from src.model.architectures import build_model
from src.core.config import load_config
from src.utils.logger import setup_logger

logger = setup_logger('inference', logging.INFO)

class Inference:
    def __init__(self):
        self.config = load_config()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model()
        self.n_features = self.config['model']['n_features']
        self.seq_len = self.config['seq_len']

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
        Предикт: да/нет — будет ли прибыль ≥ TP и дойдёт ли до TP раньше SL
        """
        timeframes = self.config['timeframes']

        # Подготовка multi-TF input (5 тензоров)
        inputs = []
        for tf in timeframes:
            if tf not in features_by_tf or not features_by_tf[tf]:
                # Fallback: zeros если TF отсутствует
                input_tf = np.zeros((self.seq_len, self.n_features))
            else:
                input_tf = features_by_tf[tf]  # np.array из feature_engine
                if input_tf.shape != (self.seq_len, self.n_features):
                    logger.error(f"Shape mismatch для {tf}: {input_tf.shape} vs ({self.seq_len}, {self.n_features})")
                    return 0.0

            # Добавление quiet_streak как дополнительного канала (repeat по seq_len)
            quiet_streak = extra_features.get('quiet_streak', 0)
            quiet_col = np.full((self.seq_len, 1), quiet_streak)
            input_tf = np.concatenate([input_tf, quiet_col], axis=1)  # +1 канал

            # Тензор (1, seq_len, n_features + 1)
            inputs.append(torch.tensor(input_tf, dtype=torch.float32).unsqueeze(0))

        # Предикт
        with torch.no_grad():
            prob = self.model(inputs)[0].item()  # binary prob (0-1)

        logger.debug(f"Предикт для {anomaly_type}: {prob:.4f}")
        return prob