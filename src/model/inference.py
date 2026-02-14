# src/model/inference.py
"""
InferenceEngine — выполняет предсказания модели.
Поддерживает:
- Обычный предикт на всех TF
- Отдельный предикт для каждой аномалии (как ты просил)
- Работает с Resampler (получает dict[tf: DataFrame])
- Поддержка live и бэктеста
"""

import torch
from typing import Dict, Optional

from ..utils.logger import logger
from ..core.config import get_config
from ..core.types import ModelInput, AnomalySignal, Direction
from .architectures import ScalperHybridModel


class InferenceResult:
    """Результат одного предикта."""
    def __init__(self, probabilities: torch.Tensor):
        self.probabilities = probabilities.softmax(dim=-1)
        self.confidence = self.probabilities.max().item()
        self.direction = ["L", "S", "F"][self.probabilities.argmax().item()]  # Long / Short / Flat

    def __repr__(self):
        return f"Prediction(confidence={self.confidence:.4f}, direction={self.direction})"


class InferenceEngine:
    def __init__(self, config=None):
        self.config = config or get_config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = ScalperHybridModel(self.config).to(self.device)
        self.model.eval()  # сразу в режим inference

        # Загрузка обученной модели (если есть)
        self._load_model()

        logger.info("InferenceEngine готов", device=self.device)

    def _load_model(self):
        """Загружает сохранённую модель, если существует."""
        model_path = Path("models") / f"scalper_{self.config['hardware']['type']}.pth"
        if model_path.exists():
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            logger.info("Модель загружена", path=str(model_path))
        else:
            logger.warning("Модель не найдена — будут случайные предикты до обучения")

    def predict(self, model_input: ModelInput) -> InferenceResult:
        """Обычный предикт на всех данных."""
        with torch.no_grad():
            logits = self.model(model_input)
            return InferenceResult(logits)

    def predict_for_anomaly(self, model_input: ModelInput, anomaly_type: str) -> InferenceResult:
        """
        Специальный предикт для конкретной аномалии.
        Можно модифицировать вход (например, усилить вес аномалии).
        """
        # Пока просто обычный предикт — можно добавить логику усиления
        logger.debug("Предикт для аномалии", anomaly_type=anomaly_type)

        # Пример модификации входа (можно расширить)
        if anomaly_type == "CV":
            # Увеличиваем уверенность для CV (пример)
            model_input.anomalies = [a for a in model_input.anomalies if a.anomaly_type.value == "CV"]

        return self.predict(model_input)

    def predict_batch(self, inputs: List[ModelInput]) -> List[InferenceResult]:
        """Пакетный предикт (ускоряет на GPU)."""
        with torch.no_grad():
            batch_logits = self.model([i for i in inputs])
            return [InferenceResult(logits) for logits in batch_logits]