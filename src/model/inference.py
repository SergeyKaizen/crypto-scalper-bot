"""
src/model/inference.py

=== Основной принцип работы файла ===

InferenceEngine — модуль предсказания модели в live-режиме и бэктесте.
- Загрузка чекпоинта
- Forward pass модели
- Возврат вероятностей long/short
- Uncertainty через MC Dropout (если включено)
- Калибровка вероятностей (опционально)
"""

import logging
import torch
from typing import Tuple, Dict, Any

import os
import numpy as np

from src.core.config import load_config
from src.model.architectures import MultiTFHybrid, TinyHybrid
from src.utils.logger import setup_logger

setup_logger()
logger = logging.getLogger(__name__)


class InferenceEngine:
    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model()
        self.mc_passes = 20

        self.ensemble_models = []
        self.load_ensemble()

        self.calibration_slope = 1.0
        self.calibration_intercept = 0.0
        self.calibration_enabled = config.get("calibration_enabled", False)

    def _load_model(self):
        """Загрузка обученной модели из чекпоинта"""
        checkpoint_path = "models/best_model.pth"
        if self.config.get("use_tiny_model", False):
            model = TinyHybrid(self.config).to(self.device)
        else:
            model = MultiTFHybrid(self.config).to(self.device)

        try:
            model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
            logger.info(f"Model loaded from {checkpoint_path}")
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise

        model.eval()
        return model

    def load_ensemble(self):
        """Загрузка нескольких моделей для ensemble"""
        ensemble_paths = self.config.get("ensemble_models", [])
        for path in ensemble_paths:
            if os.path.exists(path):
                model = self._load_model()
                model.load_state_dict(torch.load(path, map_location=self.device))
                model.eval()
                self.ensemble_models.append(model)
                logger.info(f"Ensemble model loaded: {path}")
        if self.ensemble_models:
            logger.info(f"Ensemble enabled: {len(self.ensemble_models)} models")

    def predict(self, features: Dict[str, Any]) -> Tuple[float, float, float]:
        """
        Основной метод предсказания.
        features: {"sequences": {tf: tensor}, "features": tensor}
        Возвращает prob_long, prob_short, uncertainty
        """
        sequences = {tf: t.to(self.device) for tf, t in features.get("sequences", {}).items()}
        agg_features = features.get("features")
        if agg_features is not None:
            agg_features = agg_features.to(self.device)

        self.model.train()
        probs = []
        with torch.no_grad():
            for _ in range(self.mc_passes):
                prob, _ = self.model(sequences, agg_features)
                probs.append(prob.squeeze())
        probs = torch.stack(probs)

        if self.ensemble_models:
            ensemble_probs = []
            for model in self.ensemble_models:
                model.train()
                for _ in range(self.mc_passes):
                    prob, _ = model(sequences, agg_features)
                    ensemble_probs.append(prob.squeeze())
            ensemble_probs = torch.stack(ensemble_probs)
            probs = torch.cat([probs, ensemble_probs], dim=0)

        mean_prob = probs.mean(dim=0)
        uncertainty = probs.std(dim=0).mean().item()

        if mean_prob.shape[0] > 1:
            prob_long = mean_prob[0].item()
            prob_short = mean_prob[1].item()
        else:
            prob_long = mean_prob.item()
            prob_short = 1.0 - prob_long

        if self.calibration_enabled:
            prob_long = self._calibrate(prob_long)
            prob_short = 1.0 - prob_long

        logger.debug(f"Prediction: long={prob_long:.3f}, short={prob_short:.3f}, uncertainty={uncertainty:.3f}")

        return prob_long, prob_short, uncertainty

    def _calibrate(self, prob: float) -> float:
        return 1 / (1 + np.exp(-self.calibration_slope * (prob - self.calibration_intercept)))

    def predict_single(self, features: Dict[str, Any]) -> float:
        self.model.eval()
        sequences = {tf: t.to(self.device) for tf, t in features.get("sequences", {}).items()}
        agg_features = features.get("features")
        if agg_features is not None:
            agg_features = agg_features.to(self.device)

        with torch.no_grad():
            prob, _ = self.model(sequences, agg_features)
        return prob.squeeze().item()

    def calibrate(self, prob: float) -> float:
        return prob

    def load_checkpoint(self, path: str):
        try:
            self.model.load_state_dict(torch.load(path, map_location=self.device))
            logger.info(f"Checkpoint loaded: {path}")
        except Exception as e:
            logger.error(f"Failed to load checkpoint {path}: {e}")