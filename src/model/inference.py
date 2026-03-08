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
from src.model.architectures import HybridMultiTFConvGRU, build_model  # ← исправлено
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

        model = build_model(self.config).to(self.device)  # ← исправлено

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
                model = build_model(self.config).to(self.device)  # ← исправлено
                model.load_state_dict(torch.load(path, map_location=self.device))
                model.eval()
                self.ensemble_models.append(model)
                logger.info(f"Ensemble model loaded: {path}")
        if self.ensemble_models:
            logger.info(f"Ensemble enabled: {len(self.ensemble_models)} models")

    def _prepare_inputs(self, features: Dict[str, Any]):
        """Вспомогательный метод: dict → list[5 tensors] + cluster_id"""
        timeframes = self.config['timeframes']  # порядок строго по config
        sequences = []
        for tf in timeframes:
            tensor = features.get("sequences", {}).get(tf)
            if tensor is None:
                raise ValueError(f"Missing sequence for timeframe {tf}")
            sequences.append(tensor.to(self.device))

        cluster_id = torch.zeros(sequences[0].shape[0], dtype=torch.long, device=self.device)  # fallback=0
        return sequences, cluster_id

    def predict(self, features: Dict[str, Any]) -> Tuple[float, float, float]:
        """
        Основной метод предсказания.
        features: {"sequences": {tf: tensor}, "features": tensor}
        Возвращает prob_long, prob_short, uncertainty
        """
        sequences, cluster_id = self._prepare_inputs(features)

        self.model.train()
        probs = []
        with torch.no_grad():
            for _ in range(self.mc_passes):
                prob, _ = self.model(sequences, cluster_id)  # ← исправлено
                probs.append(prob.squeeze())
        probs = torch.stack(probs)

        if self.ensemble_models:
            ensemble_probs = []
            for model in self.ensemble_models:
                model.train()
                for _ in range(self.mc_passes):
                    prob, _ = model(sequences, cluster_id)  # ← исправлено
                    ensemble_probs.append(prob.squeeze())
            ensemble_probs = torch.stack(ensemble_probs)
            probs = torch.cat([probs, ensemble_probs], dim=0)

        mean_prob = probs.mean(dim=0)
        uncertainty = probs.std(dim=0).mean().item()

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
        sequences, cluster_id = self._prepare_inputs(features)

        with torch.no_grad():
            prob, _ = self.model(sequences, cluster_id)  # ← исправлено
        return prob.squeeze().item()

    def calibrate(self, prob: float) -> float:
        return prob

    def load_checkpoint(self, path: str):
        try:
            self.model.load_state_dict(torch.load(path, map_location=self.device))
            logger.info(f"Checkpoint loaded: {path}")
        except Exception as e:
            logger.error(f"Failed to load checkpoint {path}: {e}")