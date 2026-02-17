# src/model/inference.py
"""
Модуль реального инференса (предсказания) для скальпинг-бота.

Ключевые принципы из ТЗ:
- Предсказание запускается только при аномалии (candle/volume/cv) или в quiet-режиме (q_condition)
- Модель видит последовательности свечей на 4 окнах (24/50/74/100) и всех TF (1m/3m/5m/10m/15m)
- Вход: 16 признаков (12 базовых + 4 бинарных условия)
- Учёт веса сценария из ScenarioTracker (бинарная статистика)
- Выявление лучшего TF для открытия позиции
- Отдельные пороги вероятности: min_prob_anomaly и min_prob_quiet (настраиваются в конфиге)
- Поддержка phone_tiny (квантизация int8, меньше окон)
- Возврат: prob, raw_prob, best_tf, anomaly_type, signal, детали

Логика:
1. Получаем свежие свечи со всех TF (resampler → storage)
2. Проверяем аномалии/q_condition
3. Готовим вход {tf: {window: tensor[1, seq_len, 16]}}
4. Инференс по TF → средняя или max вероятность
5. Корректировка весом сценария
6. Signal = adjusted_prob >= min_prob (по типу аномалии)
"""

import torch
import torch.nn.functional as F
import os
from typing import Dict, Any, Optional
import polars as pl
import numpy as np

from src.model.architectures import create_model
from src.features.feature_engine import FeatureEngine
from src.features.anomaly_detector import AnomalyDetector
from src.model.scenario_tracker import ScenarioTracker
from src.data.resampler import Resampler
from src.data.storage import Storage
from src.core.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

class InferenceEngine:
    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Загружаем модель
        self.model = create_model(config).to(self.device)
        self._load_best_checkpoint()

        # Квантизация на телефоне (экономия памяти/CPU)
        if config["hardware_mode"] == "phone_tiny":
            self.model = torch.quantization.quantize_dynamic(
                self.model, {nn.Linear, nn.GRU}, dtype=torch.qint8
            )

        self.model.eval()

        self.feature_engine = FeatureEngine(config)
        self.anomaly_detector = AnomalyDetector(config)
        self.scenario_tracker = ScenarioTracker(config)
        self.resampler = Resampler(config)
        self.storage = Storage(config)

        # Пороги вероятности (отдельные для обычных аномалий и quiet-режима)
        self.min_prob_anomaly = config["model"].get("min_prob_anomaly", 0.65)
        self.min_prob_quiet = config["model"].get("min_prob_quiet", 0.78)

        # Quiet-режим включён?
        self.quiet_mode = config.get("quiet_mode", False)

    def _load_best_checkpoint(self):
        """Загружает лучшую модель из чекпоинтов"""
        checkpoint_path = os.path.join(
            self.config["paths"]["checkpoints"], "best.pth"
        )
        if os.path.exists(checkpoint_path):
            state = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(state["model_state_dict"])
            logger.info(f"Загружена лучшая модель: {checkpoint_path}")
        else:
            logger.warning("best.pth не найден → используем случайные веса (нужна тренировка)")

    @torch.no_grad()
    def predict(self, symbol: str) -> Dict[str, Any]:
        """
        Основной метод предсказания для одной монеты.

        Возвращает:
        {
            "prob": float,                  # итоговая скорректированная вероятность
            "raw_prob": float,              # сырая вероятность от модели
            "scenario_weight": float,
            "best_tf": str,
            "anomaly_type": str or None,
            "signal": bool,
            "details": dict
        }
        """
        # 1. Получаем свежие свечи со всех TF (resampler → storage)
        candles = {}
        for tf in ["1m", "3m", "5m", "10m", "15m"]:
            df = self.resampler.get_window(tf, 100)
            if df is None or len(df) < 100:
                df = self.storage.load_candles(symbol, tf, limit=100)
            if df is not None and len(df) >= 100:
                candles[tf] = df.tail(100).sort("open_time")

        if not candles:
            return {"prob": 0.0, "signal": False, "error": "недостаточно свечей"}

        # 2. Проверяем аномалии/q_condition на всех TF
        anomalies = {}
        for tf, df in candles.items():
            result = self.anomaly_detector.detect(df)
            if result["anomaly_type"] or (self.quiet_mode and result["q_condition"]):
                anomalies[tf] = result

        if not anomalies:
            return {"prob": 0.0, "signal": False, "reason": "нет аномалий и quiet-режим выключен"}

        # 3. Готовим вход для модели
        inputs = self._prepare_model_input(candles, anomalies)
        if not inputs:
            return {"prob": 0.0, "signal": False, "error": "не удалось подготовить вход"}

        # 4. Инференс по TF
        inputs_device = {tf: {int(w): t.to(self.device) for w, t in ws.items()}
                         for tf, ws in inputs.items()}

        probs_per_tf = {}
        with torch.no_grad():
            for tf, tf_inputs in inputs_device.items():
                tf_prob = self.model({tf: tf_inputs}).item()
                probs_per_tf[tf] = tf_prob

        # 5. Финальная вероятность — берём максимум (агрессивнее и логичнее для скальпа)
        raw_prob = max(probs_per_tf.values()) if probs_per_tf else 0.0

        # 6. Учёт веса сценария (бинарная статистика)
        last_features = self.feature_engine.get_last_features(symbol, "1m")
        scenario_key = self.scenario_tracker.get_scenario_key(last_features)
        scenario_weight = self.scenario_tracker.get_scenario_weight(scenario_key)

        adjusted_prob = raw_prob * scenario_weight  # 0.5–2.0 диапазон

        # 7. Лучший TF
        best_tf = max(probs_per_tf, key=probs_per_tf.get, default=None)

        # 8. Тип аномалии и порог вероятности
        anomaly_info = anomalies.get(best_tf, {})
        anomaly_type = anomaly_info.get("anomaly_type")
        is_quiet = anomaly_info.get("q_condition", False)

        min_prob = self.min_prob_quiet if is_quiet else self.min_prob_anomaly
        signal = adjusted_prob >= min_prob

        result = {
            "prob": round(adjusted_prob, 4),
            "raw_prob": round(raw_prob, 4),
            "scenario_weight": round(scenario_weight, 3),
            "best_tf": best_tf,
            "anomaly_type": anomaly_type,
            "is_quiet": is_quiet,
            "signal": signal,
            "details": {
                "probs_per_tf": {tf: round(p, 4) for tf, p in probs_per_tf.items()},
                "scenario_key": scenario_key,
                "min_prob_used": min_prob
            }
        }

        if signal:
            logger.info(f"[СИГНАЛ] {symbol} | prob={adjusted_prob:.4f} | tf={best_tf} | "
                        f"type={anomaly_type} | quiet={is_quiet}")

        return result

    def _prepare_model_input(self, candles: Dict[str, pl.DataFrame], anomalies: Dict) -> Dict:
        """
        Готовит вход для модели: {tf: {window_size: tensor[1, seq_len, 16]}}
        Только для TF с аномалией или q_condition
        """
        inputs = {}

        for tf, df in candles.items():
            if tf not in anomalies:
                continue

            windows = {}
            for size in [24, 50, 74, 100]:
                window_df = df.tail(size)
                if len(window_df) < size // 2:
                    continue

                feat_array = self.feature_engine.compute_sequence_features(window_df)
                tensor = torch.from_numpy(feat_array).float().unsqueeze(0)  # [1, seq, 16]
                windows[size] = tensor

            if windows:
                inputs[tf] = windows

        return inputs


if __name__ == "__main__":
    config = load_config()
    engine = InferenceEngine(config)
    
    result = engine.predict("BTCUSDT")
    print(result)