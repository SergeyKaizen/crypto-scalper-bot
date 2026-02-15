# src/model/inference.py
"""
Модуль реального инференса (предсказания) для скальпинг-бота.

Ключевые требования из ТЗ:
- Предсказание запускается только при аномалии (candle / volume / cv) или в quiet-режиме (q_condition)
- Модель видит последовательности свечей на 4 окнах (24/50/74/100) и всех TF
- Данные подаются как time-series sequences (16 признаков: 12 базовых + 4 бинарных условия)
- Выявлять, на каком TF в момент аномалии лучше открывать позицию
- Учёт весов сценариев из ScenarioTracker (бинарная статистика)
- Поддержка phone_tiny / colab / server (разные модели, квантизация на телефоне)
- Авто-расчёт TP/SL риск-профита после сигнала
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
        
        if self.config["hardware_mode"] == "phone_tiny":
            self.model = torch.quantization.quantize_dynamic(
                self.model, {nn.Linear, nn.GRU}, dtype=torch.qint8
            )
        
        self.model.eval()
        
        self.feature_engine = FeatureEngine(config)
        self.anomaly_detector = AnomalyDetector(config)
        self.scenario_tracker = ScenarioTracker(config)
        self.resampler = Resampler(config)
        self.storage = Storage(config)
        
        self.min_prob = config["model"].get("min_prob", 0.65)
        self.quiet_mode = config.get("quiet_mode", False)

    def _load_best_checkpoint(self):
        checkpoint_path = os.path.join(
            self.config["paths"]["checkpoints"], "best.pth"
        )
        if os.path.exists(checkpoint_path):
            state = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(state["model_state_dict"])
            logger.info(f"Загружена лучшая модель из {checkpoint_path}")
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
        # 1. Получаем свежие свечи со всех TF
        candles = {}
        for tf in ["1m", "3m", "5m", "10m", "15m"]:
            df = self.resampler.get_window(tf, 100)
            if df is None or len(df) < 100:
                df = self.storage.load_candles(symbol, tf, limit=100)
            if df is not None and len(df) >= 100:
                candles[tf] = df.tail(100).sort("open_time")

        if not candles:
            return {"prob": 0.0, "signal": False, "error": "недостаточно свечей"}

        # 2. Проверяем аномалии на всех TF
        anomalies = {}
        for tf, df in candles.items():
            result = self.anomaly_detector.detect(df)
            if result["anomaly_type"] or (self.quiet_mode and result["q_condition"]):
                anomalies[tf] = result

        if not anomalies:
            return {"prob": 0.0, "signal": False, "reason": "нет аномалий и quiet-режим выключен"}

        # 3. Собираем вход для модели
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

        # 5. Финальная вероятность (средняя по TF с аномалиями)
        final_prob = np.mean(list(probs_per_tf.values()))

        # 6. Учёт веса сценария (бинарная статистика)
        last_features = self.feature_engine.get_last_features(symbol, "1m")
        scenario_key = self.scenario_tracker.get_scenario_key(last_features)
        scenario_weight = self.scenario_tracker.get_scenario_weight(scenario_key)
        
        adjusted_prob = final_prob * scenario_weight  # 0.5–2.0 диапазон

        # 7. Лучший TF
        best_tf = max(probs_per_tf, key=probs_per_tf.get, default=None)

        # 8. Сигнал на открытие
        signal = adjusted_prob >= self.min_prob

        result = {
            "prob": round(adjusted_prob, 4),
            "raw_prob": round(final_prob, 4),
            "scenario_weight": round(scenario_weight, 3),
            "best_tf": best_tf,
            "anomaly_type": anomalies.get(best_tf, {}).get("anomaly_type"),
            "signal": signal,
            "details": {
                "probs_per_tf": {tf: round(p, 4) for tf, p in probs_per_tf.items()},
                "scenario_key": scenario_key
            }
        }

        if signal:
            logger.info(f"[СИГНАЛ] {symbol} | prob={adjusted_prob:.4f} | tf={best_tf} | type={result['anomaly_type']}")

        return result

    def _prepare_model_input(self, candles: Dict[str, pl.DataFrame], anomalies: Dict) -> Dict:
        """
        Подготавливает вход для модели: {tf: {window_size: tensor[1, seq_len, 16]}}
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