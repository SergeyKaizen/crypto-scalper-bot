# src/model/inference.py
"""
Модуль выполнения inference модели в реальном времени.

Основные функции:
- model_inference() — основной предикт модели на текущих данных
- run_quiet_inference() — выполнение inference в тихом режиме (по таймеру TF)
- should_run_quiet() — проверка таймера для каждого TF
- process_signal() — обработка сигнала: проверка порога, создание Signal, открытие позиции

Логика:
- Inference запускается:
  - при любой аномалии (C/V/CV)
  - в тихом режиме (каждые N свечей по TF)
- Порог вероятности: config["prob_threshold"] (обычный) / config["quiet_prob_threshold"] (тихий)
- Выход модели: prob (0..1) + exp_return (опционально)
- Если prob > порог → создаётся Signal → передаётся в trading/order_executor
- Тихий режим: prob > quiet_prob_threshold (обычно выше обычного) → сигнал типа "Q"
- Q-сигналы имеют вес 0.7 в PR (меньше влияют на статистику)

Зависимости:
- src/model/architectures.py — модель
- src/features/feature_engine.py — подготовка признаков
- src/trading/order_executor.py — передача сигнала на открытие позиции
- config — quiet_mode, quiet_prob_threshold, quiet_inference_every_by_tf
"""

import logging
import time
from datetime import datetime
from typing import Dict, Tuple, Optional

import torch

from src.core.config import load_config
from src.core.enums import AnomalyType, Direction
from src.core.types import Signal
from src.model.architectures import MultiTFHybrid
from src.features.feature_engine import FeatureEngine
from src.trading.order_executor import process_signal  # Передача сигнала на открытие

logger = logging.getLogger(__name__)

class InferenceEngine:
    """Движок inference — предсказания модели в live-режиме"""

    def __init__(self, config: Dict, model: MultiTFHybrid):
        self.config = config
        self.model = model
        self.feature_engine = FeatureEngine(config)
        self.device = next(model.parameters()).device

        # Последний запуск тихого режима по TF (timestamp)
        self.last_quiet_run = {tf: 0 for tf in config["timeframes"]}

        logger.info("InferenceEngine initialized on device: %s", self.device)

    @torch.no_grad()
    def model_inference(self, sequences: Dict[str, torch.Tensor], agg_features: Dict[str, Dict]) -> Tuple[float, Optional[float]]:
        """
        Основной предикт модели

        Args:
            sequences: Dict[tf: Tensor(batch=1, seq_len, features)]
            agg_features: Dict[tf: Dict[str, float]]

        Returns:
            probability: float (0..1)
            expected_return: float or None
        """
        self.model.eval()

        # Подготовка входа
        seq_tensors = []
        for tf in self.config["timeframes"][:self.config["max_tf"]]:
            if tf in sequences:
                seq_tensors.append(sequences[tf].to(self.device))
            else:
                # Заполняем нулями, если TF отсутствует
                seq_tensors.append(torch.zeros(1, self.config["seq_len"], sequences[list(sequences.keys())[0]].shape[-1]).to(self.device))

        seq_batch = torch.stack(seq_tensors, dim=1)  # (batch=1, num_tf, seq_len, features)

        # Agg features — можно добавить в модель (в текущей версии не используются напрямую)
        prob, exp_return = self.model(seq_batch, agg_features)

        prob = prob.item()  # float
        exp_return = exp_return.item() if exp_return is not None else None

        logger.debug("Inference: prob=%.4f, expected_return=%.2f%%", prob, exp_return or 0)

        return prob, exp_return

    def should_run_quiet(self, tf: str, current_time: int) -> bool:
        """Проверка: пора ли запускать тихий inference для данного TF"""
        if not self.config["quiet_mode"]:
            return False

        interval = self.config["quiet_inference_every_by_tf"].get(tf, 10)  # свечи
        last_run = self.last_quiet_run.get(tf, 0)

        # Интервал в свечах — переводим в timestamp (примерно)
        candle_duration_ms = {
            '1m': 60_000,
            '3m': 180_000,
            '5m': 300_000,
            '10m': 600_000,
            '15m': 900_000
        }.get(tf, 60_000)

        if current_time - last_run >= interval * candle_duration_ms:
            self.last_quiet_run[tf] = current_time
            return True
        return False

    async def run_quiet_inference(self, data: Dict[str, pl.DataFrame], current_time: int) -> Optional[Signal]:
        """Запуск тихого режима (если пора)"""
        signals = []

        for tf in self.config["timeframes"][:self.config["max_tf"]]:
            if not self.should_run_quiet(tf, current_time):
                continue

            df = data.get(tf)
            if df is None or len(df) < self.config["seq_len"]:
                continue

            features_dict = await self.feature_engine.build_features({tf: df.tail(self.config["seq_len"] * 2)})

            sequences = features_dict["sequences"]
            agg_features = features_dict["features"]

            prob, exp_return = self.model_inference(sequences, agg_features)

            if prob > self.config["quiet_prob_threshold"]:
                signal = Signal(
                    timestamp=datetime.now(),
                    symbol=data[list(data.keys())[0]]["symbol"][0],  # Берём символ из первого TF
                    timeframe=tf,
                    window=100,  # Основное окно
                    anomaly_type=AnomalyType.QUIET.value,
                    direction="L" if exp_return > 0 else "S" if exp_return < 0 else "LS",  # Пример
                    probability=prob,
                    expected_return=exp_return
                )
                signals.append(signal)
                logger.info("Quiet signal generated: TF=%s, prob=%.4f", tf, prob)

        return signals  # Возвращаем список сигналов (может быть несколько TF)

    async def process_new_data(self, data: Dict[str, pl.DataFrame]) -> List[Signal]:
        """Основная точка входа — вызывается из live_loop"""
        signals = []

        current_time = int(datetime.now().timestamp() * 1000)

        # 1. Проверяем аномалии
        for tf, df in data.items():
            if len(df) < 100:
                continue

            anomaly_flags = self.anomaly_detector.detect(df.tail(100))

            if any(anomaly_flags.values()):  # C, V, CV
                features_dict = await self.feature_engine.build_features({tf: df.tail(self.config["seq_len"] * 2)})
                sequences = features_dict["sequences"]
                agg_features = features_dict["features"]

                prob, exp_return = self.model_inference(sequences, agg_features)

                if prob > self.config["prob_threshold"]:
                    # Определяем направление
                    direction = "L" if exp_return > 0 else "S" if exp_return < 0 else "LS"

                    signal = Signal(
                        timestamp=datetime.now(),
                        symbol=df["symbol"][0],
                        timeframe=tf,
                        window=100,
                        anomaly_type=next(k for k, v in anomaly_flags.items() if v),  # C/V/CV
                        direction=direction,
                        probability=prob,
                        expected_return=exp_return
                    )
                    signals.append(signal)
                    logger.info("Anomaly signal: TF=%s, type=%s, prob=%.4f", tf, signal.anomaly_type, prob)

        # 2. Тихий режим
        quiet_signals = await self.run_quiet_inference(data, current_time)
        if quiet_signals:
            signals.extend(quiet_signals)

        return signals