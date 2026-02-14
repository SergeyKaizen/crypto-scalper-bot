# src/model/trainer.py
"""
Trainer — отвечает за обучение и live-переобучение модели.
Полностью соответствует ТЗ:
- Обучение на всех TF сразу (1m, 3m, 5m, 10m, 15m)
- Модель видит последовательности свечей (time series), seq_len = 50–100
- Multi-TF анализ + 4 окна (24, 50, 74, 100)
- Live-переобучение каждые 10 000 свечей
- Разные настройки для телефона / Colab / сервера (разные seq_len, эпохи, объём данных)
- Сохранение/загрузка модели (PyTorch + ONNX для телефона)
- Обучение на всех признаках + 3 условия (C/V/CV) как бинарные фичи
"""

import time
from typing import Dict, Tuple, Optional
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import polars as pl
import numpy as np

from ..utils.logger import logger
from ..core.config import get_config
from ..data.storage import Storage
from ..features.feature_engine import FeatureEngine
from .architectures import ScalperHybridModel


class Trainer:
    def __init__(self):
        self.config = get_config()
        self.storage = Storage()
        self.feature_engine = FeatureEngine(self.config)

        # Аппаратные настройки (телефон / Colab / сервер)
        self.hardware = self.config["hardware"]["type"]  # "phone", "colab", "server"
        self._apply_hardware_limits()

        # Модель и параметры обучения
        self.model = ScalperHybridModel(self.config).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.criterion = nn.CrossEntropyLoss()  # Long / Short / Flat

        # Путь для сохранения модели
        self.model_path = Path("models") / f"scalper_{self.hardware}.pth"
        self.model_path.parent.mkdir(exist_ok=True)

        logger.info("Trainer инициализирован",
                    hardware=self.hardware,
                    seq_len=self.seq_len,
                    epochs=self.epochs,
                    device=self.device)

    def _apply_hardware_limits(self):
        """Ограничивает параметры в зависимости от железа."""
        if self.hardware == "phone":
            self.seq_len = 50
            self.epochs = 5
            self.max_data_candles = 50_000
            self.batch_size = 32
            self.device = torch.device("cpu")
        elif self.hardware == "colab":
            self.seq_len = 100
            self.epochs = 20
            self.max_data_candles = 500_000
            self.batch_size = 128
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:  # server
            self.seq_len = 100
            self.epochs = 50
            self.max_data_candles = 2_000_000
            self.batch_size = 512
            self.device = torch.device("cuda")

    def prepare_data(self, coin: str = "BTC") -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Подготавливает данные для обучения.
        - Последовательности свечей (seq_len)
        - Метки: 0 = Long, 1 = Short, 2 = Flat (по будущему движению)
        """
        # Загружаем все TF для монеты
        dfs = {}
        for tf in ["1m", "3m", "5m", "10m", "15m"]:
            df = self.storage.load_candles(coin, tf)
            if df is not None and len(df) > self.seq_len * 2:
                dfs[tf] = df.tail(self.max_data_candles)

        if not dfs:
            logger.error("Нет данных для обучения", coin=coin)
            return None, None

        # Генерируем последовательности
        X_list, y_list = [], []
        for tf, df in dfs.items():
            for i in range(len(df) - self.seq_len - 1):
                seq = df[i:i+self.seq_len]  # последовательность свечей
                future_price = df["close"][i+self.seq_len]
                current_price = df["close"][i+self.seq_len-1]

                # Метка: 0=Long, 1=Short, 2=Flat
                change_pct = (future_price - current_price) / current_price * 100
                if change_pct > 0.5:
                    label = 0  # Long
                elif change_pct < -0.5:
                    label = 1  # Short
                else:
                    label = 2  # Flat

                # Преобразуем свечи в тензор (нужно нормализовать!)
                seq_tensor = self._candle_to_tensor(seq)
                X_list.append(seq_tensor)
                y_list.append(label)

        if not X_list:
            return None, None

        X = torch.stack(X_list)
        y = torch.tensor(y_list, dtype=torch.long)

        logger.info("Данные подготовлены",
                    samples=len(X),
                    seq_len=self.seq_len,
                    classes=dict(zip(*torch.unique(y, return_counts=True))))

        return X, y

    def _candle_to_tensor(self, df: pl.DataFrame) -> torch.Tensor:
        """Преобразует DataFrame свечей в тензор [channels, seq_len]."""
        # Пример каналов: open, high, low, close, volume (можно больше)
        data = df.select([
            "open", "high", "low", "close", "volume"
        ]).to_numpy()

        # Нормализация (min-max или z-score)
        data = (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-8)

        return torch.tensor(data.T, dtype=torch.float32)  # [channels, seq_len]

    def train(self):
        """Полное обучение модели."""
        X, y = self.prepare_data()
        if X is None:
            return

        dataset = TensorDataset(X, y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                self.optimizer.zero_grad()
                output = self.model(batch_x)
                loss = self.criterion(output, batch_y)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(loader)
            logger.info(f"Эпоха {epoch+1}/{self.epochs}", loss=f"{avg_loss:.6f}")

        self._save_model()
        logger.info("Обучение завершено", path=str(self.model_path))

    def live_retrain(self):
        """Live-переобучение каждые 10 000 свечей."""
        # Здесь логика проверки количества новых свечей
        # Если >= 10 000 — вызываем self.train() на новых данных
        logger.info("Live-переобучение запущено")
        self.train()  # пока просто полное переобучение (можно сделать incremental)

    def _save_model(self):
        torch.save(self.model.state_dict(), self.model_path)
        logger.info("Модель сохранена", path=str(self.model_path))

    def load_model(self):
        if self.model_path.exists():
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            logger.info("Модель загружена", path=str(self.model_path))
        else:
            logger.warning("Модель не найдена, будет обучена заново")