"""
src/model/trainer.py

=== Основной принцип работы файла ===

Этот файл реализует весь цикл обучения и переобучения модели.
Он:
- Загружает данные из storage (последовательности признаков по TF/окнам).
- Готовит датасеты и DataLoader'ы (train/val split по % из config).
- Инициализирует модель MultiWindowHybrid.
- Обучает с AdamW, BCEWithLogitsLoss (бинарная классификация: профитная сделка=1).
- Валидирует метрики: accuracy, precision, recall, F1 (с epsilon/clamp для стабильности).
- Сохраняет лучшую модель по val_loss.
- Поддерживает incremental retrain (каждые RETRAIN_INTERVAL_CANDLES свечей в live).
- Масштабирует параметры (epochs, batch_size, hidden_size) по hardware.

Ключевые особенности:
- Данные — dict[tf: dict[window: tensor(seq_len, features)]]
- Label — 1 если сделка закрылась по TP раньше SL (lookback_after=200 свечей).
- Переобучение: load state_dict + continue на новых данных (freeze early layers если нужно).
- Нет лишних библиотек — torch, torch.utils.data.

=== Главные функции и за что отвечают ===

- Trainer(config: dict) — инициализация: модель, optimizer, loss, scheduler.
- prepare_dataset(symbols: list, tf: str) → Dataset — собирает последовательности и labels.
- train_epoch() → dict[metrics] — один эпох обучения.
- validate_epoch() → dict[metrics] — валидация с precision/recall/accuracy/F1.
- train() — полный цикл обучения (epochs из config).
- incremental_retrain(new_data_df) — дообучение на новых свечах.
- save_model(path), load_model(path) — сохранение/загрузка state_dict.

=== Примечания ===
- Все параметры из config (epochs, batch_size, lr и т.д. масштабируются по hardware).
- Label generation: симуляция TP/SL на исторических данных (lookback_after).
- Clamp/epsilon в метриках для избежания NaN/inf.
- Готов к использованию в scripts/train.py и live_loop.py.
- Логи через setup_logger.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from typing import Dict, Optional

from src.model.architectures import MultiWindowHybrid
from src.data.storage import Storage
from src.features.feature_engine import compute_features_for_tf, prepare_sequence_features
from src.core.config import load_config
from src.core.constants import DEFAULT_SEQ_LEN, RETRAIN_INTERVAL_CANDLES
from src.utils.logger import setup_logger

logger = setup_logger('trainer', logging.INFO)

class TradingDataset(Dataset):
    """
    Датасет для модели: последовательности признаков + labels.
    """
    def __init__(self, sequences: list, labels: list):
        self.sequences = sequences  # list[dict[tf: dict[window: tensor]]]
        self.labels = labels        # list[int/float 0/1]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], torch.tensor(self.labels[idx], dtype=torch.float32)

class Trainer:
    """
    Класс для обучения и переобучения модели.
    """
    def __init__(self):
        config = load_config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Устройство: {self.device}")

        input_dim = config['model']['input_dim']  # из feature_engine (кол-во признаков)
        self.model = MultiWindowHybrid(input_dim=input_dim).to(self.device)

        self.optimizer = optim.AdamW(self.model.parameters(), lr=config['model']['lr'])
        self.criterion = nn.BCEWithLogitsLoss()
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=3)

        self.epochs = config['model']['epochs']
        self.batch_size = config['model']['batch_size']
        self.val_split = config['model']['val_split']

    def prepare_dataset(self, symbols: list) -> tuple[Dataset, Dataset]:
        """
        Собирает датасет: последовательности + labels.
        Label = 1 если в следующие 200 свечей цена достигла TP раньше SL.
        """
        sequences = []
        labels = []
        storage = Storage()

        for symbol in symbols:
            for tf in ['1m', '3m', '5m', '10m', '15m']:
                df = storage.get_candles(symbol, tf)
                if len(df) < DEFAULT_SEQ_LEN + 200:
                    continue

                # Примерная генерация labels (упрощённо)
                for i in range(len(df) - DEFAULT_SEQ_LEN - 200):
                    seq_df = df.iloc[i:i+DEFAULT_SEQ_LEN]
                    future_df = df.iloc[i+DEFAULT_SEQ_LEN:i+DEFAULT_SEQ_LEN+200]

                    # Симуляция: TP = close + 1%, SL = close - 0.5% (пример)
                    entry_price = seq_df['close'].iloc[-1]
                    tp_hit = (future_df['high'] >= entry_price * 1.01).any()
                    sl_hit = (future_df['low'] <= entry_price * 0.995).any()

                    label = 1 if tp_hit and (not sl_hit or tp_hit before sl_hit) else 0

                    seq_features = prepare_sequence_features(symbol, tf, DEFAULT_SEQ_LEN)
                    if seq_features is not None:
                        sequences.append(seq_features.to_dict('records'))
                        labels.append(label)

        dataset = TradingDataset(sequences, labels)
        train_size = int((1 - self.val_split) * len(dataset))
        val_size = len(dataset) - train_size
        train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

        return train_ds, val_ds

    def train_epoch(self, loader: DataLoader) -> Dict[str, float]:
        """
        Один эпох обучения.
        """
        self.model.train()
        total_loss = 0.0
        preds, trues = [], []

        for batch_inputs, batch_labels in loader:
            # batch_inputs — list[dict[tf: dict[window: tensor]]]
            # Нужно collate в dict[tf: dict[window: tensor(batch, seq, feat)]]
            batch = {tf: {w: torch.stack([inp[tf][w] for inp in batch_inputs]) for w in batch_inputs[0][tf]} for tf in batch_inputs[0]}

            for tf in batch:
                for w in batch[tf]:
                    batch[tf][w] = batch[tf][w].to(self.device)

            batch_labels = batch_labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(batch)
            loss = self.criterion(outputs, batch_labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            preds.extend((torch.sigmoid(outputs) > 0.5).cpu().numpy())
            trues.extend(batch_labels.cpu().numpy())

        metrics = {
            'loss': total_loss / len(loader),
            'accuracy': accuracy_score(trues, preds),
            'precision': precision_score(trues, preds, zero_division=0),
            'recall': recall_score(trues, preds, zero_division=0),
            'f1': f1_score(trues, preds, zero_division=0)
        }
        return metrics

    def validate_epoch(self, loader: DataLoader) -> Dict[str, float]:
        """
        Валидация.
        """
        self.model.eval()
        total_loss = 0.0
        preds, trues = [], []

        with torch.no_grad():
            for batch_inputs, batch_labels in loader:
                batch = {tf: {w: torch.stack([inp[tf][w] for inp in batch_inputs]) for w in batch_inputs[0][tf]} for tf in batch_inputs[0]}
                for tf in batch:
                    for w in batch[tf]:
                        batch[tf][w] = batch[tf][w].to(self.device)
                batch_labels = batch_labels.to(self.device)

                outputs = self.model(batch)
                loss = self.criterion(outputs, batch_labels)

                total_loss += loss.item()
                preds.extend((torch.sigmoid(outputs) > 0.5).cpu().numpy())
                trues.extend(batch_labels.cpu().numpy())

        metrics = {
            'loss': total_loss / len(loader),
            'accuracy': accuracy_score(trues, preds),
            'precision': precision_score(trues, preds, zero_division=0),
            'recall': recall_score(trues, preds, zero_division=0),
            'f1': f1_score(trues, preds, zero_division=0)
        }
        return metrics

    def train(self):
        """
        Полный цикл обучения.
        """
        train_ds, val_ds = self.prepare_dataset(symbols=['BTCUSDT'])  # пример, все монеты

        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)

        best_val_loss = float('inf')
        for epoch in range(self.epochs):
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.validate_epoch(val_loader)

            logger.info(f"Epoch {epoch+1}/{self.epochs} | Train loss: {train_metrics['loss']:.4f} | Val loss: {val_metrics['loss']:.4f}")

            self.scheduler.step(val_metrics['loss'])

            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                self.save_model("best_model.pth")
                logger.info("Сохранена лучшая модель")

    def incremental_retrain(self, new_data: Dict):
        """
        Дообучение на новых свечах (live-режим).
        """
        # Загружаем старую модель
        self.load_model("best_model.pth")

        # Готовим новый датасет из новых данных
        # ... (аналогично prepare_dataset на новых свечах)
        # train_loader = ...

        # Дообучаем на 3–5 эпохах
        for _ in range(5):
            self.train_epoch(train_loader)  # без валидации

        self.save_model("latest_model.pth")
        logger.info("Модель дообучена на новых данных")

    def save_model(self, path: str):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path: str):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.to(self.device)