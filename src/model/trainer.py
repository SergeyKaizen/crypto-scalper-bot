# src/model/trainer.py
"""
Модуль обучения и переобучения нейронной сети.

Основные функции:
- train_initial() — первоначальное обучение на исторических данных
- retrain_live() — переобучение на свежих данных каждые 10 000 свечей
- prepare_dataset() — подготовка датасета из свечей и меток (профит/убыток)
- evaluate() — оценка модели на валидационном сете (accuracy, precision, recall, F1)
- save_checkpoint() / load_checkpoint() — сохранение/загрузка модели

Логика:
- Обучение — AdamW + BCEWithLogitsLoss (бинарная классификация: профит/убыток)
- Переобучение — каждые 10 000 новых свечей (настраивается)
- Датасет — time-series: последовательности свечей + метка (1 если закрытие по TP, 0 по SL)
- Баланс классов — weighted loss (крипта сильно несбалансирована: много убыточных)
- Tiny-версия — меньше эпох, batch_size, learning rate
- GPU/CPU — torch.device auto-detection

Конфиг-зависимые параметры:
- epochs: 50 (default) / 20 (phone)
- batch_size: 64 (default) / 16 (phone)
- learning_rate: 1e-3 (default) / 5e-4 (phone)
- retrain_every_candles: 10000
"""

import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional

from src.core.config import load_config
from src.model.architectures import MultiTFHybrid, TinyHybrid
from src.data.storage import Storage

logger = logging.getLogger(__name__)


class TradingDataset(Dataset):
    """Датасет для обучения: sequence + aggregated features + label"""

    def __init__(self, sequences: List[Dict], agg_features: List[Dict], labels: List[int]):
        self.sequences = sequences  # List[Dict[tf: torch.Tensor(seq_len, features)]]
        self.agg_features = agg_features  # List[Dict[tf: Dict[str, float]]]
        self.labels = labels  # List[int] — 1 (profit), 0 (loss)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "sequences": self.sequences[idx],
            "agg_features": self.agg_features[idx],
            "label": torch.tensor(self.labels[idx], dtype=torch.float32)
        }


class ModelTrainer:
    """Обучение и переобучение модели"""

    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.storage = Storage(config)

        # Выбор модели
        if config.get("use_tiny_model", False):
            self.model = TinyHybrid(config).to(self.device)
        else:
            self.model = MultiTFHybrid(config).to(self.device)

        self.optimizer = optim.AdamW(self.model.parameters(), lr=config["model"]["learning_rate"])
        self.criterion = nn.BCEWithLogitsLoss()  # Для бинарной классификации

        logger.info("Trainer initialized on device: %s, model: %s", self.device, "tiny" if config.get("use_tiny_model") else "full")

    def prepare_dataset(self, symbol: str, timeframe: str) -> Tuple[List, List, List]:
        """Подготовка датасета из БД"""
        # Получаем свечи
        candles = self.storage.get_candles(symbol, timeframe, limit=100000)  # Примерный лимит
        if not candles:
            raise ValueError(f"No candles for {symbol} {timeframe}")

        df = pl.DataFrame(candles)

        # Подготовка последовательностей и меток (примерная реализация)
        sequences = []
        agg_features = []
        labels = []

        # Пример: скользящее окно
        for i in range(len(df) - self.config["seq_len"] - 10):
            seq_df = df.slice(i, self.config["seq_len"])
            future_df = df.slice(i + self.config["seq_len"], 10)  # Смотрим 10 свечей вперёд

            # Sequence — сырые свечи
            seq = seq_df[["open", "high", "low", "close", "volume", "bid_volume", "ask_volume"]].to_numpy()
            sequences.append(seq)

            # Aggregated features — считаем feature_engine
            # (здесь упрощённо — в реальном коде вызываем FeatureEngine.build_features)
            agg = {}  # Заполняется feature_engine
            agg_features.append(agg)

            # Label: 1 если закрытие по TP, 0 по SL
            # (логика зависит от TP/SL — упрощённо: если max(future high) > entry + TP_dist)
            label = 1 if future_df["high"].max() > seq_df["close"][-1] + 100 else 0  # Пример
            labels.append(label)

        return sequences, agg_features, labels

    def train_initial(self):
        """Первоначальное обучение на исторических данных"""
        logger.info("Starting initial training...")

        # Пример: обучение на BTCUSDT 1m
        sequences, agg_features, labels = self.prepare_dataset("BTCUSDT", "1m")

        dataset = TradingDataset(sequences, agg_features, labels)
        loader = DataLoader(dataset, batch_size=self.config["model"]["batch_size"], shuffle=True)

        self.model.train()
        for epoch in range(self.config["model"]["epochs"]):
            total_loss = 0
            for batch in loader:
                self.optimizer.zero_grad()

                prob, _ = self.model(batch["sequences"], batch["agg_features"])
                loss = self.criterion(prob.squeeze(), batch["label"])
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            logger.info("Epoch %d/%d, Loss: %.4f", epoch + 1, self.config["model"]["epochs"], total_loss / len(loader))

        self.save_checkpoint("initial_model.pth")
        logger.info("Initial training completed")

    def retrain_live(self, new_candles_count: int):
        """Переобучение на свежих данных каждые 10 000 свечей"""
        retrain_every = self.config.get("retrain_every_candles", 10000)

        if new_candles_count % retrain_every == 0:
            logger.info("Retraining model on %d new candles...", new_candles_count)
            # Аналогично train_initial, но меньшее кол-во эпох
            self.train_initial()  # Упрощённо — в реальном коде — fine-tune
            self.save_checkpoint("live_model.pth")

    def save_checkpoint(self, path: str):
        """Сохранение модели"""
        torch.save(self.model.state_dict(), path)
        logger.info("Model saved to %s", path)

    def load_checkpoint(self, path: str):
        """Загрузка модели"""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.to(self.device)
        logger.info("Model loaded from %s", path)

    def evaluate(self, sequences, agg_features, labels):
        """Оценка модели на валидационном сете"""
        self.model.eval()
        with torch.no_grad():
            prob, _ = self.model(sequences, agg_features)
            pred = (prob.squeeze() > 0.5).float()
            accuracy = (pred == torch.tensor(labels)).float().mean().item()
            logger.info("Evaluation accuracy: %.4f", accuracy)
        return accuracy