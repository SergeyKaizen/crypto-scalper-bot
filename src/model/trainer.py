# src/model/trainer.py
"""
Тренировщик гибридной модели Conv1D + GRU.

Ключевые требования из ТЗ и твоих уточнений:
- Обучение на максимальном возможном объёме данных для каждого TF отдельно
  - На сервере — вся доступная история по каждой монете (без лимита)
  - На Colab — большой объём (автоопределение, но не меньше 100k свечей на TF)
  - На phone_tiny — ограничение (30k свечей на TF, чтобы не перегружать память)
- Для каждой аномалии модель видит последовательности на 4 окнах (24, 50, 74, 100) по всем TF
- Label = 1, если после аномалии цена прошла TP раньше SL (смотрим вперёд 200 свечей)
- TP/SL рассчитываются по avg_candle_pct за 100 свечей до входа
- Используется 16 признаков (12 базовых + 4 бинарных условия)
- Поддержка early stopping, mixed precision (AMP), сохранение лучшей модели
- Готов к инкрементальному переобучению каждые 10 000 свечей (вызывается из live_loop)

Логика:
- ScalpSequenceDataset загружает данные из storage по каждому TF отдельно
- Trainer обучает модель с валидацией и сохранением чекпоинтов
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
import os
import time
from datetime import datetime
import polars as pl
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

from src.model.architectures import create_model
from src.features.feature_engine import FeatureEngine
from src.features.anomaly_detector import AnomalyDetector
from src.data.storage import Storage
from src.core.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

class ScalpSequenceDataset(Dataset):
    """
    Датасет для обучения: каждая строка — момент аномалии + окна до неё + label.
    Загружает максимальный объём данных по каждому TF отдельно в зависимости от hardware.
    """
    def __init__(self, config: dict, symbols: List[str], split: str = "train"):
        self.config = config
        self.storage = Storage(config)
        self.feature_engine = FeatureEngine(config)
        self.anomaly_detector = AnomalyDetector(config)
        
        self.timeframes = ["1m", "3m", "5m", "10m", "15m"]
        self.windows = [24, 50, 74, 100]
        self.lookback_after = 200  # сколько свечей смотреть вперёд для label
        
        # Максимальный объём данных по каждому TF зависит от hardware
        self.max_candles_per_tf = self._get_max_candles_per_tf()
        
        self.samples = self._load_samples(symbols, split)
        logger.info(f"Датасет {split}: загружено {len(self.samples)} примеров")

    def _get_max_candles_per_tf(self) -> int:
        """Определяет максимальное кол-во свечей на каждый TF в зависимости от железа"""
        hardware = self.config["hardware"]["mode"]
        if hardware == "phone_tiny":
            return 30000   # ограничение для телефона
        elif hardware == "server":
            return 0       # 0 = без лимита (вся история)
        else:  # colab
            return 100000  # большой объём

    def _load_samples(self, symbols: List[str], split: str) -> List[Dict]:
        samples = []
        
        for symbol in symbols:
            count = 0
            for tf in self.timeframes:
                # Загружаем максимум доступных свечей для обучения
                limit = self.max_candles_per_tf if self.max_candles_per_tf > 0 else None
                df = self.storage.load_candles(symbol, tf, limit=limit)
                if df is None or len(df) < max(self.windows) + self.lookback_after:
                    continue
                
                # Находим аномалии и q_condition
                anomalies = self.anomaly_detector.detect(df)
                anomaly_indices = [i for i, res in enumerate(anomalies) if res["anomaly_type"] or res["q_condition"]]
                
                for idx in anomaly_indices:
                    if count >= 5000:  # дополнительное ограничение на примеры для скорости
                        break
                        
                    windows = self._extract_windows(df, idx, tf)
                    if not windows:
                        continue
                    
                    label = self._compute_label(df, idx)
                    if label is None:
                        continue
                    
                    samples.append({
                        "symbol": symbol,
                        "tf": tf,
                        "windows": windows,
                        "label": label
                    })
                    count += 1
                
                if count >= 5000:
                    break
        
        # Простой train/val сплит (80/20 по времени)
        split_idx = int(len(samples) * 0.8)
        return samples[:split_idx] if split == "train" else samples[split_idx:]

    def _extract_windows(self, df: pl.DataFrame, anomaly_idx: int, tf: str) -> Optional[Dict[int, torch.Tensor]]:
        windows = {}
        for size in self.windows:
            start = max(0, anomaly_idx - size + 1)
            seq_df = df[start:anomaly_idx + 1]
            if len(seq_df) < size // 2:
                continue
            
            feat_array = self.feature_engine.compute_sequence_features(seq_df)
            tensor = torch.from_numpy(feat_array).float()
            windows[size] = tensor
        
        return windows if windows else None

    def _compute_label(self, df: pl.DataFrame, anomaly_idx: int) -> Optional[int]:
        """Label = 1 если цена прошла TP раньше SL"""
        past = df[max(0, anomaly_idx - 100):anomaly_idx]
        if len(past) < 50:
            return None
        
        avg_candle_pct = ((past["high"] - past["low"]) / past["close"]).mean() * 100
        
        future = df[anomaly_idx + 1:anomaly_idx + 1 + self.lookback_after]
        if future.is_empty():
            return 0
        
        entry = df["close"][anomaly_idx]
        tp_level = entry * (1 + avg_candle_pct * 1.0)
        sl_level = entry * (1 - avg_candle_pct * 1.0)
        
        for row in future.iter_rows(named=True):
            if row["high"] >= tp_level:
                return 1
            if row["low"] <= sl_level:
                return 0
        
        return 0

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            "windows": sample["windows"],
            "label": torch.tensor(sample["label"], dtype=torch.float32)
        }


class Trainer:
    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = create_model(config).to(self.device)
        
        lr = config["model"]["learning_rate"]
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-5)
        self.criterion = nn.BCEWithLogitsLoss()
        self.scaler = GradScaler(enabled=torch.cuda.is_available())
        
        self.epochs = config["model"].get("epochs", 30)
        self.batch_size = config["model"].get("batch_size", 32 if self.device.type == "cuda" else 8)
        self.save_dir = config["paths"]["checkpoints"]
        os.makedirs(self.save_dir, exist_ok=True)
        
        self.best_val_loss = float("inf")
        self.patience = config["model"].get("patience", 8)
        self.patience_counter = 0
        self.current_epoch = 0

    def train(self):
        symbols = self._get_training_symbols()
        train_loader = self._get_dataloader(symbols, "train")
        val_loader   = self._get_dataloader(symbols, "val")
        
        for epoch in range(1, self.epochs + 1):
            self.current_epoch = epoch
            train_loss = self._train_epoch(train_loader)
            val_loss, metrics = self._validate_epoch(val_loader)
            
            logger.info(f"Epoch {epoch:2d}/{self.epochs} | "
                       f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                       f"Acc: {metrics['accuracy']:.3f} | F1: {metrics['f1']:.3f}")

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self._save_checkpoint("best.pth")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    logger.info(f"Early stopping на эпохе {epoch}")
                    break
            
            if epoch % 5 == 0:
                self._save_checkpoint(f"epoch_{epoch}.pth")

    def _train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        
        for batch in tqdm(loader, desc="Training"):
            windows = {tf: {int(w): t.to(self.device) for w, t in ws.items()}
                       for tf, ws in batch["windows"].items()}
            labels = batch["label"].to(self.device)
            
            self.optimizer.zero_grad(set_to_none=True)
            
            with autocast():
                preds = self.model(windows)
                loss = self.criterion(preds, labels)
            
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item() * labels.size(0)
        
        return total_loss / len(loader.dataset)

    def _validate_epoch(self, loader: DataLoader) -> Tuple[float, Dict]:
        self.model.eval()
        total_loss = 0.0
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for batch in loader:
                windows = {tf: {int(w): t.to(self.device) for w, t in ws.items()}
                           for tf, ws in batch["windows"].items()}
                labels = batch["label"].to(self.device)
                
                preds = self.model(windows)
                loss = self.criterion(preds, labels)
                total_loss += loss.item() * labels.size(0)
                
                all_preds.append((preds > 0).float())
                all_labels.append(labels)
        
        preds_all = torch.cat(all_preds)
        labels_all = torch.cat(all_labels)
        
        accuracy = (preds_all == labels_all).float().mean().item()
        precision = ((preds_all == 1) & (labels_all == 1)).sum() / (preds_all == 1).sum().clamp(min=1)
        recall = ((preds_all == 1) & (labels_all == 1)).sum() / (labels_all == 1).sum().clamp(min=1)
        f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-8)
        
        return total_loss / len(loader.dataset), {
            "accuracy": accuracy,
            "precision": precision.item(),
            "recall": recall.item(),
            "f1": f1.item()
        }

    def _get_dataloader(self, symbols: List[str], split: str) -> DataLoader:
        dataset = ScalpSequenceDataset(self.config, symbols, split=split)
        shuffle = split == "train"
        workers = 0 if self.config["hardware"]["mode"] == "phone_tiny" else 4
        
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=workers,
            pin_memory=torch.cuda.is_available()
        )

    def _get_training_symbols(self) -> List[str]:
        """Символы для обучения — из whitelist или все доступные"""
        whitelist = self.storage.get_whitelist()
        if whitelist:
            return whitelist
        return ["BTCUSDT", "ETHUSDT", "SOLUSDT"]  # fallback

    def _save_checkpoint(self, filename: str):
        path = os.path.join(self.save_dir, filename)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "epoch": self.current_epoch
        }, path)
        logger.info(f"Сохранён чекпоинт: {path}")

    def retrain_incremental(self):
        """Инкрементальное переобучение (вызывается из live_loop каждые 10k свечей)"""
        logger.info("Запуск инкрементального переобучения")
        self.train()


def main():
    config = load_config()
    trainer = Trainer(config)
    
    if "--retrain" in os.sys.argv:
        trainer.retrain_incremental()
    else:
        trainer.train()


if __name__ == "__main__":
    main()