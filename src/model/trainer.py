"""
src/model/trainer.py
"""

import argparse
import asyncio
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
from tqdm import tqdm
from datetime import datetime
import polars as pl

from src.core.config import load_config
from src.data.downloader import Downloader
from src.data.storage import Storage
from src.features.feature_engine import FeatureEngine
from src.model.architectures import HybridMultiTFConvGRU, build_model
from src.trading.tp_sl_manager import TP_SL_Manager
from src.utils.logger import setup_logger

setup_logger()
logger = logging.getLogger(__name__)

class TradingDataset(Dataset):
    def __init__(self, sequences, agg_features, labels):
        self.sequences = sequences
        self.agg_features = agg_features
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return {
            "sequences": self.sequences[idx],
            "agg_features": self.agg_features[idx],
            "label": self.labels[idx]
        }

def parse_args():
    parser = argparse.ArgumentParser(description="Обучение модели Crypto Scalper Bot")
    parser.add_argument("--hardware", default="phone_tiny", choices=["phone_tiny", "colab", "server"])
    parser.add_argument("--mode", default="balanced", choices=["conservative", "balanced", "aggressive", "custom"])
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--timeframe", default="1m")
    return parser.parse_args()

async def prepare_dataset(config, symbol: str, timeframe: str):
    storage = Storage()
    downloader = Downloader(config)
    await downloader.download_full_history(symbol, timeframe)
    candles = await storage.get_candles(symbol, timeframe, limit=config["data"]["max_history_candles"])
    if len(candles) < config["seq_len"] * 2:
        raise ValueError(f"Недостаточно свечей: {len(candles)}")

    df = pl.DataFrame(candles)
    feature_engine = FeatureEngine(config)
    tp_sl_manager = TP_SL_Manager()
    sequences = []
    agg_features_list = []
    labels = []

    for i in tqdm(range(len(df) - config["seq_len"] - 1), desc="Подготовка датасета"):
        window_df = df.slice(i, config["seq_len"])
        features_dict = await feature_engine.build_features({timeframe: window_df})
        seq = features_dict["sequences"].get(timeframe)
        if seq is None:
            continue
        agg = features_dict["features"].get(timeframe, {})

        # Реальные labels по ТЗ (используем длину TP/SL)
        tp_sl = tp_sl_manager.calculate_tp_sl(features_dict, "C")
        tp_length = abs(tp_sl.get('tp', 0) - window_df["close"][-1])
        sl_length = abs(window_df["close"][-1] - tp_sl.get('sl', 0))
        label = 1 if tp_length > sl_length else 0

        sequences.append(seq)
        agg_features_list.append(agg)
        labels.append(label)

    return sequences, agg_features_list, labels

def train_model(config, sequences, agg_features, labels):
    dataset = TradingDataset(sequences, agg_features, labels)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=config["model"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["model"]["batch_size"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Обучение на {device}")

    model = build_model(config).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=config["model"]["learning_rate"])
    criterion = nn.BCEWithLogitsLoss()

    epochs = config["model"]["epochs"]
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            optimizer.zero_grad()
            seq = batch["sequences"].to(device)
            sequences_input = [seq.clone() for _ in range(len(config["timeframes"]))]
            cluster_id = torch.zeros(seq.shape[0], dtype=torch.long, device=device)  # ← cluster_id работает
            prob, _ = model(sequences_input, cluster_id)
            loss = criterion(prob.squeeze(), batch["label"].to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                seq = batch["sequences"].to(device)
                sequences_input = [seq.clone() for _ in range(len(config["timeframes"]))]
                cluster_id = torch.zeros(seq.shape[0], dtype=torch.long, device=device)
                prob, _ = model(sequences_input, cluster_id)
                pred = (prob.squeeze() > 0.5).float()
                correct += (pred == batch["label"].to(device)).sum().item()
                total += len(batch["label"])
        accuracy = correct / total if total > 0 else 0
        logger.info(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}, Val Acc: {accuracy:.4f}")

    path = f"models/model_{config.get('hardware_profile', 'unknown')}_{datetime.now().strftime('%Y%m%d_%H%M')}.pth"
    torch.save(model.state_dict(), path)
    logger.info(f"Модель сохранена: {path}")

async def retrain(config, symbol: str = "BTCUSDT", timeframe: str = "1m"):
    logger.info(f"Запуск retrain для {symbol} {timeframe}")
    sequences, agg_features, labels = await prepare_dataset(config, symbol, timeframe)
    if not sequences:
        logger.error("Датасет пустой")
        return
    config["model"]["epochs"] = max(5, config["model"]["epochs"] // 5)
    train_model(config, sequences, agg_features, labels)
    logger.info("Retrain завершён")

async def main():
    args = parse_args()
    config = load_config()
    logger.info(f"Запуск обучения | Symbol: {args.symbol} | TF: {args.timeframe}")
    sequences, agg_features, labels = await prepare_dataset(config, args.symbol, args.timeframe)
    if not sequences:
        logger.error("Датасет пустой")
        return
    train_model(config, sequences, agg_features, labels)

if __name__ == "__main__":
    asyncio.run(main())