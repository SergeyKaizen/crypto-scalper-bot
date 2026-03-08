"""
Скрипт первоначального обучения и переобучения модели.
"""

import argparse
import asyncio
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from datetime import datetime
import polars as pl  # ← добавлено

from src.core.config import load_config
from src.core.types import Candle
from src.data.downloader import Downloader
from src.data.storage import Storage
from src.features.feature_engine import FeatureEngine
from src.model.architectures import HybridMultiTFConvGRU, build_model  # ← исправлено
from src.model.trainer import TradingDataset  # ← теперь совместим
from src.utils.logger import setup_logger

setup_logger()
logger = logging.getLogger(__name__)

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
    """Подготовка датасета: sequences, agg_features, labels"""
    storage = Storage()
    downloader = Downloader(config)

    await downloader.download_full_history(symbol, timeframe)

    candles = await storage.get_candles(symbol, timeframe, limit=config["data"]["max_history_candles"])
    if len(candles) < config["seq_len"] * 2:
        raise ValueError(f"Недостаточно свечей для обучения: {len(candles)} < {config['seq_len'] * 2}")

    df = pl.DataFrame(candles)

    feature_engine = FeatureEngine(config)

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

        entry_price = window_df["close"][-1]
        expected_rr = 2.0
        label = 1 if expected_rr > 1.5 else 0

        sequences.append(seq)
        agg_features_list.append(agg)
        labels.append(label)

    return sequences, agg_features_list, labels

def train_model(config, sequences, agg_features, labels):
    """Обучение модели"""
    dataset = TradingDataset(sequences, agg_features, labels)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=config["model"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["model"]["batch_size"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Обучение будет производиться на устройстве: {device}")

    model = build_model(config).to(device)  # ← исправлено

    optimizer = optim.AdamW(model.parameters(), lr=config["model"]["learning_rate"])
    criterion = nn.BCEWithLogitsLoss()

    epochs = config["model"]["epochs"]
    if "retrain" in globals() and globals()["args"].retrain:  # поддержка --retrain
        epochs = max(10, epochs // 5)

    logger.info("Обучение: %d эпох, batch=%d, device=%s", epochs, config["model"]["batch_size"], device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            optimizer.zero_grad()

            # Адаптация под новую forward
            seq = batch["sequences"].to(device)
            sequences_input = [seq.clone() for _ in range(len(config["timeframes"]))]
            cluster_id = torch.zeros(seq.shape[0], dtype=torch.long, device=device)

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
        logger.info("Epoch %d/%d, Loss: %.4f, Val Accuracy: %.4f", epoch+1, epochs, total_loss / len(train_loader), accuracy)

    path = f"models/model_{config.get('hardware_profile', 'unknown')}_{datetime.now().strftime('%Y%m%d_%H%M')}.pth"
    torch.save(model.state_dict(), path)
    logger.info("Модель сохранена: %s", path)

async def main():
    global args
    args = parse_args()
    config = load_config()
    logger.info(f"Запуск обучения | Symbol: {args.symbol} | TF: {args.timeframe} | Retrain: {args.retrain}")

    sequences, agg_features, labels = await prepare_dataset(config, args.symbol, args.timeframe)

    if not sequences:
        logger.error("Датасет пустой — обучение невозможно")
        return

    train_model(config, sequences, agg_features, labels)

if __name__ == "__main__":
    asyncio.run(main())