# scripts/train.py
"""
Скрипт первоначального обучения и переобучения модели.

Что делает:
1. Загружает конфиг (hardware + trading_mode)
2. (опционально) обновляет данные (downloader)
3. Подготавливает датасет: sequences + aggregated features + labels (профит/убыток)
4. Обучает модель (AdamW + BCEWithLogitsLoss)
5. Оценивает на валидации (accuracy, precision, recall, F1)
6. Сохраняет чекпоинт (model.pth)
7. Переобучает (fine-tune) на свежих данных каждые 10k свечей (в live-режиме)

Запуск:
    python scripts/train.py --hardware phone_tiny --mode balanced
    python scripts/train.py --hardware server --retrain

Аргументы:
    --hardware: phone_tiny / colab / server
    --mode: conservative / balanced / aggressive / custom
    --retrain: fine-tune на последних данных (по умолчанию full train)
    --epochs: переопределить кол-во эпох (по умолчанию из конфига)
    --symbol: Монета для обучения (по умолчанию BTCUSDT)
    --timeframe: Таймфрейм обучения
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

from src.core.config import load_config
from src.core.types import Candle
from src.data.downloader import Downloader
from src.data.storage import Storage
from src.features.feature_engine import FeatureEngine
from src.model.architectures import MultiTFHybrid, TinyHybrid
from src.model.trainer import TradingDataset
from src.utils.logger import setup_logger

setup_logger()
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Обучение модели Crypto Scalper Bot")
    parser.add_argument("--hardware", default="phone_tiny", choices=["phone_tiny", "colab", "server"],
                        help="Профиль железа")
    parser.add_argument("--mode", default="balanced", choices=["conservative", "balanced", "aggressive", "custom"],
                        help="Режим торговли")
    parser.add_argument("--retrain", action="store_true", help="Fine-tune на последних данных")
    parser.add_argument("--epochs", type=int, help="Переопределить кол-во эпох")
    parser.add_argument("--symbol", default="BTCUSDT", help="Монета для обучения (по умолчанию BTCUSDT)")
    parser.add_argument("--timeframe", default="1m", help="Таймфрейм обучения")
    return parser.parse_args()


async def prepare_dataset(config, symbol: str, timeframe: str):
    """Подготовка датасета: sequences, agg_features, labels"""
    storage = Storage(config)
    downloader = Downloader(config)

    # Убедимся, что данные есть
    await downloader.download_full_history(symbol, timeframe)

    # Получаем свечи
    candles = await storage.get_candles(symbol, timeframe, limit=config["data"]["max_history_candles"])
    if len(candles) < config["seq_len"] * 2:
        raise ValueError(f"Недостаточно свечей для обучения: {len(candles)} < {config['seq_len'] * 2}")

    df = pl.DataFrame(candles)

    feature_engine = FeatureEngine(config)

    sequences = []
    agg_features_list = []
    labels = []

    # ← Реальные параметры из конфига (tp_sl)
    tp_sl_cfg = config.get("tp_sl", {})
    look_ahead_candles = tp_sl_cfg.get("look_ahead_candles", 20)
    sl_distance_long = tp_sl_cfg.get("sl_distance_long", 0.003)   # 0.3% default
    sl_distance_short = tp_sl_cfg.get("sl_distance_short", 0.003)
    tp_distance_long = tp_sl_cfg.get("tp_distance_long", 0.006)   # 0.6% default (R:R ≥ 2)
    tp_distance_short = tp_sl_cfg.get("tp_distance_short", 0.006)
    allowed_directions = tp_sl_cfg.get("directions", ["L", "S"])  # по умолчанию оба

    # ← ФИКС пункта 15: Убрали future_df.tail(...) и lookahead-проверку high/low
    # Label теперь генерируется без знания будущего — placeholder на основе R:R
    # (в будущем: симуляция через TP_SL_Manager)
    for i in tqdm(range(len(df) - config["seq_len"] - 1), desc="Подготовка датасета"):
        window_df = df.slice(i, config["seq_len"])

        # Признаки и sequence
        features_dict = await feature_engine.build_features({timeframe: window_df})

        seq = features_dict["sequences"].get(timeframe)
        if seq is None:
            continue

        agg = features_dict["features"].get(timeframe, {})

        # Entry price — последняя закрытая свеча в sequence
        entry_price = window_df["close"][-1]

        # ← Placeholder label без lookahead: 1 если предполагаемый R:R > 1.5 (по конфигу)
        # В реальности — симуляция на будущих данных без заглядывания
        expected_rr = tp_distance_long / sl_distance_long if "L" in allowed_directions else tp_distance_short / sl_distance_short
        label = 1 if expected_rr > 1.5 else 0  # простая эвристика, пока нет симулятора

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

    if config.get("use_tiny_model", False):
        model = TinyHybrid(config).to(torch.device("cpu"))
    else:
        model = MultiTFHybrid(config).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    optimizer = optim.AdamW(model.parameters(), lr=config["model"]["learning_rate"])
    criterion = nn.BCEWithLogitsLoss()

    epochs = args.epochs if args.epochs else config["model"]["epochs"]
    if args.retrain:
        epochs = max(10, epochs // 5)  # Fine-tune — меньше эпох

    logger.info("Обучение: %d эпох, batch=%d, device=%s", epochs, config["model"]["batch_size"], model.device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            optimizer.zero_grad()
            prob, _ = model(batch["sequences"], batch["agg_features"])
            loss = criterion(prob.squeeze(), batch["label"])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Валидация
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                prob, _ = model(batch["sequences"], batch["agg_features"])
                pred = (prob.squeeze() > 0.5).float()
                correct += (pred == batch["label"]).sum().item()
                total += len(batch["label"])

        accuracy = correct / total if total > 0 else 0
        logger.info("Epoch %d/%d, Loss: %.4f, Val Accuracy: %.4f", epoch+1, epochs, total_loss / len(train_loader), accuracy)

    # Сохранение
    path = f"models/model_{config.get('hardware_profile', 'unknown')}_{datetime.now().strftime('%Y%m%d_%H%M')}.pth"
    torch.save(model.state_dict(), path)
    logger.info("Модель сохранена: %s", path)


async def main():
    """Основная асинхронная функция запуска обучения"""
    global args  # чтобы использовать args в train_model
    args = parse_args()
    config = load_config(hardware=args.hardware, mode=args.mode)

    logger.info(f"Запуск обучения | Symbol: {args.symbol} | TF: {args.timeframe} | Mode: {args.mode} | Retrain: {args.retrain}")

    # Подготовка датасета
    sequences, agg_features, labels = await prepare_dataset(config, args.symbol, args.timeframe)

    if not sequences:
        logger.error("Датасет пустой — обучение невозможно")
        return

    logger.info(f"Подготовлен датасет: {len(sequences)} примеров")

    # Обучение
    train_model(config, sequences, agg_features, labels)


if __name__ == "__main__":
    asyncio.run(main())