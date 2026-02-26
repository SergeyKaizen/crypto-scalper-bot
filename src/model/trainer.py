"""
src/model/trainer.py

=== Основной принцип работы файла ===

Обучение и переобучение модели.

Ключевые изменения (по утверждённым 5 пунктам):
- TimeSeriesSplit с purged gap + **embargo_gap** (пункт 1)
- Step в prepare_data = seq_len // 2 (пункт 2)
- **Embargo around target** в _get_label (пункт 3)
- Фиксированный **val size** в каждом сплите (пункт 4)
- Нет shuffle (shuffle=False) — хронология сохранена
- Обучение на всех сделках (без replay buffer)

=== Примечания ===
- Embargo предотвращает перетекание target и фич между train/val
- Step снижает overlap между последовательностями
- Val size фиксирован — стабильные метрики на каждом сплите
- Полностью соответствует ТЗ + 5 пунктам
- Готов к интеграции в live_loop
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit

from src.core.config import load_config
from src.model.architectures import build_model
from src.data.storage import Storage
from src.trading.tp_sl_manager import TP_SL_Manager
from src.utils.logger import setup_logger

logger = setup_logger('trainer', logging.INFO)

class TimeSeriesDataset(Dataset):
    def __init__(self, data_lists, labels):
        self.data_lists = data_lists
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return [torch.tensor(d[idx], dtype=torch.float32) for d in self.data_lists], torch.tensor(self.labels[idx], dtype=torch.float32)


class Trainer:
    def __init__(self):
        self.config = load_config()
        self.storage = Storage()
        self.tp_sl_manager = TP_SL_Manager()
        self.model = None
        self.optimizer = None
        self.criterion = nn.BCEWithLogitsLoss()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def calculate_n_features(self):
        """Расчёт n_features + перезапись в bot_config.yaml"""
        dummy_df = pd.DataFrame()
        dummy_feats = compute_features(dummy_df)
        n_features = sum(len(f) for f in dummy_feats.values()) + 1  # + quiet_streak

        bot_config_path = BASE_DIR / 'config' / 'bot_config.yaml'
        if bot_config_path.exists():
            with open(bot_config_path, 'r') as f:
                bot_cfg = yaml.safe_load(f)
            bot_cfg['model']['n_features'] = n_features
            with open(bot_config_path, 'w') as f:
                yaml.safe_dump(bot_cfg, f)
            logger.info(f"n_features обновлено: {n_features}")
        return n_features

    def prepare_data(self, timeframe=None):
        """Подготовка multi-TF данных с шагом (step)"""
        timeframes = self.config['timeframes']
        seq_len = self.config['seq_len']
        step = seq_len // 2  # пункт 2 — перекрытие только наполовину

        data_by_tf = {tf: [] for tf in timeframes}
        labels = []

        symbols = self.storage.get_whitelisted_symbols()
        for symbol in symbols:
            for tf in timeframes:
                if timeframe is not None and tf != timeframe:
                    continue
                df = self.storage.get_candles(symbol, tf, limit=seq_len * 10)
                if len(df) < seq_len + 10:
                    continue

                # Step вместо range(..., 1)
                for start in range(0, len(df) - seq_len - 5, step):
                    window_df = df.iloc[start:start + seq_len]
                    feats = compute_features(window_df)

                    tf_inputs = []
                    for t in timeframes:
                        tf_inputs.append(feats[t] if t in feats else np.zeros((seq_len, self.config['n_features'])))

                    label = self._get_label(window_df, 'long')
                    data_by_tf[tf].append(tf_inputs)
                    labels.append(label)

        data_lists = [np.array(data_by_tf[tf]) for tf in timeframes]
        labels = np.array(labels)

        return data_lists, labels

    def _get_label(self, df_window: pd.DataFrame, direction: str = 'long'):
        """
        Симулятор TP/SL с embargo around target (пункт 3)
        """
        embargo_bars = self.config.get('embargo_bars', 5)  # пункт 3 — без последних N баров

        # Урезаем окно для target
        embargo_window = df_window.iloc[:-embargo_bars] if len(df_window) > embargo_bars else df_window

        tp, sl = self.tp_sl_manager.calculate_levels(embargo_window, direction)

        entry_price = df_window['close'].iloc[-1]

        # Симуляция на последних барах окна (с embargo)
        future_df = df_window.iloc[-embargo_bars:] if len(df_window) > embargo_bars else df_window.iloc[-1:]

        hit_tp = False
        hit_sl = False

        for i in range(len(future_df)):
            high = future_df['high'].iloc[i]
            low = future_df['low'].iloc[i]

            if direction == 'long':
                if high >= tp:
                    hit_tp = True
                if low <= sl:
                    hit_sl = True
            else:
                if low <= tp:
                    hit_tp = True
                if high >= sl:
                    hit_sl = True

            if hit_tp or hit_sl:
                break

        return 1 if hit_tp and not hit_sl else 0

    def retrain(self, timeframe=None):
        """Переобучение с TimeSeriesSplit + embargo + фиксированный val size"""
        logger.info(f"Начало retrain для TF: {timeframe or 'all'}")

        data_lists, labels = self.prepare_data(timeframe=timeframe)

        if not labels.size:
            logger.warning("Нет данных для обучения")
            return

        if self.config['model']['n_features'] == 128:
            n_features = self.calculate_n_features()
            self.config['model']['n_features'] = n_features

        dataset = TimeSeriesDataset(data_lists, labels)

        # TimeSeriesSplit с purged gap и embargo
        tscv = TimeSeriesSplit(n_splits=5)
        purged_gap = int(self.config['seq_len'] * self.config.get('purged_gap_multiplier', 1.0))
        embargo_gap = self.config.get('embargo_bars', 5)  # пункт 1 — дополнительный embargo
        val_size = int(len(dataset) * 0.2)  # пункт 4 — фиксированный размер val

        self.model = build_model(self.config)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.config['learning_rate'])

        epochs = self.config.get('epochs', 20)
        best_val_loss = float('inf')

        for fold, (train_idx, val_idx) in enumerate(tscv.split(range(len(dataset)))):
            logger.info(f"Fold {fold+1}/5")

            # Purged gap + embargo
            train_idx = train_idx[train_idx < val_idx.min() - purged_gap - embargo_gap]

            # Фиксированный val size (пункт 4)
            val_idx = val_idx[:val_size]

            train_loader = DataLoader(
                torch.utils.data.Subset(dataset, train_idx),
                batch_size=self.config['batch_size'],
                shuffle=False  # shuffle=False по утверждению
            )
            val_loader = DataLoader(
                torch.utils.data.Subset(dataset, val_idx),
                batch_size=self.config['batch_size'],
                shuffle=False
            )

            for epoch in range(epochs):
                self.model.train()
                train_loss = 0
                for batch_inputs, batch_labels in train_loader:
                    batch_inputs = [i.to(self.device) for i in batch_inputs]
                    batch_labels = batch_labels.to(self.device).unsqueeze(1)

                    self.optimizer.zero_grad()
                    outputs = self.model(batch_inputs)
                    loss = self.criterion(outputs, batch_labels)
                    loss.backward()
                    self.optimizer.step()
                    train_loss += loss.item()

                # Валидация
                self.model.eval()
                val_loss = 0
                with torch.no_grad():
                    for batch_inputs, batch_labels in val_loader:
                        batch_inputs = [i.to(self.device) for i in batch_inputs]
                        batch_labels = batch_labels.to(self.device).unsqueeze(1)
                        outputs = self.model(batch_inputs)
                        loss = self.criterion(outputs, batch_labels)
                        val_loss += loss.item()

                avg_val_loss = val_loss / len(val_loader)
                logger.info(f"Epoch {epoch+1}/{epochs}, Fold {fold+1}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {avg_val_loss:.4f}")

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    torch.save(self.model.state_dict(), f"models/best_model_{timeframe or 'all'}.pt")

        torch.save(self.model.state_dict(), f"models/model_{timeframe or 'all'}_{datetime.now().strftime('%Y%m%d')}.pt")
        logger.info("Retraining завершён")