"""
src/model/architectures.py

=== Основной принцип работы файла ===

Этот файл содержит архитектуру гибридной модели Conv1D + GRU по ТЗ.
Модель обрабатывает мульти-TF и мульти-окна последовательности:
- Отдельные ветки Conv1D + GRU для каждого таймфрейма (1m,3m,5m,10m,15m).
- Для каждого TF — отдельные входы по окнам (24,50,74,100 свечей).
- Fusion: concatenation hidden states всех веток → Dense слои → бинарный выход (вероятность профитной сделки).

Ключевые особенности:
- Input — dict[tf: dict[window: tensor(seq_len, num_features)]]
- Conv1D для локальных паттернов в последовательности.
- GRU для долгосрочной памяти и переходов между половинами.
- Dropout и BatchNorm для регуляризации.
- Масштабирование по hardware (hidden_size, num_layers из config).
- Выход — sigmoid (вероятность 0..1).

Модель полностью соответствует ТЗ: гибрид Conv1D+GRU, sequences (не статичные признаки), мульти-TF fusion.

=== Главные классы и за что отвечают ===

- WindowBranch(nn.Module): ветка для одного окна (Conv1D → GRU)
- MultiWindowHybrid(nn.Module): основная модель — dict веток по TF и окнам, fusion

- forward(inputs: dict) → torch.Tensor (вероятность)
  Обрабатывает dict входов, возвращает вероятность профитной сделки.

=== Примечания ===
- Input_dim = количество признаков из feature_engine (все признаки ТЗ).
- Hidden_size и num_layers масштабируются по hardware (phone_tiny — меньше).
- Нет лишних слоёв — строго Conv1D + GRU + fusion.
- Готов к обучению в trainer.py и inference.
- Логи через setup_logger (если нужно).
"""

import torch
import torch.nn as nn

from src.core.config import load_config
from src.core.enums import Timeframe
from src.utils.logger import setup_logger

logger = setup_logger('architectures', logging.INFO)

class WindowBranch(nn.Module):
    """
    Ветка для одного окна (одного периода свечей).
    Conv1D → GRU → hidden state.
    """
    def __init__(self, input_dim: int, hidden_size: int, num_layers: int = 1, dropout: float = 0.2):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=input_dim,
            out_channels=hidden_size,
            kernel_size=3,
            padding=1
        )
        self.bn = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, features) → (batch, features, seq_len) для Conv1D
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = x.permute(0, 2, 1)  # обратно (batch, seq_len, hidden)

        _, hn = self.gru(x)
        return hn[-1]  # last hidden state (batch, hidden_size)

class MultiWindowHybrid(nn.Module):
    """
    Гибридная модель Conv1D + GRU с мульти-TF и мульти-окнами.
    Входы: dict[tf_str: dict[window_int: tensor(batch, seq_len, features)]]
    Выход: вероятность профитной сделки (sigmoid).
    """
    def __init__(self, input_dim: int):
        super().__init__()
        config = load_config()
        hidden_size = config['model']['hidden_size']
        num_layers = config['model']['num_gru_layers']
        dropout = config['model']['dropout']

        self.branches = nn.ModuleDict()
        for tf_enum in Timeframe:
            tf_key = tf_enum.value  # '1m', '3m' и т.д.
            self.branches[tf_key] = nn.ModuleDict()

            for window in config['windows_sizes']:  # [24,50,74,100]
                branch = WindowBranch(
                    input_dim=input_dim,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout
                )
                self.branches[tf_key][str(window)] = branch

        # Fusion
        num_branches = len(Timeframe) * len(config['windows_sizes'])
        fusion_in = hidden_size * num_branches
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()  # вероятность 0..1
        )

    def forward(self, inputs: Dict[str, Dict[str, torch.Tensor]]) -> torch.Tensor:
        """
        inputs: {
            '1m': { '24': tensor(batch, seq_len, features), '50': ..., ... },
            '3m': { ... },
            ...
        }
        """
        hidden_states = []

        for tf_key in self.branches:
            if tf_key not in inputs:
                continue  # TF может отсутствовать в батче
            for window_str in self.branches[tf_key]:
                if window_str not in inputs[tf_key]:
                    continue
                x = inputs[tf_key][window_str]  # (batch, seq_len, features)
                h = self.branches[tf_key][window_str](x)  # (batch, hidden_size)
                hidden_states.append(h)

        if not hidden_states:
            raise ValueError("Нет входных данных для модели")

        # Concat всех hidden states
        concat = torch.cat(hidden_states, dim=1)  # (batch, hidden_size * num_branches)
        out = self.fusion(concat)  # (batch, 1)
        return out.squeeze(-1)  # (batch,)