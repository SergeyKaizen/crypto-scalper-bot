# src/model/architectures.py
"""
Гибридная архитектура Conv1D + GRU для скальпинга на крипте (Binance Futures).

Основные принципы из ТЗ:
- Данные подаются как последовательности свечей (time-series sequences), а не статичные признаки
- 4 окна: 24, 50, 74, 100 свечей (без паддинга нулями)
- Вход: 16 признаков (12 базовых + 4 бинарных условия: candle, volume, cv, q)
- Модель видит полный ряд для анализа переходов между половинами, таймфреймами и окнами
- Обучение сразу на всех TF: 1m, 3m, 5m, 10m, 15m
- Гибрид: Conv1D (локальные паттерны) + GRU (временные зависимости)
- Адаптация под железо: phone_tiny (маленький hidden, 2 окна), colab/server (полная версия)

Архитектура:
- 4 независимые ветки (по одной на каждое окно)
- В каждой ветке: Conv1D → ReLU → GRU → delta половин + last hidden → Linear
- Fusion всех 4 выходов → Linear → sigmoid (вероятность профитной сделки)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

class WindowBranch(nn.Module):
    """
    Одна ветка для конкретного окна (24/50/74/100).
    Обрабатывает последовательность → Conv1D → GRU → delta половин + last hidden
    """
    def __init__(self, in_features: int, hidden_size: int, seq_len: int):
        super().__init__()
        self.seq_len = seq_len
        self.half = seq_len // 2

        self.conv = nn.Conv1d(
            in_channels=in_features,
            out_channels=hidden_size,
            kernel_size=3,
            padding=1,
            stride=1
        )

        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )

        # Delta половин + last hidden → Linear
        self.fc_delta = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, in_features] → [batch, in_features, seq_len]
        x = x.permute(0, 2, 1)

        x = self.conv(x)
        x = F.relu(x)
        x = x.permute(0, 2, 1)  # [batch, seq, hidden]

        _, hn = self.gru(x)
        last_hidden = hn[-1]  # [batch, hidden]

        # Среднее по левой и правой половине
        left_mean = x[:, :self.half, :].mean(dim=1)
        right_mean = x[:, self.half:, :].mean(dim=1)
        delta = right_mean - left_mean

        combined = torch.cat([last_hidden, delta], dim=1)
        out = self.fc_delta(combined)
        out = F.relu(out)

        return out


class MultiWindowHybrid(nn.Module):
    """
    Основная модель: 4 параллельные ветки → fusion → вероятность профита.
    Поддерживает multi-TF через словарь входов {tf: {window_size: tensor}}
    """
    def __init__(self,
                 in_features: int = 16,               # 12 базовых + 4 условия
                 hidden_size: int = 64,
                 windows: list = [24, 50, 74, 100],
                 dropout: float = 0.1,
                 tiny_mode: bool = False):
        super().__init__()

        self.windows = windows
        self.tiny_mode = tiny_mode

        # Уменьшаем размер на телефоне
        branch_hidden = hidden_size // (2 if tiny_mode else 1)

        self.branches = nn.ModuleDict({
            str(w): WindowBranch(
                in_features=in_features,
                hidden_size=branch_hidden,
                seq_len=w
            ) for w in windows
        })

        # Fusion всех выходов
        fusion_in = branch_hidden * len(windows)
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)  # бинарная классификация
        )

    def forward(self, inputs: Dict[str, Dict[int, torch.Tensor]]) -> torch.Tensor:
        """
        inputs = {
            "1m":  {24: tensor[bs,24,16], 50: ..., 74: ..., 100: ...},
            "3m":  {...},
            ...
        }
        """
        all_tf_outputs = []

        for tf, windows_dict in inputs.items():
            tf_outputs = []
            for w_str, branch in self.branches.items():
                w = int(w_str)
                if w in windows_dict:
                    x = windows_dict[w]  # [batch, seq_w, 16]
                    out = branch(x)
                    tf_outputs.append(out)

            if tf_outputs:
                # Среднее по окнам внутри TF
                tf_out = torch.mean(torch.stack(tf_outputs), dim=0)
                all_tf_outputs.append(tf_out)

        if not all_tf_outputs:
            raise ValueError("Нет валидных входных данных для модели")

        fused = torch.cat(all_tf_outputs, dim=1)
        logits = self.fusion(fused)
        prob = torch.sigmoid(logits).squeeze(-1)  # [batch]

        return prob


def create_model(config: dict) -> nn.Module:
    """
    Фабрика моделей в зависимости от hardware-режима.
    """
    hardware = config["hardware_mode"]

    if hardware == "phone_tiny":
        return MultiWindowHybrid(
            in_features=16,
            hidden_size=32,
            windows=[50, 100],          # на телефоне только 2 окна
            dropout=0.15,
            tiny_mode=True
        )
    elif hardware == "server":
        return MultiWindowHybrid(
            in_features=16,
            hidden_size=128,
            windows=[24, 50, 74, 100],
            dropout=0.1,
            tiny_mode=False
        )
    else:  # colab — средний вариант
        return MultiWindowHybrid(
            in_features=16,
            hidden_size=64,
            windows=[24, 50, 74, 100],
            dropout=0.1,
            tiny_mode=False
        )


if __name__ == "__main__":
    import torch
    config = {"hardware_mode": "phone_tiny"}
    model = create_model(config)
    print(model)

    # Тестовый вход (заглушка для проверки формы)
    dummy = {
        "1m": {
            50: torch.randn(2, 50, 16),
            100: torch.randn(2, 100, 16)
        }
    }

    with torch.no_grad():
        out = model(dummy)
    print("Output shape:", out.shape)  # должен быть torch.Size([2])