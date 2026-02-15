# src/model/architectures.py
"""
Гибридная архитектура Conv1D + GRU для скальпинга на крипте (Binance futures).

Ключевые принципы из ТЗ:
- Данные подаются как последовательности свечей (time-series sequences)
- 4 окна: 24, 50, 74, 100 свечей
- Модель видит полный ряд для анализа переходов между половинами, TF и окнами
- Обучение сразу на всех TF: 1m, 3m, 5m, 10m, 15m
- Вход: 16 признаков (12 базовых + 4 бинарных условия: candle, volume, cv, q)
- Гибрид Conv1D + GRU
- Адаптация под железо: tiny (телефон) — маленький hidden, меньше окон
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

class WindowBranch(nn.Module):
    """
    Одна ветка для конкретного окна (24/50/74/100).
    Conv1D → GRU → delta половин + last hidden → фичи окна
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
        
        self.fc_delta = nn.Linear(hidden_size * 2, hidden_size)  # last + delta половин

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, in_features] → [batch, in_features, seq_len]
        x = x.permute(0, 2, 1)
        
        x = self.conv(x)
        x = F.relu(x)
        x = x.permute(0, 2, 1)  # [batch, seq, hidden]
        
        _, hn = self.gru(x)
        last_hidden = hn[-1]  # [batch, hidden]
        
        # Delta половин (среднее по левой и правой части)
        left_mean  = x[:, :self.half, :].mean(dim=1)
        right_mean = x[:, self.half:, :].mean(dim=1)
        delta = right_mean - left_mean
        
        combined = torch.cat([last_hidden, delta], dim=1)
        out = self.fc_delta(combined)
        out = F.relu(out)
        
        return out


class MultiWindowHybrid(nn.Module):
    """
    Основная модель: 4 параллельные ветки → fusion → вероятность профита.
    Поддерживает multi-TF через словарь входов.
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
        
        # Уменьшаем hidden на телефоне
        branch_hidden = hidden_size // (2 if tiny_mode else 1)
        
        self.branches = nn.ModuleDict({
            str(w): WindowBranch(
                in_features=in_features,
                hidden_size=branch_hidden,
                seq_len=w
            ) for w in windows
        })
        
        fusion_in = branch_hidden * len(windows)
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, inputs: Dict[str, Dict[int, torch.Tensor]]) -> torch.Tensor:
        """
        inputs = {
            "1m":  {24: tensor[bs,24,16], 50: ..., 74: ..., 100: ...},
            "3m":  {...},
            ...
        }
        """
        all_outputs = []
        
        for tf, windows_dict in inputs.items():
            tf_outputs = []
            for w_str, branch in self.branches.items():
                w = int(w_str)
                if w in windows_dict:
                    x = windows_dict[w]  # [batch, seq_w, 16]
                    out = branch(x)
                    tf_outputs.append(out)
            
            if tf_outputs:
                tf_out = torch.mean(torch.stack(tf_outputs), dim=0)
                all_outputs.append(tf_out)
        
        if not all_outputs:
            raise ValueError("Нет данных для модели")
        
        fused = torch.cat(all_outputs, dim=1)
        logits = self.fusion(fused)
        prob = torch.sigmoid(logits).squeeze(-1)  # [batch]
        
        return prob


def create_model(config: dict) -> nn.Module:
    """
    Фабрика моделей по hardware-режиму
    """
    hardware = config["hardware_mode"]
    
    if hardware == "phone_tiny":
        return MultiWindowHybrid(
            in_features=16,          # 12 + 4 условия
            hidden_size=32,
            windows=[50, 100],       # на телефоне только 2 окна
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
    else:  # colab
        return MultiWindowHybrid(
            in_features=16,
            hidden_size=64,
            windows=[24, 50, 74, 100],
            dropout=0.1,
            tiny_mode=False
        )


if __name__ == "__main__":
    config = {"hardware_mode": "phone_tiny"}
    model = create_model(config)
    print(model)
    
    # Тестовый вход (заглушка)
    dummy = {
        "1m": {
            50: torch.randn(2, 50, 16),
            100: torch.randn(2, 100, 16)
        }
    }
    with torch.no_grad():
        out = model(dummy)
    print("Output shape:", out.shape)