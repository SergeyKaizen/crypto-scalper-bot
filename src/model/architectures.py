"""
src/model/architectures.py

=== Основной принцип работы файла ===

Гибридная архитектура Conv1D + Bidirectional GRU для предсказания вероятности профитной сделки.

Ключевые особенности (по ТЗ + все утверждённые изменения):
- Multi-TF input: 5 отдельных тензоров (1m,3m,5m,10m,15m)
- Отдельная Conv1D ветка на каждый TF → concat hidden states → fusion Dense
- Bidirectional GRU для захвата контекста с обеих сторон последовательности
- Input shape: (batch, seq_len, n_features) на каждый TF
- n_features динамический (рассчитывается в trainer/inference)
- Масштабирование: model_size ('small'/'medium'/'large') под Colab/server
- Dropout, BatchNorm для снижения оверфита на шумных данных крипты
- Выход: sigmoid вероятность профита (>0.5 → сигнал на вход)

=== Главные классы ===

- ConvBranch — Conv1D ветка для одного TF
- HybridMultiTFConvGRU — основная модель

=== Примечания ===
- Bidirectional GRU: hidden_size * 2 на выходе
- Fusion: concat по всем TF → Linear → GRU
- Поддержка extra фич (quiet_streak) через n_features
- На сервере ('large') — больше слоёв/нейронов
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.core.config import load_config

class ConvBranch(nn.Module):
    """Conv1D ветка для одного таймфрейма"""
    def __init__(self, in_channels, hidden_size):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, hidden_size // 2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_size // 2, hidden_size, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_size // 2)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # x: (batch, seq_len, features) → (batch, features, seq_len)
        x = x.transpose(1, 2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x).squeeze(-1)  # (batch, hidden_size)
        return x


class HybridMultiTFConvGRU(nn.Module):
    def __init__(self, n_features: int, num_tf: int = 5, seq_len: int = 100,
                 hidden_size: int = 128, dropout: float = 0.3, model_size: str = 'medium'):
        super().__init__()
        self.num_tf = num_tf
        self.seq_len = seq_len
        self.n_features = n_features

        # Масштабирование под железо
        if model_size == 'large':
            hidden_size *= 2
            gru_layers = 3
        elif model_size == 'small':
            hidden_size //= 2
            gru_layers = 1
        else:
            gru_layers = 2

        # Conv ветки для каждого TF
        self.conv_branches = nn.ModuleList([
            ConvBranch(n_features, hidden_size) for _ in range(num_tf)
        ])

        # Fusion после concat
        fusion_in = hidden_size * num_tf
        self.fusion_dense = nn.Linear(fusion_in, hidden_size * 2)
        self.bn_fusion = nn.BatchNorm1d(hidden_size * 2)

        # Bidirectional GRU
        self.gru = nn.GRU(
            input_size=hidden_size * 2,
            hidden_size=hidden_size,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if gru_layers > 1 else 0.0
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, 1)  # *2 из-за bidirectional

    def forward(self, inputs: list):
        """
        inputs: list of tensors, len=5 (по TF)
        каждый: (batch, seq_len, n_features)
        """
        if len(inputs) != self.num_tf:
            raise ValueError(f"Ожидается {self.num_tf} входов по TF, получено {len(inputs)}")

        # Обработка каждого TF отдельно
        branch_outputs = []
        for i, branch in enumerate(self.conv_branches):
            branch_outputs.append(branch(inputs[i]))

        # Concat по TF
        fused = torch.cat(branch_outputs, dim=1)  # (batch, hidden_size * num_tf)

        # Fusion Dense + BN
        fused = F.relu(self.bn_fusion(self.fusion_dense(fused)))

        # Подготовка для GRU: добавляем dim seq_len=1
        fused = fused.unsqueeze(1)  # (batch, 1, hidden_size*2)

        # Bidirectional GRU
        gru_out, _ = self.gru(fused)
        gru_out = gru_out[:, -1, :]  # last hidden (batch, hidden_size*2)

        out = self.dropout(gru_out)
        out = torch.sigmoid(self.fc(out))  # вероятность профита [0,1]

        return out


def build_model(config):
    """
    Фабрика модели под текущий config
    """
    n_features = config.get('n_features', 128)  # должно быть рассчитано заранее
    seq_len = config.get('seq_len', 100)
    num_tf = len(config['timeframes'])  # 5
    model_size = config.get('model_size', 'medium')
    dropout = config.get('dropout', 0.3)
    hidden_size = config.get('hidden_size', 128)

    model = HybridMultiTFConvGRU(
        n_features=n_features,
        num_tf=num_tf,
        seq_len=seq_len,
        hidden_size=hidden_size,
        dropout=dropout,
        model_size=model_size
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()  # по умолчанию inference mode

    logger.info(f"Модель построена: {model_size}, TF: {num_tf}, features: {n_features}, seq_len: {seq_len}")
    return model