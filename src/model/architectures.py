"""
src/model/architectures.py

=== Основной принцип работы файла ===

Гибридная архитектура Conv1D + Bidirectional GRU для предсказания.

Изменения по последнему ТЗ:
- Multi-TF input: 5 отдельных тензоров (1m,3m,5m,10m,15m)
- Input shape: (batch, seq_len, n_features) на каждый TF
- Предсказание: не только да/нет (binary prob), но и ожидаемый профит (tp_distance / sl_distance если TP раньше SL, иначе -1)
- Multi-task: BCE для binary + MSE для regression
- Bidirectional GRU включён по умолчанию
- Масштабирование под железо (model_size)

=== Главные классы ===

- ConvBranch — Conv1D ветка для одного TF
- HybridMultiTFConvGRU — основная модель (multi-task output)

=== Примечания ===
- Input: list of 5 tensors (по TF)
- Output: [prob, expected_profit_ratio] (prob > min_prob → сигнал, ratio >1 → TP раньше SL)
- n_features динамический (из config)
- Готов к trainer/inference/live_loop
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

        # Multi-task output: prob (sigmoid) + expected_profit_ratio (linear)
        self.fc_binary = nn.Linear(hidden_size * 2, 1)   # для да/нет
        self.fc_regression = nn.Linear(hidden_size * 2, 1)  # для ожидаемого профита

    def forward(self, inputs: list):
        """
        inputs: list of 5 tensors (по TF), каждый (batch, seq_len, n_features)
        """
        if len(inputs) != self.num_tf:
            raise ValueError(f"Ожидается {self.num_tf} входов по TF, получено {len(inputs)}")

        branch_outputs = []
        for i, branch in enumerate(self.conv_branches):
            branch_outputs.append(branch(inputs[i]))

        fused = torch.cat(branch_outputs, dim=1)  # (batch, hidden_size * num_tf)

        fused = F.relu(self.bn_fusion(self.fusion_dense(fused)))

        fused = fused.unsqueeze(1)  # (batch, 1, hidden_size*2)

        gru_out, _ = self.gru(fused)
        gru_out = gru_out[:, -1, :]  # last hidden (batch, hidden_size*2)

        out = self.dropout(gru_out)

        prob = torch.sigmoid(self.fc_binary(out))             # да/нет (0-1)
        expected_profit = self.fc_regression(out)             # ожидаемый профит (linear)

        return prob, expected_profit


def build_model(config):
    """
    Фабрика модели под текущий config
    """
    n_features = config.get('n_features', 128)
    seq_len = config.get('seq_len', 100)
    num_tf = len(config['timeframes'])
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
    model.eval()  # inference mode

    logger.info(f"Модель построена: {model_size}, TF: {num_tf}, features: {n_features}, seq_len: {seq_len}")
    return model