# src/model/architectures.py
"""
Архитектура нейронной сети — гибрид Conv1D + GRU с мульти-TF анализом.

Основные особенности:
- Отдельные ветки Conv1D + GRU для каждого TF (1m, 3m, 5m, 10m, 15m)
- Concatenation hidden states всех TF → fusion Dense
- Два выхода:
  - probability (sigmoid) — вероятность профитной сделки (0–1)
  - expected_return (linear) — ожидаемая доходность в % (опционально)
- Tiny-версия для телефона: меньше hidden_size, слоёв, Conv filters
- Поддержка quantization (int8) и ONNX export

Вход:
- sequences: Dict[tf: torch.Tensor(batch, seq_len, features)]
- aggregated_features: Dict[tf: Dict[str, float]] — 12 признаков + 4 условия

Выход:
- prob: torch.Tensor(batch, 1) → sigmoid → 0..1
- exp_return: torch.Tensor(batch, 1) → float (может быть None)

Конфиг-зависимые параметры:
- hidden_size: 32 (phone) / 64–128 (server)
- conv_filters: 16 (phone) / 32–64
- num_tf: config["max_tf"]
- add_regression_head: true/false (из конфига)

Логика:
- Conv1D извлекает локальные паттерны в последовательности
- GRU запоминает временную зависимость
- Fusion объединяет знания всех TF
- Head_prob — бинарная классификация (профит/убыток)
- Head_return — регрессия ожидаемой доходности
"""

import torch
import torch.nn as nn
import logging

from src.core.config import load_config

logger = logging.getLogger(__name__)


class MultiTFHybrid(nn.Module):
    """Основная модель: Conv1D + GRU с мульти-TF"""

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config

        # Параметры из конфига
        self.seq_len = config["model"]["seq_len"]
        self.num_features = config["model"]["num_features"]  # 7 (OHLCV + bid/ask) + доп. признаки
        self.hidden_size = config["model"].get("hidden_size", 64)
        self.conv_filters = config["model"].get("conv_filters", 32)
        self.num_tf = config["max_tf"]
        self.add_regression_head = config["model"].get("add_regression_head", True)
        self.is_tiny = config.get("use_tiny_model", False)

        # Если tiny — уменьшаем сложность
        if self.is_tiny:
            self.hidden_size = 32
            self.conv_filters = 16
            logger.info("Using tiny model: hidden_size=32, conv_filters=16")

        # Conv1D — извлекает локальные паттерны
        self.conv = nn.Conv1d(
            in_channels=self.num_features,
            out_channels=self.conv_filters,
            kernel_size=3,
            padding=1
        )
        self.relu = nn.ReLU()

        # GRU — временная зависимость
        self.gru = nn.GRU(
            input_size=self.conv_filters,
            hidden_size=self.hidden_size,
            batch_first=True,
            num_layers=1  # Одна слой — достаточно для скальпа
        )

        # Fusion — объединение всех TF
        self.fc_fusion = nn.Linear(self.hidden_size * self.num_tf, 128)
        self.dropout = nn.Dropout(0.3)

        # Head — вероятность профита
        self.head_prob = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # 0..1
        )

        # Опционально — регрессия ожидаемой доходности
        if self.add_regression_head:
            self.head_return = nn.Linear(128, 1)  # float %

        logger.info("Model initialized: %d TF, seq_len=%d, hidden=%d, tiny=%s", 
                    self.num_tf, self.seq_len, self.hidden_size, self.is_tiny)

    def forward(self, sequences: Dict[str, torch.Tensor], agg_features: Dict[str, Dict]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass

        Args:
            sequences: Dict[tf: Tensor(batch, seq_len, features)]
            agg_features: Dict[tf: Dict[str, float]] — aggregated признаки

        Returns:
            prob: Tensor(batch, 1) — вероятность профита
            exp_return: Tensor(batch, 1) или None
        """
        batch_size = next(iter(sequences.values())).shape[0]  # Берём batch из первого TF

        tf_outputs = []

        for tf in self.config["timeframes"][:self.num_tf]:
            if tf not in sequences:
                continue

            x = sequences[tf]  # (batch, seq_len, features)
            x = x.permute(0, 2, 1)  # (batch, features, seq_len) — для Conv1D

            # Conv1D
            x = self.conv(x)  # (batch, conv_filters, seq_len)
            x = self.relu(x)
            x = x.permute(0, 2, 1)  # (batch, seq_len, conv_filters)

            # GRU
            _, hn = self.gru(x)  # hn: (1, batch, hidden_size)
            tf_outputs.append(hn.squeeze(0))  # (batch, hidden_size)

        # Concat всех TF
        if not tf_outputs:
            raise ValueError("No TF data provided")
        fused = torch.cat(tf_outputs, dim=1)  # (batch, hidden_size * num_tf)

        # Fusion Dense
        fused = self.fc_fusion(fused)
        fused = self.relu(fused)
        fused = self.dropout(fused)

        # Вероятность профита
        prob = self.head_prob(fused)  # (batch, 1)

        # Ожидаемая доходность (опционально)
        exp_return = None
        if self.add_regression_head:
            exp_return = self.head_return(fused)  # (batch, 1)

        return prob, exp_return


# Tiny-версия модели (для телефона)
class TinyHybrid(nn.Module):
    """Упрощённая модель для телефона (меньше параметров)"""

    def __init__(self, config: Dict):
        super().__init__()
        self.num_tf = config["max_tf"]  # Обычно 3
        self.hidden_size = 32
        self.conv_filters = 16

        self.conv = nn.Conv1d(7, self.conv_filters, kernel_size=3, padding=1)
        self.gru = nn.GRU(self.conv_filters, self.hidden_size, batch_first=True)
        self.fc = nn.Linear(self.hidden_size * self.num_tf, 64)
        self.head_prob = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid())

        logger.info("TinyHybrid initialized: %d TF, hidden=%d", self.num_tf, self.hidden_size)

    def forward(self, sequences: Dict, agg_features: Dict):
        tf_outputs = []
        for tf in list(sequences.keys())[:self.num_tf]:
            x = sequences[tf].permute(0, 2, 1)
            x = self.conv(x).permute(0, 2, 1)
            _, hn = self.gru(x)
            tf_outputs.append(hn.squeeze(0))

        fused = torch.cat(tf_outputs, dim=1)
        fused = self.fc(fused)
        prob = self.head_prob(fused)
        return prob, None  # Без regression в tiny


# Пример использования (для тестов)
if __name__ == "__main__":
    config = load_config()
    model = MultiTFHybrid(config)

    # Имитация входа
    sequences = {
        "1m": torch.randn(8, 100, 7),
        "5m": torch.randn(8, 100, 7),
    }
    agg_features = {}  # Пустой для примера

    prob, exp_return = model(sequences, agg_features)
    print("Probability shape:", prob.shape)  # torch.Size([8, 1])
    print("Expected return:", exp_return.shape if exp_return is not None else "None")