# src/model/architectures.py
"""
Архитектура основной модели — гибрид Conv1D + GRU с multi-scale и multi-TF подходом.
"""

import torch
import torch.nn as nn


class ScalperHybridModel(nn.Module):
    """
    Основная модель скальпера 2026 года.
    
    Входы:
    • глобальный контекст (сравнение двух половин 100 свечей) → простая MLP-ветка
    • 4 окна разного масштаба (24,50,74,100) → отдельные Conv1D+GRU ветки
    • 5 таймфреймов (1m,3m,5m,10m,15m) → отдельные Conv1D+GRU ветки
    
    Выход: вероятности [Long, Short, Flat]
    """

    def __init__(self, config):
        super().__init__()
        self.cfg = config
        
        # Размерность признаков на свечу (open,high,low,close,volume,bid,ask,delta,...)
        self.in_channels = config["model"]["in_channels"]           # обычно ~12–16
        
        # 1. Ветка для глобального сравнения половин (16 бит + процентные изменения)
        self.half_branch = nn.Sequential(
            nn.Linear(32, 64),      # 16 бит + 16 процентов изменений
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32)
        )
        
        # 2. Conv1D+GRU ветки для каждого окна
        self.window_branches = nn.ModuleDict()
        for size in [24, 50, 74, 100]:
            self.window_branches[str(size)] = self._make_scale_branch(seq_len=size)
        
        # 3. Conv1D+GRU ветки для каждого таймфрейма
        self.tf_branches = nn.ModuleDict()
        for tf in ["1m", "3m", "5m", "10m", "15m"]:
            self.tf_branches[tf] = self._make_scale_branch(seq_len=100)
        
        # Финальный fusion слой
        # 32 от half + 32×4 от окон + 32 от tf (усреднённые или concat — зависит от реализации)
        fusion_in = 32 + 32 * 4 + 32
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)                # Long / Short / Flat
        )

    def _make_scale_branch(self, seq_len: int):
        """Шаблон ветки для одного масштаба (окна или TF)."""
        return nn.Sequential(
            # Conv1D по времени
            nn.Conv1d(
                in_channels=self.in_channels,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            
            # GRU по последовательности
            nn.GRU(
                input_size=32,
                hidden_size=64,
                num_layers=1,
                batch_first=True,
                bidirectional=False
            ),
            
            # Берём последнее скрытое состояние
            nn.Linear(64, 32)
        )

    def forward(self, model_input):
        # model_input — экземпляр ModelInput
        
        # 1. Глобальный контекст половин
        half_vec = torch.tensor(model_input.half.binary_vector + 
                               list(model_input.half.percent_changes.values()), 
                               dtype=torch.float32, device=self.device)
        half_feat = self.half_branch(half_vec)                      # [batch, 32]

        # 2. Окна
        window_feats = []
        for size_str, df in model_input.windows.items():
            x = self._df_to_tensor(df)                              # [batch, channels, seq_len]
            feat = self.window_branches[size_str](x)                # [batch, 32]
            window_feats.append(feat)
        window_feats = torch.cat(window_feats, dim=1)               # [batch, 32*4]

        # 3. Multi-TF (усредняем или конкатенируем — здесь усредняем для простоты)
        tf_feats_list = []
        for tf, df in model_input.multi_tf.items():
            x = self._df_to_tensor(df)
            feat = self.tf_branches[tf](x)
            tf_feats_list.append(feat)
        tf_feat = torch.mean(torch.stack(tf_feats_list), dim=0)     # [batch, 32]

        # 4. Объединяем всё
        combined = torch.cat([half_feat, window_feats, tf_feat], dim=1)
        logits = self.fusion(combined)
        
        return logits  # [batch, 3] → softmax на Long/Short/Flat

    def _df_to_tensor(self, df: pl.DataFrame) -> torch.Tensor:
        """Преобразование Polars DataFrame свечей в тензор [batch=1, channels, time]."""
        # Здесь должна быть реальная реализация (выбор колонок, нормализация и т.д.)
        # Пока заглушка
        return torch.zeros(1, self.in_channels, len(df), device=self.device)