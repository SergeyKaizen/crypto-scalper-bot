# src/core/types.py
"""
Центральное место всех структур данных проекта.
Обновлено 14 февраля 2026:
- Добавлено поле тайм-аута в Position
- Уточнены типы для PR-обновления
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

import polars as pl


class AnomalyType(Enum):
    CANDLE = "C"
    VOLUME = "V"
    CANDLE_VOLUME = "CV"


class Direction(Enum):
    LONG = "L"
    SHORT = "S"
    FLAT = "F"


@dataclass
class HalfComparisonResult:
    binary_vector: List[int]
    percent_changes: Dict[str, float]
    left_features: Dict[str, float]
    right_features: Dict[str, float]
    is_valid: bool = True


@dataclass
class AnomalySignal:
    tf: str
    anomaly_type: AnomalyType
    direction_hint: Direction
    strength: float
    threshold: float
    current_volume: float
    candle_size_pct: float
    timestamp: int
    coin: Optional[str] = None


@dataclass
class ModelInput:
    half: HalfComparisonResult
    windows: Dict[str, pl.DataFrame]
    anomalies: List[AnomalySignal]
    multi_tf: Dict[str, pl.DataFrame]


@dataclass
class TradeConfig:
    """Финальная конфигурация монеты после фильтра PR."""
    tf: str
    period: int
    anomaly_type: AnomalyType
    direction: Direction
    pr_value: float
    last_update: datetime = field(default_factory=datetime.now)


@dataclass
class Position:
    """Единая структура для виртуальной и реальной позиции."""
    coin: str
    side: Direction
    entry_price: float
    size: float
    entry_time: int                     # unix ms
    tp_price: float
    sl_price: float
    anomaly_signal: AnomalySignal
    
    # === НОВОЕ: тайм-аут ===
    max_hold_time_ms: Optional[int] = None   # None = выключен (по умолчанию)
    is_real: bool = False
    order_id: Optional[str] = None
    close_reason: Optional[str] = None       # "tp", "sl", "timeout"