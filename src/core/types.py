# src/core/types.py
"""
Все основные типы данных проекта (dataclasses + TypedDict).
Используются для строгой типизации и передачи данных между модулями.

Преимущества:
- Автодополнение в IDE (VS Code, PyCharm)
- Ошибки на этапе написания кода, а не в runtime
- Читаемость — сразу видно, какие поля у объекта

Основные типы:
- Candle       — одна свеча (OHLCV + bid/ask volume)
- Signal       — сигнал от модели (аномалия + направление + вероятность)
- Position     — открытая позиция (вход, TP, SL, trailing)
- TradeResult  — результат закрытия (TP/SL, профит/убыток)
- Scenario     — бинарный сценарий для HDBSCAN
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Union, Tuple
from typing_extensions import TypedDict


@dataclass
class Candle:
    """Одна свеча (klines) с Binance Futures"""
    timestamp: datetime             # Время открытия свечи (ms → datetime)
    open: float
    high: float
    low: float
    close: float
    volume: float                   # Общий объём
    bid_volume: float               # Объём рыночных продаж (bid)
    ask_volume: float               # Объём рыночных покупок (ask)
    symbol: str                     # Например "BTCUSDT"
    timeframe: str                  # "1m", "5m" и т.д.

    def __post_init__(self):
        """Проверка корректности данных"""
        if self.high < self.low:
            raise ValueError("High < Low в свече")
        if self.volume < 0:
            raise ValueError("Отрицательный volume")


@dataclass
class Signal:
    """Сигнал от модели — основа для открытия позиции"""
    timestamp: datetime
    symbol: str                     # "BTCUSDT"
    timeframe: str                  # "1m", "5m" и т.д.
    window: int                     # Размер окна (24, 50, 74, 100)
    anomaly_type: str               # "C", "V", "CV", "Q"
    direction: str                  # "L", "S", "LS"
    probability: float              # 0.0–1.0 (вероятность профита)
    expected_return: Optional[float] = None  # Ожидаемая доходность в % (регрессия)
    features: Optional[Dict] = None  # Для отладки — все признаки

    def __post_init__(self):
        if not 0 <= self.probability <= 1:
            raise ValueError("Probability должна быть в [0, 1]")


@dataclass
class Position:
    """Открытая позиция (реальная или виртуальная)"""
    entry_time: datetime
    entry_price: float
    size: float                     # Размер позиции (в USDT или контрактах)
    direction: str                  # "L" / "S"
    tp_price: float
    sl_price: float
    trailing_active: bool = False
    trailing_price: Optional[float] = None  # Текущий уровень trailing
    soft_entry_parts: List[Tuple[float, float]] = field(default_factory=list)  # [(price, size), ...]
    symbol: str
    leverage: int = 50              # Плечо (по умолчанию 10x)
    is_virtual: bool = False        # Реальная или виртуальная

    @property
    def current_pnl_pct(self) -> float:
        """Текущий P&L в % (без учёта комиссии)"""
        if self.direction == "L":
            return (self.current_price - self.entry_price) / self.entry_price * 100
        else:
            return (self.entry_price - self.current_price) / self.entry_price * 100

    current_price: float = 0.0  # Обновляется в live_loop


@dataclass
class TradeResult:
    """Результат закрытия позиции"""
    position: Position
    exit_time: datetime
    exit_price: float
    pnl_pct: float                  # Профит/убыток в %
    pnl_usdt: float
    reason: str                     # "TP", "SL", "Trailing", "Manual"
    is_win: bool                    # True — TP, False — SL


class Scenario(TypedDict):
    """Бинарный сценарий для HDBSCAN (статистика сценариев)"""
    features: Dict[str, Union[int, float]]  # 12 признаков + 4 условия (0/1)
    outcome: int                            # 1 — профит, 0 — убыток
    count: int                              # Сколько раз встречался
    winrate: float                          # Винрейт сценария
    cluster_label: Optional[int]            # Метка кластера от HDBSCAN


# Пример использования (для тестов)
if __name__ == "__main__":
    candle = Candle(
        timestamp=datetime.now(),
        open=60000.0,
        high=60500.0,
        low=59800.0,
        close=60200.0,
        volume=1000.0,
        bid_volume=600.0,
        ask_volume=400.0,
        symbol="BTCUSDT",
        timeframe="1m"
    )
    print(candle)