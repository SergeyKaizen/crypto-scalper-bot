"""
src/core/enums.py

=== Основной принцип работы файла ===

Этот файл содержит все перечисления (enums) проекта с использованием Python Enum.
Он обеспечивает типобезопасность, читаемость и удобство использования констант в коде.

Enums используются везде, где есть фиксированные категории:
- Типы аномалий (C, V, CV, Q)
- Направления позиций (Long/Short)
- Режимы торговли (real/virtual)
- Типы TP/SL (classic, partial_trailing и т.д. — без dynamic по ТЗ)
- Таймфреймы (1m, 3m, 5m, 10m, 15m)
- Hardware типы (phone_tiny, colab, server)

=== Главные enums и за что отвечают ===

- AnomalyType: типы аномалий и quiet режим
- Direction: Long / Short
- TradeMode: real / virtual
- TpSlMode: classic / partial_trailing / chandelier (dynamic удалён по ТЗ)
- Timeframe: 1m / 3m / 5m / 10m / 15m (с минутами)
- HardwareType: phone_tiny / colab / server (для выбора БД, epochs и т.д.)

=== Примечания ===
- Все enums наследуются от str, Enum — можно использовать как строки в БД/моделях.
- Нет заглушек — все типы из ТЗ и логики проекта.
- Импортируется как from src.core.enums import AnomalyType, Direction и т.д.
- Легко расширяем при необходимости (например, новые режимы риска).
"""

from enum import Enum, auto

class AnomalyType(str, Enum):
    """Типы аномалий и тихий режим."""
    CANDLE = "C"
    VOLUME = "V"
    CV = "CV"
    QUIET = "Q"

class Direction(str, Enum):
    """Направление позиции."""
    LONG = "L"          # FIX Фаза 7: "L"/"S" для полной совместимости с БД и whitelist
    SHORT = "S"

class TradeMode(str, Enum):
    REAL = "real"
    VIRTUAL = "virtual"

class TpSlMode(str, Enum):
    CLASSIC = "classic"
    PARTIAL_TRAILING = "partial_trailing"
    CHANDELIER = "chandelier"

class Timeframe(str, Enum):
    M1 = "1m"
    M3 = "3m"
    M5 = "5m"
    M10 = "10m"
    M15 = "15m"

    @property
    def minutes(self) -> int:
        return int(self.value[:-1])

class HardwareType(str, Enum):
    PHONE_TINY = "phone_tiny"
    COLAB = "colab"
    SERVER = "server"

class PositionStatus(str, Enum):
    OPEN = "open"
    CLOSED_TP = "closed_tp"
    CLOSED_SL = "closed_sl"
    CLOSED_MANUAL = "closed_manual"