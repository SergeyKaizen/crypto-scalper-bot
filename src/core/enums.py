# src/core/enums.py
"""
Все перечисления (Enums) проекта.
Используются для типобезопасности и читаемости кода.
Вместо строк 'C', 'L', 'classic' — используются константы.

Пример:
    if anomaly_type == AnomalyType.MIXED:
        # ...
    direction = Direction.BOTH
"""

from enum import Enum, auto


class AnomalyType(Enum):
    """Типы аномалий / сигналов"""
    CANDLE = "C"        # Свечная аномалия
    VOLUME = "V"        # Объёмная аномалия
    MIXED = "CV"        # Свечная + объёмная
    QUIET = "Q"         # Тихий режим (паттерн от модели без явной аномалии)


class Direction(Enum):
    """Направление позиции / сигнала"""
    LONG = "L"          # Только лонг
    SHORT = "S"         # Только шорт
    BOTH = "LS"         # Прибыльно и в лонг, и в шорт (PR_LS максимальный)


class TpMode(Enum):
    """Режим Take Profit"""
    AVERAGE_CANDLE = auto()     # Средний размер свечи (фиксированный)
    DYNAMIC_LEVEL = auto()      # Следующий уровень VAH/VAL или channel


class SlMode(Enum):
    """Режим Stop Loss"""
    CLASSIC = auto()            # Ближайший HH/LL ±0.05%, лимит 2×средний размер
    ATR_CHANDELIER = auto()     # ATR × multiplier + chandelier exit


class TrailingType(Enum):
    """Тип trailing stop"""
    CLASSIC = auto()            # Активация % + шаг % перестановки SL
    DISTANCE = auto()           # SL всегда на % от high/low после активации


class RiskBase(Enum):
    """База для расчёта риска на сделку"""
    INITIAL = auto()            # От начального депозита (фиксировано)
    CURRENT = auto()            # От текущего баланса (растёт/падает)


class PrMode(Enum):
    """Режим расчёта Profitable Rating"""
    CLASSIC = auto()            # Простой: (TP_count × TP_size) - (SL_count × SL_size)
    NORMALIZED = auto()         # × log10(кол-во +5) / 2.0 (штраф за малое кол-во)


class TradingMode(Enum):
    """Режим торговли (real / virtual)"""
    REAL = auto()
    VIRTUAL = auto()


class HardwareProfile(Enum):
    """Профили железа"""
    PHONE_TINY = auto()
    COLAB = auto()
    SERVER = auto()