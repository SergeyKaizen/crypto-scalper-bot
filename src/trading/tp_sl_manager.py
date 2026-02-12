# src/trading/tp_sl_manager.py
"""
Модуль управления Take Profit, Stop Loss и Trailing Stop.

Основные функции:
- calculate_tp() — расчёт TP при открытии позиции
- calculate_sl() — расчёт SL при открытии позиции
- calculate_sl_distance_pct() — расстояние до SL в % (для risk_manager)
- update_trailing() — обновление trailing stop на каждом новом баре
- get_dynamic_tp_level() — следующий уровень VAH/VAL или channel

Логика:
- TP modes:
  - average_candle — средний размер свечи за последние 100 свечей
  - dynamic_level — следующий VAH (long) / VAL (short) или channel upper/lower
- SL modes:
  - classic — ближайший HH/LL ±0.05%, но не дальше 2×средний размер свечи
  - atr_chandelier — lowest low (long) / highest high (short) -/+ ATR × multiplier
- Trailing:
  - classic — активация при +activation_pct, потом шаг перестановки
  - distance — SL всегда на distance_pct от high/low после активации

Конфиг-зависимые параметры:
- tp_mode
- sl_mode
- trailing_enabled, trailing_type, trailing_activation_pct, trailing_distance_pct
- atr_period, atr_multiplier (для chandelier)
"""

import logging
from typing import Dict, Tuple, Optional

import polars as pl

from src.core.config import load_config
from src.core.types import Position
from src.features.channels import ChannelsCalculator

logger = logging.getLogger(__name__)


class TpSlManager:
    """Управление TP, SL и Trailing Stop"""

    def __init__(self, config: Dict):
        self.config = config
        self.tp_mode = config["trading"]["tp_mode"]
        self.sl_mode = config["trading"]["sl_mode"]
        self.trailing_enabled = config["trading"]["trailing_enabled"]
        self.trailing_type = config["trading"]["trailing_type"]
        self.activation_pct = config["trading"]["trailing_activation_pct"]
        self.distance_pct = config["trading"]["trailing_distance_pct"]
        self.atr_period = config["features"].get("atr_period", 14)
        self.atr_multiplier = config["features"].get("atr_multiplier", 2.0)

        self.channel_calc = ChannelsCalculator(config)

        logger.info("TpSlManager initialized: TP=%s, SL=%s, Trailing=%s (%s)",
                    self.tp_mode, self.sl_mode, self.trailing_enabled, self.trailing_type)

    def calculate_tp(self, df: pl.DataFrame, entry_price: float, direction: str) -> float:
        """Расчёт Take Profit при открытии позиции"""
        if self.tp_mode == "average_candle":
            sizes = df["high"] - df["low"]
            avg_size = sizes.mean()
            if direction == "L":
                return entry_price + avg_size * 1.0  # 1× средний размер
            else:
                return entry_price - avg_size * 1.0

        elif self.tp_mode == "dynamic_level":
            channel = self.channel_calc.calculate_price_channel(df.tail(100))
            va = self.channel_calc.calculate_value_area(df.tail(100))

            if direction == "L":
                return max(channel["upper"], va["vah"])
            else:
                return min(channel["lower"], va["val"])

        logger.warning("Unknown TP mode: %s. Using average_candle", self.tp_mode)
        return entry_price * (1.005 if direction == "L" else 0.995)  # Fallback 0.5%

    def calculate_sl(self, df: pl.DataFrame, entry_price: float, direction: str) -> float:
        """Расчёт Stop Loss при открытии позиции"""
        if self.sl_mode == "classic":
            # Ближайший HH/LL ±0.05%
            if direction == "L":
                recent_low = df.tail(100)["low"].min()
                sl = recent_low * 0.9995  # -0.05%
            else:
                recent_high = df.tail(100)["high"].max()
                sl = recent_high * 1.0005  # +0.05%

            # Ограничение: не дальше 2× средний размер свечи
            avg_size = (df["high"] - df["low"]).mean()
            max_sl_distance = entry_price * (2 * avg_size / entry_price)
            if direction == "L" and entry_price - sl > max_sl_distance:
                sl = entry_price - max_sl_distance
            elif direction == "S" and sl - entry_price > max_sl_distance:
                sl = entry_price + max_sl_distance

            return sl

        elif self.sl_mode == "atr_chandelier":
            # Chandelier exit
            atr = self._calculate_atr(df.tail(100))
            if direction == "L":
                lowest_low = df.tail(self.atr_period)["low"].min()
                sl = lowest_low - atr * self.atr_multiplier
            else:
                highest_high = df.tail(self.atr_period)["high"].max()
                sl = highest_high + atr * self.atr_multiplier

            return sl

        logger.warning("Unknown SL mode: %s. Using classic", self.sl_mode)
        return entry_price * (0.995 if direction == "L" else 1.005)  # Fallback 0.5%

    def calculate_sl_distance_pct(self, entry_price: float, sl_price: float, direction: str) -> float:
        """Расстояние до SL в % (для risk_manager)"""
        if direction == "L":
            return abs(entry_price - sl_price) / entry_price * 100
        else:
            return abs(sl_price - entry_price) / entry_price * 100

    def update_trailing(self, position: Position, current_high: float, current_low: float) -> Position:
        """Обновление trailing stop на новом баре"""
        if not self.trailing_enabled or not position.trailing_active:
            # Проверка активации
            if position.direction == "L":
                profit_pct = (current_high - position.entry_price) / position.entry_price * 100
            else:
                profit_pct = (position.entry_price - current_low) / position.entry_price * 100

            if profit_pct >= self.activation_pct:
                position.trailing_active = True
                if self.trailing_type == "classic":
                    position.trailing_price = position.entry_price * (1 + self.activation_pct / 100) if position.direction == "L" else position.entry_price * (1 - self.activation_pct / 100)
                else:  # distance
                    position.trailing_price = current_high * (1 - self.distance_pct / 100) if position.direction == "L" else current_low * (1 + self.distance_pct / 100)

                logger.info("Trailing activated for %s at %.2f", position.symbol, position.trailing_price)
            return position

        # Обновление trailing
        if self.trailing_type == "classic":
            # Пример классического шага (можно усложнить)
            if position.direction == "L" and current_high > position.trailing_price:
                position.trailing_price = current_high * 0.995  # -0.5% от нового high
        else:  # distance
            if position.direction == "L" and current_high > position.trailing_price / (1 - self.distance_pct / 100):
                position.trailing_price = current_high * (1 - self.distance_pct / 100)
            elif position.direction == "S" and current_low < position.trailing_price / (1 + self.distance_pct / 100):
                position.trailing_price = current_low * (1 + self.distance_pct / 100)

        return position

    def _calculate_atr(self, df: pl.DataFrame) -> float:
        """Расчёт ATR (Average True Range)"""
        tr = pl.max_horizontal(
            df["high"] - df["low"],
            (df["high"] - df["close"].shift(1)).abs(),
            (df["low"] - df["close"].shift(1)).abs()
        )
        atr = tr.ewm_mean(span=self.atr_period, adjust=False).mean()
        return atr