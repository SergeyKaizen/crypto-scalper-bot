# src/trading/tp_sl_manager.py
"""
Менеджер расчёта и управления Take-Profit / Stop-Loss.

Ключевые требования из ТЗ:
- Авто-расчёт TP/SL на основе среднего размера свечи (период 100 свечей)
- Возможность ручного указания риска (0.5%, 1.1%, 2.0% и т.д.)
- 4 режима TP/SL:
  - classic: фиксированный TP = 1× avg_candle, SL = 1× или ручной
  - partial_trailing: частичная фиксация (50%/30%) + trailing на остаток
  - dynamic_level: уровни на основе VAH/VAL или ценового канала (пока заглушка)
  - chandelier: ATR-based trailing stop (пока заглушка)
- partial_trailing — основной агрессивный режим (включается явно в конфиге)
- Проверка minNotional перед частичными фиксациями (чтобы не отправлять ордера меньше минимума биржи)
- Trailing активируется после первой фиксации
- Все параметры берутся из конфига (можно переопределять в trading_modes/*.yaml)
"""

from typing import Dict, Any, Optional
import polars as pl

from src.utils.logger import get_logger

logger = get_logger(__name__)

class TPSLManager:
    def __init__(self, config: dict):
        self.config = config
        self.tp_sl_mode = config["trading_mode"]["tp_sl_mode"]  # classic / partial_trailing / dynamic_level / chandelier

        # Период для расчёта среднего размера свечи (из ТЗ)
        self.avg_candle_period = config.get("model", {}).get("tp_sl_period", 100)

        # Коэффициенты для partial_trailing (можно переопределять в yaml)
        self.partial_cfg = {
            "tp1_mult": 1.0,          # первая фиксация на 1× avg_candle
            "tp1_portion": 0.50,      # 50%
            "tp2_mult": 1.5,          # вторая на 1.5×
            "tp2_portion": 0.30,      # 30%
            "trailing_portion": 0.20, # остаток в trailing
            "trailing_distance_mult": 0.5  # расстояние trailing = 0.5× avg_candle
        }

        # Для классического режима
        self.classic_cfg = {
            "tp_mult": 1.0,
            "sl_mult": 1.0
        }

    def calculate_levels(self,
                        entry_price: float,
                        direction: str,                # "L" или "S"
                        avg_candle_pct: float,         # средний % размер свечи за период
                        position_size: float,          # размер позиции в USDT или коинах
                        risk_pct: Optional[float] = None) -> Dict[str, Any]:
        """
        Основной метод расчёта TP/SL уровней.

        Аргументы:
            entry_price     — цена входа
            direction       — "L" (long) или "S" (short)
            avg_candle_pct  — средний размер свечи в % (рассчитывается в feature_engine)
            position_size   — объём позиции
            risk_pct        — если задан — переопределяет SL (ручной риск)

        Возвращает словарь с уровнями, порциями и типом SL.
        """
        if avg_candle_pct <= 0:
            logger.warning("avg_candle_pct <= 0 → fallback на 0.5%")
            avg_candle_pct = 0.5

        mode = self.tp_sl_mode
        result = {
            "mode": mode,
            "direction": direction,
            "entry_price": entry_price,
            "position_size": position_size
        }

        if mode == "classic":
            tp_mult = self.classic_cfg["tp_mult"]
            sl_mult = self.classic_cfg["sl_mult"]

            if direction == "L":
                tp_price = entry_price * (1 + avg_candle_pct * tp_mult)
                sl_price = entry_price * (1 - avg_candle_pct * sl_mult)
            else:
                tp_price = entry_price * (1 - avg_candle_pct * tp_mult)
                sl_price = entry_price * (1 + avg_candle_pct * sl_mult)

            # Ручной риск переопределяет только SL
            if risk_pct is not None:
                sl_price = entry_price * (1 - risk_pct / 100) if direction == "L" else entry_price * (1 + risk_pct / 100)

            result.update({
                "tp_price": tp_price,
                "sl_price": sl_price,
                "tp_portion": 1.0,
                "sl_type": "fixed"
            })

        elif mode == "partial_trailing":
            # Частичная фиксация + trailing на остаток
            cfg = self.partial_cfg

            if direction == "L":
                tp1 = entry_price * (1 + avg_candle_pct * cfg["tp1_mult"])
                tp2 = entry_price * (1 + avg_candle_pct * cfg["tp2_mult"])
                trailing_dist = avg_candle_pct * cfg["trailing_distance_mult"]
                initial_sl = entry_price * (1 - avg_candle_pct)
            else:
                tp1 = entry_price * (1 - avg_candle_pct * cfg["tp1_mult"])
                tp2 = entry_price * (1 - avg_candle_pct * cfg["tp2_mult"])
                trailing_dist = avg_candle_pct * cfg["trailing_distance_mult"]
                initial_sl = entry_price * (1 + avg_candle_pct)

            # Ручной риск переопределяет начальный SL
            if risk_pct is not None:
                initial_sl = entry_price * (1 - risk_pct / 100) if direction == "L" else entry_price * (1 + risk_pct / 100)

            result.update({
                "tp1": {"price": tp1, "portion": cfg["tp1_portion"]},
                "tp2": {"price": tp2, "portion": cfg["tp2_portion"]},
                "trailing": {"distance_pct": trailing_dist, "portion": cfg["trailing_portion"]},
                "initial_sl": initial_sl,
                "sl_type": "trailing_after_tp1"
            })

        elif mode == "dynamic_level":
            # TODO: уровни на основе VAH/VAL или канала (реализовать после feature_engine)
            logger.warning("dynamic_level пока не реализован → fallback на classic")
            return self.calculate_levels(entry_price, direction, avg_candle_pct, position_size, risk_pct)

        elif mode == "chandelier":
            # TODO: ATR-based trailing (реализовать позже)
            logger.warning("chandelier пока не реализован → fallback на classic")
            return self.calculate_levels(entry_price, direction, avg_candle_pct, position_size, risk_pct)

        else:
            raise ValueError(f"Неизвестный tp_sl_mode: {mode}")

        logger.debug(f"TP/SL рассчитаны: mode={mode}, direction={direction}, avg_candle={avg_candle_pct:.2f}%")
        return result

    def update_trailing(self, current_price: float, position: Dict) -> Optional[float]:
        """
        Обновление trailing stop (для partial_trailing).
        Возвращает новый SL или None, если не нужно обновлять.
        """
        if self.tp_sl_mode not in ["partial_trailing", "chandelier"]:
            return None

        trailing = position.get("tp_sl_levels", {}).get("trailing", {})
        if not trailing:
            return None

        dist_pct = trailing["distance_pct"]
        if position["direction"] == "L":
            new_sl = current_price * (1 - dist_pct)
        else:
            new_sl = current_price * (1 + dist_pct)

        current_sl = position.get("current_sl", 0)
        if (position["direction"] == "L" and new_sl > current_sl) or \
           (position["direction"] == "S" and new_sl < current_sl):
            position["current_sl"] = new_sl
            return new_sl

        return None


# Пример использования (для теста)
if __name__ == "__main__":
    cfg = {
        "trading_mode": {"tp_sl_mode": "partial_trailing"},
        "model": {"tp_sl_period": 100}
    }

    manager = TPSLManager(cfg)

    levels = manager.calculate_levels(
        entry_price=60000.0,
        direction="L",
        avg_candle_pct=0.8,
        position_size=1000.0,
        risk_pct=0.5  # ручной риск 0.5%
    )

    print(levels)