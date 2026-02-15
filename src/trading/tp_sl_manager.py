# src/trading/tp_sl_manager.py
"""
Менеджер расчёта и управления Take-Profit / Stop-Loss.

Ключевые требования из ТЗ:
- Авто-расчёт TP/SL на основе риск-профита (базово период 100 свечей)
- Возможность ручного указания риска (0.5%, 1.1%, 2.0% и т.д.)
- Несколько режимов TP/SL:
  - classic          → фиксированный TP = 1× средний размер свечи
  - dynamic_level    → уровни на основе VAH/VAL или ценового канала
  - chandelier       → ATR-based trailing stop (как Chandelier Exit)
  - partial_trailing → частичная фиксация + трейлинг (новый режим, опциональный)

Все режимы переключаются через конфиг trading_modes/*.yaml.
По умолчанию — classic (самый простой и надёжный для скальпинга).
partial_trailing — только если явно включён (рекомендуется для агрессивного режима).

Логика:
- Рассчитываем avg_candle_pct — средний % размер свечи за период (базово 100)
- На основе него строим уровни TP/SL
- Поддержка partial take-profit + trailing для части позиции
"""

from typing import Dict, Any, Optional
import polars as pl
from src.utils.logger import get_logger

logger = get_logger(__name__)

class TPSLManager:
    """
    Менеджер TP/SL для одной позиции или всех позиций.
    Один экземпляр на сессию / монету.
    """

    def __init__(self, config: dict):
        self.config = config
        self.tp_sl_mode = config["trading_mode"]["tp_sl_mode"]  # classic / dynamic_level / chandelier / partial_trailing
        
        # Период для расчёта среднего размера свечи (из ТЗ)
        self.avg_candle_period = config.get("model", {}).get("tp_sl_period", 100)
        
        # Коэффициенты для разных режимов (можно переопределять в yaml)
        self.coeffs = {
            "classic": {
                "tp_mult": 1.0,      # TP = 1 × avg_candle
                "sl_mult": 1.0       # SL = 1 × avg_candle (или по риску)
            },
            "partial_trailing": {
                "tp1_mult": 1.0,
                "tp1_portion": 0.50,
                "tp2_mult": 1.5,
                "tp2_portion": 0.30,
                "trailing_portion": 0.20,
                "trailing_distance_mult": 0.5  # 50% от avg_candle
            }
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
            logger.warning("avg_candle_pct <= 0 → использую fallback 0.5%")
            avg_candle_pct = 0.5

        mode = self.tp_sl_mode
        result = {
            "mode": mode,
            "direction": direction,
            "entry_price": entry_price,
            "position_size": position_size
        }

        if mode == "classic":
            # Простой режим: один TP и один SL
            tp_mult = self.coeffs["classic"]["tp_mult"]
            sl_mult = self.coeffs["classic"]["sl_mult"]

            if direction == "L":
                tp_price = entry_price * (1 + avg_candle_pct * tp_mult)
                sl_price = entry_price * (1 - avg_candle_pct * sl_mult)
            else:
                tp_price = entry_price * (1 - avg_candle_pct * tp_mult)
                sl_price = entry_price * (1 + avg_candle_pct * sl_mult)

            # Если задан ручной риск — переопределяем SL
            if risk_pct is not None:
                if direction == "L":
                    sl_price = entry_price * (1 - risk_pct / 100)
                else:
                    sl_price = entry_price * (1 + risk_pct / 100)

            result.update({
                "tp_price": tp_price,
                "sl_price": sl_price,
                "tp_portion": 1.0,      # вся позиция
                "sl_type": "fixed"
            })

        elif mode == "partial_trailing":
            # Частичная фиксация + трейлинг (рекомендуется для агрессивного режима)
            cfg = self.coeffs["partial_trailing"]

            if direction == "L":
                tp1 = entry_price * (1 + avg_candle_pct * cfg["tp1_mult"])
                tp2 = entry_price * (1 + avg_candle_pct * cfg["tp2_mult"])
                trailing_dist = avg_candle_pct * cfg["trailing_distance_mult"]
                sl_price = entry_price * (1 - avg_candle_pct)  # начальный SL
            else:
                tp1 = entry_price * (1 - avg_candle_pct * cfg["tp1_mult"])
                tp2 = entry_price * (1 - avg_candle_pct * cfg["tp2_mult"])
                trailing_dist = avg_candle_pct * cfg["trailing_distance_mult"]
                sl_price = entry_price * (1 + avg_candle_pct)

            # Если ручной риск — переопределяем начальный SL
            if risk_pct is not None:
                if direction == "L":
                    sl_price = entry_price * (1 - risk_pct / 100)
                else:
                    sl_price = entry_price * (1 + risk_pct / 100)

            result.update({
                "tp1": {"price": tp1, "portion": cfg["tp1_portion"]},
                "tp2": {"price": tp2, "portion": cfg["tp2_portion"]},
                "trailing": {"distance_pct": trailing_dist, "portion": cfg["trailing_portion"]},
                "initial_sl": sl_price,
                "sl_type": "trailing_after_tp1"
            })

        elif mode == "dynamic_level":
            # TODO: реализация на основе VAH/VAL или канала (из channels.py)
            # Пока заглушка — возвращаем classic
            logger.warning("dynamic_level не полностью реализован → fallback на classic")
            return self.calculate_levels(entry_price, direction, avg_candle_pct, position_size, risk_pct)

        elif mode == "chandelier":
            # TODO: ATR-based trailing (Chandelier Exit)
            # Пока заглушка
            logger.warning("chandelier не полностью реализован → fallback на classic")
            return self.calculate_levels(entry_price, direction, avg_candle_pct, position_size, risk_pct)

        else:
            raise ValueError(f"Неизвестный tp_sl_mode: {mode}")

        logger.debug(f"TP/SL рассчитаны: mode={mode}, direction={direction}, avg_candle={avg_candle_pct:.2f}%")
        return result

    def update_trailing(self, current_price: float, position: Dict) -> Optional[float]:
        """
        Обновление trailing stop (если режим partial_trailing или chandelier).
        Возвращает новый SL или None, если не нужно обновлять.
        """
        if self.tp_sl_mode not in ["partial_trailing", "chandelier"]:
            return None

        # Пример для partial_trailing (можно расширить)
        if "trailing" in position.get("tp_sl_levels", {}):
            dist_pct = position["tp_sl_levels"]["trailing"]["distance_pct"]
            if position["direction"] == "L":
                new_sl = current_price * (1 - dist_pct)
            else:
                new_sl = current_price * (1 + dist_pct)
            
            # Trailing только вверх (для лонга) или вниз (для шорта)
            current_sl = position.get("current_sl", 0)
            if (position["direction"] == "L" and new_sl > current_sl) or \
               (position["direction"] == "S" and new_sl < current_sl):
                position["current_sl"] = new_sl
                return new_sl
        
        return None


# Пример использования
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