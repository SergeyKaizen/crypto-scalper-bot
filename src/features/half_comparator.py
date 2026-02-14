# ================================================
# 1. src/features/half_comparator.py
# ================================================
"""
ЕДИНСТВЕННАЯ ТОЧКА ПРАВДЫ для сравнения левой и правой половины 
основного периода (по умолчанию 100 свечей 1m).
Используется в feature_engine, backtest, inference и scenario_tracker.
"""

from dataclasses import dataclass
import polars as pl
from typing import Dict, Optional

@dataclass
class HalfComparisonResult:
    """Результат сравнения двух половин периода."""
    binary_vector: list[int]          # 16 бит (0/1) — именно то, что идёт в HDBSCAN
    percent_changes: Dict[str, float] # процентное изменение каждого признака
    left_features: Dict[str, float]
    right_features: Dict[str, float]
    is_valid: bool = True


class HalfComparator:
    """Сравнивает левую и правую половину основного периода (только 100 свечей)."""

    def __init__(self, config: dict):
        """Загружаем параметры из конфига."""
        self.period = config["model"]["main_period"]          # обычно 100
        # 12 признаков + 3 условия = 15, + 1 запасной бит = 16
        self.feature_names = [
            "volume", "bid", "ask", "delta", "price_change",
            "avg_price", "volatility", "price_channel_pos",
            "va_pos", "candle_anomaly", "volume_anomaly", "cv_anomaly"
        ]

    def compare(self, df: pl.DataFrame, period: Optional[int] = None) -> HalfComparisonResult:
        """
        Основной метод проекта.
        Берёт последние N свечей, делит строго пополам и сравнивает все признаки.
        Возвращает 16-битный вектор + процентные изменения.
        """
        period = period or self.period
        
        if len(df) < period:
            return HalfComparisonResult([], {}, {}, {}, False)

        # === 1. Делим период ровно пополам ===
        half = period // 2
        left = df.tail(period).head(half)      # более старые данные
        right = df.tail(half)                  # более свежие данные

        # === 2. Извлекаем признаки для каждой половины ===
        # (реальная реализация признаков вызывается из FeatureEngine)
        left_feats = self._extract_features(left)
        right_feats = self._extract_features(right)

        # === 3. Формируем 16-битный вектор и процентные изменения ===
        binary_vector = []
        percent_changes = {}

        for name in self.feature_names:
            l = left_feats.get(name, 0.0)
            r = right_feats.get(name, 0.0)
            
            if abs(l) < 1e-9 and abs(r) < 1e-9:
                binary_vector.append(0)
                percent_changes[name] = 0.0
                continue
                
            change_pct = ((r - l) / abs(l)) * 100 if abs(l) > 1e-9 else 0.0
            percent_changes[name] = change_pct
            binary_vector.append(1 if r > l else 0)   # 1 = увеличилось, 0 = уменьшилось

        # Дополняем до ровно 16 бит
        while len(binary_vector) < 16:
            binary_vector.append(0)

        return HalfComparisonResult(
            binary_vector=binary_vector[:16],
            percent_changes=percent_changes,
            left_features=left_feats,
            right_features=right_feats
        )

    def _extract_features(self, df: pl.DataFrame) -> Dict[str, float]:
        """Заглушка. Реальные расчёты 12 признаков будут в FeatureEngine."""
        return {name: 0.0 for name in self.feature_names}