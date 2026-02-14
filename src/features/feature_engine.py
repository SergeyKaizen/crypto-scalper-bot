# ================================================
# 3. src/features/feature_engine.py (полная новая версия)
# ================================================
import polars as pl
from .half_comparator import HalfComparator
from .anomaly_detector import AnomalyDetector
from ..core.types import ModelInput

class FeatureEngine:
    """Центральный класс подготовки данных для модели."""

    def __init__(self, config: dict):
        self.config = config
        self.half_comparator = HalfComparator(config)
        self.anomaly_detector = AnomalyDetector(config)
        self.windows = [24, 50, 74, 100]   # 4 окна из ТЗ

    def process(self, dfs: dict[str, pl.DataFrame]) -> ModelInput:
        """
        Главный метод подготовки входа для нейронки.
        1. Глобальное сравнение половин (только 100 свечей)
        2. 4 окна как отдельные последовательности
        3. Все аномалии
        4. Multi-TF данные
        """
        df1m = dfs["1m"]
        
        # 1. Глобальное сравнение половин основного периода
        half_result = self.half_comparator.compare(df1m.tail(100))
        
        # 2. Окна как отдельные последовательности (НЕ делим на половины)
        windows = {}
        for w in self.windows:
            windows[str(w)] = df1m.tail(w)
        
        # 3. Детекция всех аномалий на всех TF
        anomalies = self.anomaly_detector.detect_all(dfs)
        
        # 4. Multi-TF данные для отдельных веток модели
        multi_tf = {tf: df.tail(100) for tf, df in dfs.items()}
        
        return ModelInput(
            half=half_result,
            windows=windows,
            anomalies=anomalies,
            multi_tf=multi_tf
        )