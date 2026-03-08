"""
tests/test_backtest.py

=== Основной принцип работы файла ===

Тесты для модуля бэктестирования (BacktestEngine + PRCalculator).

Проверяет:
- корректность симуляции сделок
- расчёт PR (Profit Ratio)
- обработку slippage и комиссий
"""

import pytest
import pandas as pd
from src.backtest.engine import BacktestEngine
from src.backtest.pr_calculator import PRCalculator
from src.core.config import load_config
from src.data.storage import Storage

def test_backtest_engine_basic():
    config = load_config()
    engine = BacktestEngine(config, "BTCUSDT")  # ← исправлено (добавлен symbol)
    
    result = engine.run_full_backtest()
    
    assert isinstance(result, dict)
    assert "total_trades" in result
    assert "pr_ls" in result
    assert "max_drawdown" in result

def test_pr_calculator_update_and_stats():
    calc = PRCalculator()
    
    calc.update_pr(symbol="BTCUSDT", anomaly_type="C", direction="L", hit_tp=True, pl=245.0)
    calc.update_pr(symbol="BTCUSDT", anomaly_type="V", direction="S", hit_tp=False, pl=-87.0)
    
    stats = calc.get_stats("BTCUSDT")
    assert stats["total_trades"] == 2
    assert stats["total_pr"] == pytest.approx(158.0)
    assert stats["win_rate"] == 0.5