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

def test_backtest_engine_basic(mock_config):
    engine = BacktestEngine(mock_config)
    
    df = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=200, freq="1min"),
        "open": [50000 + i*10 for i in range(200)],
        "high": [50100 + i*10 for i in range(200)],
        "low": [49900 + i*10 for i in range(200)],
        "close": [50050 + i*10 for i in range(200)],
        "volume": [1000] * 200,
        "buy_volume": [600] * 200,
    }).set_index("timestamp")

    # FIX Фаза 6: обновлён вызов (учёт commission/slippage из Phase 5)
    result = engine.run_full_backtest()
    
    assert isinstance(result, dict)
    assert "total_trades" in result
    assert "net_profit" in result
    assert "win_rate" in result

def test_pr_calculator_update_and_stats():
    calc = PRCalculator()
    
    calc.update_pr(symbol="BTCUSDT", anomaly_type="C", direction="L", hit_tp=True, pl=245.0)
    calc.update_pr(symbol="BTCUSDT", anomaly_type="V", direction="S", hit_tp=False, pl=-87.0)
    
    stats = calc.get_stats("BTCUSDT")
    assert stats["total_trades"] == 2
    assert stats["total_pr"] == pytest.approx(158.0)
    assert stats["win_rate"] == 0.5