"""
tests/test_risk.py

=== Основной принцип работы файла ===

Тесты модуля risk_manager.py

Проверяет:
- Корректный расчёт размера позиции (fixed / auto режим)
- Соблюдение global max_exposure_pct
- Защиту от отрицательного/бесконечного размера позиции
- Обновление депозита после закрытия сделок
"""

import pytest
from src.trading.risk_manager import RiskManager
from src.core.config import load_config

def test_risk_manager_fixed_mode():
    rm = RiskManager()
    config = load_config()
    config["trading"]["risk_mode"] = "fixed"
    config["trading"]["risk_pct"] = 0.01
    
    size = rm.calculate_position_size(
        balance=1000,
        entry_price=50000,
        stop_price=49000,
        symbol="BTCUSDT"
    )
    assert 0 < size < 1000 * 0.01 / 0.02  # примерный RR

def test_risk_manager_auto_mode():
    rm = RiskManager()
    config = load_config()
    config["trading"]["risk_mode"] = "auto"
    
    size = rm.calculate_position_size(
        balance=1000,
        entry_price=50000,
        stop_price=49000,
        tp_price=52000,
        symbol="BTCUSDT"
    )
    assert size > 0

def test_max_exposure_limit():
    rm = RiskManager()
    rm.current_exposure = 0.25
    config = load_config()
    config["trading"]["max_exposure_pct"] = 0.3
    
    can_open = rm.can_open_new_position(0.1)
    assert can_open is True

    can_open = rm.can_open_new_position(0.2)
    assert can_open is False

def test_update_deposit_after_close():
    rm = RiskManager()
    rm.balance = 1000
    rm.update_deposit(150.0)
    assert rm.balance == 1150.0