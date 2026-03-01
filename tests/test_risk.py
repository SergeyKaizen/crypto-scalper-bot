"""
tests/test_risk.py

=== Основной принцип работы файла ===

Тесты RiskManager — самого критичного модуля по безопасности капитала.

Проверяет:
- расчёт размера позиции
- соблюдение daily loss limit
- max positions limit
- обновление депозита
"""

import pytest
from src.trading.risk_manager import RiskManager

def test_calculate_position_size():
    rm = RiskManager()
    rm.deposit = 10000.0
    
    # FIX Фаза 6: обновлено имя метода
    size = rm.calculate_position_size(
        symbol="BTCUSDT",
        entry_price=60000.0,
        sl_price=59500.0,
        risk_pct=0.01
    )
    # 1% риска = 100$ / 500$ расстояния = 0.2 BTC
    assert abs(size - 0.2) < 1e-6

def test_daily_loss_limit():
    rm = RiskManager()
    rm.deposit = 10000.0
    rm.daily_loss_limit = 0.05  # 5%
    
    rm.update_deposit(-400)   # -4%
    # FIX Фаза 6: обновлено имя метода
    assert rm.can_open_new_position() is True
    
    rm.update_deposit(-200)   # ещё -2% → всего -6%
    assert rm.can_open_new_position() is False

def test_max_positions_limit():
    rm = RiskManager()
    rm.max_open_positions = 3
    
    # FIX Фаза 6: обновлено имя метода
    assert rm.can_open_new_position() is True
    rm.open_positions_count = 3
    assert rm.can_open_new_position() is False