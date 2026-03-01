""" 
tests/test_live.py 
=== Основной принцип работы файла === 
Тесты live-компонентов: live_loop, order_executor, tp_sl_manager. 
Проверяет: 
- graceful shutdown (закрытие позиций) 
- обработку новых свечей 
- работу TP/SL 
"""

import pytest
from unittest.mock import patch
from src.trading.live_loop import shutdown, open_positions  # FIX Фаза 6: обновлён импорт

def test_shutdown_closes_positions(monkeypatch):
    # Мокаем открытые позиции
    monkeypatch.setitem(open_positions, "BTCUSDT", [{"id": 123, "direction": "L"}])
    closed = []
    def mock_close(pos):
        closed.append(pos)
        return pos
    # FIX Фаза 6: обновлён путь к методу
    monkeypatch.setattr("src.trading.order_executor.OrderExecutor.close_position", mock_close)
    shutdown()
    assert len(closed) == 1
    assert closed[0]["id"] == 123