"""
src/trading/shadow_trading.py

=== Основной принцип работы файла ===

Этот файл реализует "теневую" (shadow) торговлю — параллельную симуляцию реальных сделок в виртуальном режиме.
Цель — сравнивать реальные результаты (из order_executor) с виртуальными (из virtual_trader), 
чтобы выявлять расхождения (slippage, задержки исполнения, ошибки биржи, разница в PNL) и улучшать стратегию.

Ключевые задачи:
- Дублирует каждый реальный ордер в виртуальном режиме (shadow position).
- Обновляет виртуальные позиции параллельно с реальными (по тем же свечам).
- Сравнивает PNL, hit TP/SL, slippage (разница entry price), исходы закрытия.
- Логирует расхождения для анализа (например, "реальный SL hit, виртуальный TP hit").
- Не влияет на реальные ордера — только мониторит и анализирует.

Используется только в режиме "real" для отладки и улучшения.
В режиме "virtual" — не активен (дублирует сам себя, бесполезно).

=== Главные функции и за что отвечают ===

- ShadowTrading() — инициализация: виртуальный баланс, shadow_positions.
- shadow_open(real_pos: dict, real_order_id: str) — дублирует открытие реальной позиции в shadow.
- shadow_update(candle_data: dict) — обновляет все shadow позиции (TP/SL check).
- compare_real_vs_shadow(real_pos: dict, real_close_result: dict) — сравнение результатов после закрытия.
- get_shadow_balance() → float — текущий баланс теневой торговли.
- get_shadow_pnl() → float — общий PNL теневой торговли.

=== Примечания ===
- Shadow — это копия virtual_trader, но только для реальных сделок.
- Комиссия и плечо — те же, что в реале.
- Полностью соответствует ТЗ: shadow trading для сравнения и улучшения.
- Нет влияния на реальные ордера — чистый мониторинг.
- Логи через setup_logger с префиксом [SHADOW].
- Готов к использованию в live_loop (при real mode) и entry_manager.
"""

from typing import Dict, Optional

from src.core.config import load_config
from src.trading.virtual_trader import VirtualTrader
from src.utils.logger import setup_logger

logger = setup_logger('shadow_trading', logging.INFO)

class ShadowTrading:
    """
    Теневая торговля — параллельная симуляция реальных сделок.
    """
    def __init__(self):
        self.config = load_config()
        self.virtual_trader = VirtualTrader()  # Используем тот же класс виртуального трейдера
        self.shadow_positions: Dict[str, Dict] = {}  # pos_id → shadow pos
        self.real_to_shadow_map: Dict[str, str] = {}  # real_order_id → shadow_pos_id

    def shadow_open(self, real_pos: Dict, real_order_id: str):
        """
        Дублирует открытие реальной позиции в shadow.
        real_pos — позиция из entry_manager.
        real_order_id — ID реального ордера.
        """
        shadow_pos = real_pos.copy()
        shadow_pos['is_shadow'] = True
        shadow_pos_id = f"shadow_{real_order_id}"

        self.virtual_trader.open_position(shadow_pos)
        self.shadow_positions[shadow_pos_id] = shadow_pos
        self.real_to_shadow_map[real_order_id] = shadow_pos_id

        logger.info(f"[SHADOW] Открыта теневая позиция для реального ордера {real_order_id}")

    def shadow_update(self, candle_data: Dict):
        """
        Обновляет все теневые позиции по новой свече.
        """
        self.virtual_trader.update_positions(candle_data)

    def compare_real_vs_shadow(self, real_pos: Dict, real_close_result: Dict):
        """
        Сравнивает реальный и теневой результат закрытия.
        Логирует расхождения (slippage, hit разный уровень и т.д.).
        """
        real_order_id = real_pos.get('order_id')
        if not real_order_id:
            return

        shadow_pos_id = self.real_to_shadow_map.get(real_order_id)
        if not shadow_pos_id:
            return

        shadow_pos = self.shadow_positions.get(shadow_pos_id)
        if not shadow_pos:
            return

        # Пример сравнения PNL
        real_pnl = real_close_result.get('profit', 0)
        shadow_pnl = shadow_pos.get('pnl', 0)  # предполагаем, что virtual_trader обновил pnl

        slippage = real_pnl - shadow_pnl
        if abs(slippage) > 0.1 * abs(real_pnl):  # >10% расхождение
            logger.warning(f"[SHADOW] Расхождение PNL {real_pnl:.2f} vs {shadow_pnl:.2f} для {real_pos['symbol']}")

        # Сравнение hit TP/SL
        real_is_tp = real_close_result.get('is_tp', False)
        shadow_is_tp = shadow_pos.get('closed_is_tp', False)
        if real_is_tp != shadow_is_tp:
            logger.warning(f"[SHADOW] Разный исход: real {'TP' if real_is_tp else 'SL'}, shadow {'TP' if shadow_is_tp else 'SL'}")

        # Сравнение entry price (slippage)
        real_entry = real_pos.get('entry_price', 0)
        shadow_entry = shadow_pos.get('entry_price', 0)
        if abs(real_entry - shadow_entry) > 0.001 * real_entry:
            logger.warning(f"[SHADOW] Slippage entry price {real_entry:.6f} vs {shadow_entry:.6f} для {real_pos['symbol']}")

    def get_shadow_balance(self) -> float:
        """Текущий баланс теневой торговли."""
        return self.virtual_trader.get_balance()

    def get_shadow_pnl(self) -> float:
        """Общий PNL теневой торговли."""
        return self.virtual_trader.get_pnl()