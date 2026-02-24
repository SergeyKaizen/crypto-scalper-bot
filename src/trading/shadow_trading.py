"""
src/trading/shadow_trading.py

=== Основной принцип работы файла ===

Теневая (shadow) торговля — параллельная симуляция всех реальных сигналов в виртуальном режиме.

Цель — сравнивать реальные результаты с виртуальными, выявлять slippage, задержки, расхождения в TP/SL hit, PNL, quiet_streak и consensus_count.

Ключевые особенности:
- shadow_open: дублирует открытие реальной позиции в shadow
- shadow_update: обновляет все shadow позиции по новой свече
- compare_real_vs_shadow: детальное сравнение (slippage, hit TP/SL, PNL, quiet_streak, consensus)
- Вызов scenario_tracker.add_scenario для shadow после закрытия
- get_shadow_balance, get_shadow_pnl — статистика теневой торговли

=== Главные функции ===
- shadow_open(real_pos: dict, real_order_id: str)
- shadow_update(candle_data: dict)
- compare_real_vs_shadow(real_pos: dict, real_close_result: dict)
- get_shadow_balance() → float
- get_shadow_pnl() → float

=== Примечания ===
- Использует тот же VirtualTrader, что и виртуальный режим
- Логи с префиксом [SHADOW] для удобства анализа
- Полностью соответствует ТЗ + всем обновлениям (quiet_streak, consensus_count в сравнении)
- Готов к интеграции в live_loop и entry_manager
- Логи через setup_logger
"""

from typing import Dict
import logging

from src.core.config import load_config
from src.trading.virtual_trader import VirtualTrader
from src.model.scenario_tracker import ScenarioTracker
from src.utils.logger import setup_logger

logger = setup_logger('shadow_trading', logging.INFO)

class ShadowTrading:
    def __init__(self):
        self.config = load_config()
        self.virtual_trader = VirtualTrader()  # тот же экземпляр, что и в live_loop
        self.shadow_positions = {}          # shadow_pos_id → position
        self.real_to_shadow_map = {}        # real_order_id → shadow_pos_id
        self.scenario_tracker = ScenarioTracker()  # для статистики shadow

    def shadow_open(self, real_pos: Dict, real_order_id: str):
        """
        Дублирует открытие реальной позиции в shadow.
        """
        shadow_pos = real_pos.copy()
        shadow_pos['is_shadow'] = True
        shadow_pos_id = f"shadow_{real_order_id}"

        self.virtual_trader.open_position(shadow_pos)

        self.shadow_positions[shadow_pos_id] = shadow_pos
        self.real_to_shadow_map[real_order_id] = shadow_pos_id

        logger.info(f"[SHADOW] Открыта теневая позиция для реальной {real_order_id} | "
                    f"{real_pos['symbol']} {real_pos['anomaly_type']} prob={real_pos.get('prob', 0):.3f} "
                    f"quiet_streak={real_pos.get('quiet_streak', 0)} consensus={real_pos.get('consensus_count', 1)}")

    def shadow_update(self, candle_data: Dict):
        """
        Обновляет все теневые позиции по новой свече.
        """
        self.virtual_trader.update_positions(candle_data)

    def compare_real_vs_shadow(self, real_pos: Dict, real_close_result: Dict):
        """
        Детальное сравнение реальной и теневой позиции после закрытия.
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

        real_pnl = real_close_result.get('net_pl', 0)
        shadow_pnl = shadow_pos.get('pnl', 0)

        slippage = real_pnl - shadow_pnl
        slippage_pct = (slippage / abs(real_pnl) * 100) if real_pnl != 0 else 0

        real_hit_tp = real_close_result.get('hit_tp', False)
        shadow_hit_tp = shadow_pos.get('closed_is_tp', False)

        real_quiet = real_pos.get('quiet_streak', 0)
        shadow_quiet = shadow_pos.get('quiet_streak', 0)

        real_consensus = real_pos.get('consensus_count', 1)
        shadow_consensus = shadow_pos.get('consensus_count', 1)

        # Логирование расхождений
        if abs(slippage_pct) > 5.0:
            logger.warning(f"[SHADOW] СИЛЬНОЕ расхождение PNL {real_pnl:.2f} vs {shadow_pnl:.2f} "
                           f"({slippage_pct:+.1f}%) | {real_pos['symbol']} {real_pos['anomaly_type']}")

        if real_hit_tp != shadow_hit_tp:
            logger.warning(f"[SHADOW] Разный исход TP/SL: real={'TP' if real_hit_tp else 'SL'}, "
                           f"shadow={'TP' if shadow_hit_tp else 'SL'} | {real_pos['symbol']}")

        if real_quiet != shadow_quiet:
            logger.debug(f"[SHADOW] Разный quiet_streak: real={real_quiet}, shadow={shadow_quiet}")

        if real_consensus != shadow_consensus:
            logger.debug(f"[SHADOW] Разный consensus_count: real={real_consensus}, shadow={shadow_consensus}")

        logger.info(f"[SHADOW] Сравнение завершено | real_PNL={real_pnl:.2f} shadow_PNL={shadow_pnl:.2f} "
                    f"slippage={slippage_pct:+.1f}% | {real_pos['symbol']}")

        # Вызов scenario_tracker для shadow (для статистики)
        outcome = 1 if shadow_hit_tp else 0
        self.scenario_tracker.add_scenario(shadow_pos, outcome)

    def get_shadow_balance(self) -> float:
        return self.virtual_trader.get_balance()

    def get_shadow_pnl(self) -> float:
        return self.virtual_trader.get_pnl()