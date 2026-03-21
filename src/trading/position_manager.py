"""
src/trading/position_manager.py

=== Основной принцип работы файла ===

Централизованный менеджер позиций — **SINGLE SOURCE OF TRUTH** (self.positions).

Ключевые изменения после аудита (Phase 4):
- Все open/close позиции идут только через этот класс
- Перед открытием — enforce ВСЕХ risk-лимитов
- Поддержка hybrid-режима (backtest + live одновременно)
- Использует total_pr вместо pnl в ScenarioTracker
- Глобальный lock на символ (одна позиция на монету)
"""

from enum import Enum
from typing import Dict, Optional
import time
import logging
import asyncio

from src.core.config import load_config
from src.trading.risk_manager import RiskManager
from src.trading.order_executor import OrderExecutor
from src.trading.virtual_trader import VirtualTrader
from src.trading.tp_sl_manager import TP_SL_Manager
from src.model.scenario_tracker import ScenarioTracker
from src.utils.logger import setup_logger

# Telegram
from telegram import InlineKeyboardButton, InlineKeyboardMarkup

# Для реальной генерации графика
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from src.data.storage import Storage

logger = setup_logger('position_manager', logging.INFO)

class PositionState(Enum):
    OPEN = "OPEN"
    PENDING_CLOSE = "PENDING_CLOSE"
    CLOSED = "CLOSED"

class PositionManager:
    def __init__(self):
        self.config = load_config()
        self.risk_manager = RiskManager()
        self.order_executor = OrderExecutor()
        self.virtual_trader = VirtualTrader()
        self.tp_sl_manager = TP_SL_Manager()
        self.scenario_tracker = ScenarioTracker()
        self.storage = Storage()

        # === SINGLE SOURCE OF TRUTH ===
        self.positions = {}  # pos_id → {'data': dict, 'state': PositionState, ...}

    async def send_telegram_alert(self, log_text: str, symbol: str):
        """Отправляет подробный лог + кнопку 'Показать график' + реальный график"""
        if not self.config.get('monitoring', {}).get('enable_telegram', False):
            return

        token = self.config['monitoring'].get('telegram_token')
        chat_id = self.config['monitoring'].get('telegram_chat_id')

        if not token or not chat_id:
            return

        keyboard = [[InlineKeyboardButton("📊 Показать график", callback_data=f"graph_{symbol}")]]
        reply_markup = InlineKeyboardMarkup(keyboard)

        try:
            from telegram.ext import Application
            app = Application.builder().token(token).build()
            await app.bot.send_message(
                chat_id=chat_id,
                text=log_text,
                reply_markup=reply_markup,
                parse_mode='HTML'
            )
        except Exception as e:
            logger.error(f"Ошибка отправки Telegram: {e}")

    def open_position(self, pos_data: Dict, is_backtest_signal: bool = False):
        """
        Главный метод открытия позиции.
        """
        pos_id = pos_data['pos_id']
        symbol = pos_data['symbol']
        direction = pos_data['direction']
        entry_price = pos_data['entry_price']
        tp = pos_data.get('tp')
        sl = pos_data.get('sl')
        marking = pos_data.get('marking')

        # 1. Проверка risk-лимитов
        risk_ok = self.risk_manager.check_all_limits(pos_data)
        if not risk_ok:
            logger.warning(f"Risk limits failed for {symbol}")
            return False

        # 2. Глобальный lock
        if self.has_any_open_position(symbol):
            logger.warning(f"Глобальный lock: уже есть открытая позиция на {symbol}")
            return False

        # Расчёт размера
        size = pos_data.get('size', 0)
        if size <= 0 and tp and sl:
            size = self.risk_manager.calculate_position_size(symbol, entry_price, tp, sl)
            pos_data['size'] = size

        if size <= 0:
            logger.error(f"Некорректный размер позиции для {symbol}")
            return False

        # 3. Исполнение
        if pos_data.get('mode') == 'real':
            order_id = self.order_executor.place_order(pos_data)
            if not order_id:
                return False
            pos_data['order_id'] = order_id
        else:
            self.virtual_trader.open_position(pos_data)

        # 4. Сохранение
        self.positions[pos_id] = {
            'data': pos_data,
            'state': PositionState.OPEN,
            'open_time': time.time(),
            'marking': marking,
            'is_backtest_signal': is_backtest_signal
        }

        self.tp_sl_manager.add_open_position(pos_data)

        # ====================== НОВЫЙ ПОДРОБНЫЙ ЛОГ ======================
        feats = pos_data.get('feats', {})
        scenario_key = self.scenario_tracker._binarize_features(feats)
        scenario_weight = self.scenario_tracker.get_weight(scenario_key)

        log_msg = f"""
🚀 ПОЗИЦИЯ ОТКРЫТА
   • Монета: {symbol} | Направление: {direction}
   • Цена входа: {entry_price:.4f} | Размер: {size:.6f}

   === РАСЧЁТ ОБЪЁМА ===
   • Режим: {self.risk_manager.risk_mode}
   • Риск на сделку: {self.risk_manager.risk_pct if self.risk_manager.risk_mode == 'fixed' else '(auto по RR)'} 
   • SL%: {abs(entry_price - sl)/entry_price*100:.2f}% | TP%: {abs(tp - entry_price)/entry_price*100:.2f}%

   === СОСТОЯНИЕ ПРИЗНАКОВ ===
   • Volume change: {feats.get('volume_change_pct', 0):+.2f}% {'↑' if feats.get('volume_increased', 0) else '↓'}
   • Delta change: {feats.get('delta_change_pct', 0):+.2f}% 
   • Price change diff: {feats.get('price_change_diff_pct', 0):+.2f}%
   • VA position: {feats.get('va_position', 0):.3f}
   • ASC position: {feats.get('asc_position_in_channel', 0):.3f}
   • Regime: Bull {feats.get('regime_bull_strength', 0)} | Bear {feats.get('regime_bear_strength', 0)}

   === СЦЕНАРИЙ И ПАТТЕРН ===
   • Сценарий: {scenario_key}
   • Винрейт сценария: {scenario_weight:.4f}
   • Ожидаемое движение: TP {pos_data.get('tp_length', 0):.2f}% | SL {pos_data.get('sl_length', 0):.2f}% | RR {pos_data.get('tp_length', 1)/pos_data.get('sl_length', 1):.2f}
"""
        logger.info(log_msg.strip())

        # === Telegram + кнопка "Показать график" ===
        if self.config.get('monitoring', {}).get('enable_telegram', False):
            asyncio.create_task(self.send_telegram_alert(log_msg.strip(), symbol))

        logger.info(f"Позиция открыта: {pos_id} | {direction} {symbol} | size={size:.4f} | marking={marking}")
        return True

    # Остальные методы без изменений
    def check_and_close(self, current_price: float, current_time: float):
        for pos_id, pos_info in list(self.positions.items()):
            if pos_info['state'] != PositionState.OPEN:
                continue

            pos_data = pos_info['data']
            symbol = pos_data['symbol']
            direction = pos_data['direction']
            tp = pos_data.get('tp')
            sl = pos_data.get('sl')

            hit_tp = (current_price >= tp) if direction == 'L' else (current_price <= tp)
            hit_sl = (current_price <= sl) if direction == 'L' else (current_price >= sl)

            if hit_tp or hit_sl:
                if pos_data.get('mode') == 'real':
                    self.order_executor.close_position(pos_id)
                else:
                    self.virtual_trader.close_position(pos_id)

                net_pnl = self._calculate_net_pnl(pos_data, current_price, hit_tp)
                self.risk_manager.update_deposit(net_pnl)

                outcome = 1 if hit_tp else 0
                pr_value = pos_data.get('tp_length', 0) if hit_tp else -pos_data.get('sl_length', 0)
                self.scenario_tracker.add_scenario(pos_data.get('feats', {}), outcome, pr_value)

                pos_info['state'] = PositionState.CLOSED
                pos_info['close_time'] = current_time
                pos_info['net_pnl'] = net_pnl

                logger.info(f"Позиция закрыта: {pos_id} | {'TP' if hit_tp else 'SL'} | PnL={net_pnl:.2f}")

    def has_any_open_position(self, symbol: str) -> bool:
        for pos_info in self.positions.values():
            if pos_info['state'] == PositionState.OPEN and pos_info['data']['symbol'] == symbol:
                return True
        return False

    def get_position_state(self, pos_id: str) -> PositionState:
        return self.positions.get(pos_id, {}).get('state', PositionState.CLOSED)

    def _calculate_net_pnl(self, pos_data: Dict, exit_price: float, hit_tp: bool) -> float:
        entry_price = pos_data['entry_price']
        size = pos_data['size']
        direction = pos_data['direction']
        commission_rate = self.config['trading']['commission']

        gross_pnl = (exit_price - entry_price) * size if direction == 'L' else (entry_price - exit_price) * size
        entry_comm = size * entry_price * commission_rate
        exit_comm = size * exit_price * commission_rate
        net_pnl = gross_pnl - entry_comm - exit_comm
        return net_pnl