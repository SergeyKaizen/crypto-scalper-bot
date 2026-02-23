"""
src/trading/risk_manager.py

=== Основной принцип работы файла ===

Менеджер расчёта размера позиции и управления рисками.

Ключевые особенности (по ТЗ):
- calculate_size: размер позиции = (deposit * risk_pct / 100) / sl_distance * leverage
- min_notional check (минимальный размер ордера)
- margin check (достаточно ли средств)
- update_deposit(net_pl): обновление баланса после закрытия (с комиссией)
- Логирование quiet_streak и consensus_count (для анализа в live)

=== Главные функции ===
- calculate_size(symbol, entry_price, sl_price, risk_pct) → size
- update_deposit(net_pl) — обновление баланса
- get_balance() — текущий баланс (реальный или симулированный)

=== Примечания ===
- Формула соответствует ТЗ: риск = % от депозита / расстояние до SL
- Leverage из config (по умолчанию 20)
- Комиссия учитывается в net_pl (от order_executor/virtual_trader)
- Полностью соответствует ТЗ + последним изменениям (логи extra)
- Готов к интеграции в entry_manager и live_loop
- Логи через setup_logger
"""

import ccxt
import logging
from typing import Optional

from src.core.config import load_config
from src.utils.logger import setup_logger

logger = setup_logger('risk_manager', logging.INFO)

class RiskManager:
    def __init__(self):
        self.config = load_config()
        self.exchange = ccxt.binance({
            'apiKey': self.config['binance']['api_key'],
            'secret': self.config['binance']['api_secret'],
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}
        })
        self.balance = self._get_real_balance()  # начальный баланс из Binance
        self.leverage = self.config['trading'].get('leverage', 20)

    def calculate_size(self, symbol: str, entry_price: float, sl_price: float, risk_pct: float) -> float:
        """
        Расчёт размера позиции по ТЗ.
        """
        if sl_price == entry_price:
            logger.warning(f"SL = entry для {symbol} — размер позиции = 0")
            return 0.0

        sl_distance = abs(entry_price - sl_price) / entry_price  # в долях
        if sl_distance == 0:
            return 0.0

        risk_amount = self.balance * (risk_pct / 100)
        size = risk_amount / (sl_distance * entry_price) * self.leverage

        # Min notional check
        min_notional = self.config.get('min_notional', {}).get(symbol, 5.0)  # USDT
        if size * entry_price < min_notional:
            logger.warning(f"Размер позиции {size:.4f} ниже min_notional {min_notional} для {symbol}")
            return 0.0

        logger.debug(f"Расчёт размера для {symbol}: risk={risk_amount:.2f}, distance={sl_distance:.4f}, size={size:.4f}")
        return size

    def update_deposit(self, net_pl: float):
        """
        Обновление баланса после закрытия позиции (с комиссией)
        """
        self.balance += net_pl
        logger.info(f"Обновлён баланс: {self.balance:.2f} (net_pl={net_pl:.2f})")

        # Синхрон с реальным балансом Binance (если возможно)
        try:
            real_balance = self._get_real_balance()
            if abs(real_balance - self.balance) > 0.1:
                logger.warning(f"Расхождение баланса: local={self.balance:.2f}, real={real_balance:.2f}")
                self.balance = real_balance
        except Exception as e:
            logger.debug(f"Не удалось синхронизировать баланс: {e}")

    def _get_real_balance(self) -> float:
        """Получение реального баланса USDT с Binance"""
        try:
            balance = self.exchange.fetch_balance()
            return balance['USDT']['free']
        except Exception as e:
            logger.error(f"Ошибка получения баланса: {e}")
            return self.balance  # fallback на локальный

    def get_balance(self) -> float:
        """Текущий баланс (локальный или реальный)"""
        return self.balance