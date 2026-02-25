"""
src/trading/risk_manager.py

=== Основной принцип работы файла ===

Менеджер расчёта размера позиции и управления рисками.

Ключевые особенности (по ТЗ + улучшения):
- calculate_size: размер = (balance * risk_pct / 100) / sl_distance * leverage
- min_notional check (минимальная стоимость позиции)
- min_lot check (минимальный объём по монете — через fetch_markets, если доступно)
- update_deposit(net_pl) — обновление баланса после закрытия (с комиссией)
- Синхронизация с реальным балансом Binance (если расхождение >0.1)
- Передача quiet_streak и consensus_count в логи (для анализа)
- Интеграция с PositionManager: вызовы через него (улучшение №5)

=== Главные функции ===
- calculate_size(symbol, entry_price, sl_price, risk_pct, quiet_streak=0, consensus_count=1) → size
- update_deposit(net_pl) — обновление баланса
- get_balance() — текущий баланс (реальный или симулированный)

=== Примечания ===
- Формула соответствует ТЗ: риск = % от депозита / расстояние до SL
- Leverage из config (по умолчанию 20)
- Комиссия учитывается в net_pl
- Полностью соответствует ТЗ + улучшениям (централизация, quiet/consensus в логах)
- Готов к интеграции в PositionManager, entry_manager, live_loop
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
        self.min_notional = self.config.get('min_notional', {}).get('default', 5.0)

        # Кэш min_lot per-symbol (загружаем при первом вызове)
        self.min_lot_cache = {}

    def calculate_size(self, symbol: str, entry_price: float, sl_price: float, risk_pct: float,
                       quiet_streak: int = 0, consensus_count: int = 1) -> float:
        """
        Расчёт размера позиции по ТЗ + проверка min_notional и min_lot
        """
        if sl_price == entry_price or np.isnan(sl_price):
            logger.warning(f"SL = entry или NaN для {symbol} — размер позиции = 0")
            return 0.0

        sl_distance = abs(entry_price - sl_price) / entry_price
        if sl_distance == 0 or np.isnan(sl_distance):
            return 0.0

        risk_amount = self.balance * (risk_pct / 100)
        size = risk_amount / (sl_distance * entry_price) * self.leverage

        # Min notional check
        if size * entry_price < self.min_notional:
            logger.warning(f"Размер позиции {size:.4f} ниже min_notional {self.min_notional} для {symbol}")
            return 0.0

        # Min lot check (минимальный объём по монете)
        min_lot = self._get_min_lot(symbol)
        if size < min_lot:
            logger.warning(f"Размер позиции {size:.4f} ниже min_lot {min_lot} для {symbol}")
            return 0.0

        logger.debug(f"Расчёт размера для {symbol}: risk={risk_amount:.2f}, distance={sl_distance:.4f}, "
                     f"size={size:.4f}, quiet_streak={quiet_streak}, consensus={consensus_count}")
        return size

    def update_deposit(self, net_pl: float):
        """
        Обновление баланса после закрытия позиции (с комиссией)
        """
        self.balance += net_pl
        logger.info(f"Обновлён баланс: {self.balance:.2f} (net_pl={net_pl:.2f})")

        # Синхрон с реальным балансом Binance
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

    def _get_min_lot(self, symbol: str) -> float:
        """Получение минимального объёма (lot size) для монеты"""
        if symbol in self.min_lot_cache:
            return self.min_lot_cache[symbol]

        try:
            markets = self.exchange.load_markets()
            if symbol in markets:
                min_amount = markets[symbol]['limits']['amount']['min']
                self.min_lot_cache[symbol] = min_amount or 0.0
                return self.min_lot_cache[symbol]
        except Exception as e:
            logger.warning(f"Не удалось загрузить min_lot для {symbol}: {e}")

        return 0.0  # fallback — нет ограничения

    def get_balance(self) -> float:
        """Текущий баланс (локальный или реальный)"""
        return self.balance