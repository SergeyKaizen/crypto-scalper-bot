# src/trading/risk_manager.py
"""
Risk Manager — расчёт размера позиции, TP/SL, контроль риска.
Полностью соответствует ТЗ + улучшения:
1. Реальный запрос к Binance /fapi/v1/exchangeInfo (кэшируется)
2. Учёт комиссии taker 0.04% (вход + выход = 0.08%)
"""

from typing import Optional, Dict, Any
import time

import ccxt
import polars as pl

from ..utils.logger import logger
from ..core.config import get_config
from ..core.types import AnomalySignal, Direction, Position


class RiskManager:
    def __init__(self):
        self.config = get_config()
        self.ccxt_client = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'future'},
            # Ключи добавишь позже — пока публичный режим
            # 'apiKey': self.config['binance']['api_key'],
            # 'secret': self.config['binance']['secret_key'],
        })

        # Текущий депозит (обновляется после каждой позиции)
        self.current_balance = self.config["trading"]["initial_balance"]  # USDT
        self.leverage = self.config["trading"]["leverage"]
        self.risk_mode = self.config["trading"]["risk_mode"]
        self.manual_risk_pct = self.config["trading"].get("manual_risk_pct", 1.0)

        # Кэш exchangeInfo (обновляется раз в час или при ошибке)
        self.coin_info: Dict[str, Dict[str, Any]] = {}
        self.last_refresh = 0
        self._refresh_coin_info()

        # Комиссия taker (Binance Futures USDT-M perpetual, 2026)
        self.taker_fee_pct = 0.0004  # 0.04%

        logger.info("RiskManager готов",
                    balance=self.current_balance,
                    leverage=self.leverage,
                    risk_mode=self.risk_mode,
                    taker_fee=f"{self.taker_fee_pct*100:.3f}%")

    def _refresh_coin_info(self):
        """Загружает /fapi/v1/exchangeInfo и кэширует данные о монетах."""
        now = time.time()
        if now - self.last_refresh < 3600:  # 1 час
            return

        try:
            info = self.ccxt_client.fapiPublicGetExchangeInfo()
            for symbol_info in info['symbols']:
                symbol = symbol_info['symbol']
                if symbol.endswith('USDT'):
                    coin = symbol[:-4]
                    filters = {f['filterType']: f for f in symbol_info['filters']}

                    self.coin_info[coin] = {
                        'max_leverage': int(symbol_info.get('leverageFilter', {}).get('maxLeverage', 125)),
                        'min_notional': float(filters.get('NOTIONAL', {}).get('minNotional', 5.0)),
                        'price_precision': int(float(filters.get('PRICE_FILTER', {}).get('tickSize', '0.01')) ** -1),
                        'quantity_precision': int(float(filters.get('LOT_SIZE', {}).get('stepSize', '0.001')) ** -1),
                    }
            self.last_refresh = now
            logger.info("Кэш монет обновлён", coins=len(self.coin_info))
        except Exception as e:
            logger.error("Ошибка exchangeInfo", error=str(e))
            # Запасные значения для основных монет
            for coin in ["BTC", "ETH", "SOL", "XRP", "ADA"]:
                self.coin_info.setdefault(coin, {
                    'max_leverage': 125,
                    'min_notional': 5.0,
                    'price_precision': 2,
                    'quantity_precision': 3,
                })

    def calculate_position(self, signal: AnomalySignal, pred, current_df: pl.DataFrame) -> Optional[Position]:
        """Рассчитывает позицию с реальными параметрами монеты и комиссией."""
        coin = signal.coin
        if not coin or coin not in self.coin_info:
            logger.error("Нет данных о монете", coin=coin)
            return None

        last_candle = current_df.tail(1)
        entry_price = last_candle["close"][0]
        timestamp = last_candle["timestamp"][0]

        is_long = signal.direction_hint == Direction.LONG

        # Средний размер свечи (100 баров)
        period = 100
        if len(current_df) < period:
            logger.warning("Мало данных для среднего размера свечи", coin=coin)
            return None

        sizes = ((current_df["high"] - current_df["low"]) / current_df["close"] * 100).tail(period)
        avg_candle_size_pct = sizes.mean()

        # TP = средний размер свечи
        tp_distance_pct = avg_candle_size_pct
        tp_price = entry_price * (1 + tp_distance_pct / 100) if is_long else \
                   entry_price * (1 - tp_distance_pct / 100)

        # SL = HH/LL + 0.05% с ограничением 2×средний размер
        lookback_sl = period
        if is_long:
            ll = current_df["low"].tail(lookback_sl).min()
            sl_price = ll * (1 - 0.0005)
        else:
            hh = current_df["high"].tail(lookback_sl).max()
            sl_price = hh * (1 + 0.0005)

        distance_to_sl = abs(entry_price - sl_price) / entry_price * 100
        max_sl_distance = 2 * avg_candle_size_pct
        if distance_to_sl > max_sl_distance:
            logger.debug("SL скорректирован", coin=coin, old=distance_to_sl, max=max_sl_distance)
            if is_long:
                sl_price = entry_price * (1 - max_sl_distance / 100)
            else:
                sl_price = entry_price * (1 + max_sl_distance / 100)

        # Риск на сделку
        risk_pct = self._get_risk_pct()
        risk_amount = self.current_balance * (risk_pct / 100)

        sl_distance_pct = abs(entry_price - sl_price) / entry_price * 100
        position_value = risk_amount / (sl_distance_pct / 100)

        # Плечо
        coin_info = self.coin_info[coin]
        max_leverage = coin_info['max_leverage']
        effective_leverage = min(self.leverage, max_leverage)

        position_size_usdt = position_value * effective_leverage

        # Проверка минимального ордера
        min_notional = coin_info['min_notional']
        if position_size_usdt < min_notional:
            logger.warning("Позиция меньше min_notional",
                           coin=coin,
                           calculated=position_size_usdt,
                           min_notional=min_notional)
            return None

        # Проверка маржи
        required_margin = position_size_usdt / effective_leverage
        if required_margin > self.current_balance:
            logger.warning("Недостаточно маржи",
                           required=required_margin,
                           available=self.current_balance)
            return None

        position = Position(
            coin=coin,
            side=signal.direction_hint,
            entry_price=entry_price,
            size=position_size_usdt / entry_price,  # кол-во контрактов
            entry_time=timestamp,
            tp_price=tp_price,
            sl_price=sl_price,
            anomaly_signal=signal,
            is_real=False
        )

        logger.info("Позиция рассчитана",
                    coin=coin,
                    side=signal.direction_hint.value,
                    entry_price=entry_price,
                    size_usdt=position_size_usdt,
                    leverage=effective_leverage,
                    risk_pct=risk_pct,
                    tp_price=tp_price,
                    sl_price=sl_price,
                    commission_pct=self.taker_fee_pct*100)

        return position

    def _get_risk_pct(self) -> float:
        mode = self.risk_mode
        if mode == "auto":
            return 1.0
        elif mode == "low":
            return 0.5
        elif mode == "medium":
            return 1.0
        elif mode == "high":
            return 2.0
        elif mode == "manual":
            return self.manual_risk_pct
        return 1.0

    def update_balance_after_close(self, gross_pnl: float):
        """Обновляет баланс с учётом комиссии (вход + выход)."""
        commission = abs(gross_pnl) * self.taker_fee_pct * 2  # 0.08% от суммы
        net_pnl = gross_pnl - commission

        self.current_balance += net_pnl
        logger.info("Баланс обновлён",
                    gross_pnl=gross_pnl,
                    commission=commission,
                    net_pnl=net_pnl,
                    new_balance=self.current_balance)

    def get_current_balance(self) -> float:
        return self.current_balance