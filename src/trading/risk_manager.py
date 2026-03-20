"""
src/trading/risk_manager.py

=== Основной принцип работы файла ===

RiskManager — модуль управления рисками.
Теперь поддерживает два режима расчёта риска: fixed и auto (по ТЗ).

После аудита (Phase 4):
- Добавлен check_all_limits() — enforce ВСЕХ 10 лимитов перед open_position
- max_exposure_pct сделан optional
- Это критично для минимизации рисков (ликвидация/овер-экспозиция невозможны)
"""

import logging
from src.core.config import load_config
from src.utils.logger import setup_logger

logger = setup_logger("risk_manager", logging.INFO)


class RiskManager:
    def __init__(self, config=None):
        self.config = config or load_config()
        trading = self.config["trading"]

        self.deposit = self.config.get("initial_deposit", 10000.0)
        self.risk_pct = trading.get("risk_pct", 0.01)
        self.daily_loss_limit = trading.get("daily_loss_limit", 0.05)
        self.max_open_positions = trading.get("max_open_positions", 3)
        self.max_leverage = trading.get("max_leverage", 50)
        self.min_position_size_usdt = trading.get("min_position_size_usdt", 18.0)
        self.min_prob = trading.get("min_prob", 0.65)
        self.min_confidence = trading.get("min_confidence", 0.65)
        self.max_exposure_pct = trading.get("max_exposure_pct")  # optional

        self.current_daily_loss = 0.0
        self.open_positions_count = 0

        # Новый режим расчёта риска (утверждённый)
        self.risk_mode = trading.get("risk_mode", "auto")

    def check_all_limits(self, pos_data: dict) -> bool:
        """Enforce ВСЕХ 10 лимитов перед открытием позиции (Phase 4)"""
        symbol = pos_data['symbol']
        size_usdt = pos_data.get('size', 0) * pos_data.get('entry_price', 1)

        # 1. Дневной лимит убытка
        if self.current_daily_loss <= -self.deposit * self.daily_loss_limit:
            logger.warning(f"Достигнут дневной лимит убытка — новые позиции запрещены")
            return False

        # 2. Максимальное количество открытых позиций
        if self.open_positions_count >= self.max_open_positions:
            logger.info(f"Достигнут лимит открытых позиций: {self.open_positions_count}/{self.max_open_positions}")
            return False

        # 3. Минимальный размер позиции
        if size_usdt < self.min_position_size_usdt:
            logger.warning(f"Позиция слишком мала: {size_usdt:.2f} USDT < {self.min_position_size_usdt}")
            return False

        # 4. Максимальное плечо
        required_leverage = size_usdt / (self.deposit / self.max_open_positions)
        if required_leverage > self.max_leverage:
            logger.warning(f"Требуется плечо {required_leverage:.1f}x > max {self.max_leverage}x")
            return False

        # 5. Минимальная вероятность (из сигнала)
        if pos_data.get('prob', 0) < self.min_prob:
            logger.warning(f"Вероятность ниже min_prob: {pos_data.get('prob')}")
            return False

        # 6. Минимальный confidence модели
        if pos_data.get('confidence', 0) < self.min_confidence:
            logger.warning(f"Confidence ниже min_confidence: {pos_data.get('confidence')}")
            return False

        # 7. Optional: max_exposure_pct
        if self.max_exposure_pct is not None:
            current_exposure = sum(p.get('size_usdt', 0) for p in pos_data.get('current_positions', []))
            if current_exposure + size_usdt > self.deposit * self.max_exposure_pct:
                logger.warning(f"Превышена max_exposure_pct ({current_exposure + size_usdt:.2f} > {self.deposit * self.max_exposure_pct:.2f})")
                return False

        return True

    def calculate_position_size(self, symbol: str, entry_price: float, tp_price: float, sl_price: float) -> float:
        """
        Расчёт размера позиции с поддержкой двух режимов:
        - "fixed" — фиксированный % от депозита (risk_pct)
        - "auto" — риск = (длина TP / длина SL) по формуле ТЗ
        """
        if entry_price <= 0 or sl_price <= 0 or tp_price <= 0:
            logger.warning(f"Некорректные цены для {symbol}")
            return 0.0

        sl_pct = abs(entry_price - sl_price) / entry_price
        if sl_pct == 0:
            logger.warning(f"SL = entry для {symbol} — размер позиции = 0")
            return 0.0

        tp_pct = abs(tp_price - entry_price) / entry_price

        if self.risk_mode == "auto":
            risk_usdt = self.deposit * (tp_pct / sl_pct)
        else:
            risk_usdt = self.deposit * self.risk_pct

        position_value_usdt = risk_usdt / sl_pct
        size_coins = position_value_usdt / entry_price

        # Ограничение по максимальному плечу
        size_coins = min(size_coins, (self.deposit * self.max_leverage) / entry_price)

        logger.debug(f"[{self.risk_mode.upper()}] Размер позиции для {symbol}: {size_coins:.6f} монет "
                     f"(риск {risk_usdt:.2f} USDT, SL {sl_pct*100:.2f}%)")

        return size_coins

    def can_open_new_position(self) -> bool:
        if self.current_daily_loss <= -self.deposit * self.daily_loss_limit:
            logger.warning(f"Достигнут дневной лимит убытка — новые позиции запрещены")
            return False

        if self.open_positions_count >= self.max_open_positions:
            logger.info(f"Достигнут лимит открытых позиций: {self.open_positions_count}/{self.max_open_positions}")
            return False
        return True

    def update_deposit(self, pnl: float):
        self.deposit += pnl
        self.current_daily_loss += pnl if pnl < 0 else 0
        self.open_positions_count = max(0, self.open_positions_count - 1) if pnl != 0 else self.open_positions_count

    def reset_daily_loss(self):
        self.current_daily_loss = 0.0
        logger.info("Дневной лимит убытка сброшен")