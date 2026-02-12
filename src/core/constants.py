# src/core/constants.py
"""
Глобальные константы проекта.
Все значения, которые не меняются в runtime и используются в разных модулях.
Не захардкоживаем в коде — только здесь.

Разделы:
- Binance API limits & fees
- Минимальные размеры ордеров
- Дефолтные значения (если конфиг не задан)
- Дополнительные лимиты и константы
"""

from decimal import Decimal

# Binance Futures — комиссии и лимиты (2026 актуальные значения)
BINANCE_FUTURES_TAKER_FEE = Decimal('0.0004')      # 0.04% taker fee (USDT-M)
BINANCE_FUTURES_MAKER_FEE = Decimal('0.0002')      # 0.02% maker fee
BINANCE_FUTURES_MAX_LEVERAGE = 125                 # Максимальное плечо (зависит от монеты)
BINANCE_FUTURES_MIN_ORDER_VALUE_USDT = Decimal('5')  # Минимальный номинал ордера в USDT
BINANCE_API_RATE_LIMIT_PER_MIN = 1200              # Запросов в минуту (примерно)
BINANCE_KLINES_LIMIT_PER_REQUEST = 1500            # Максимум свечей за один запрос

# Минимальные размеры ордеров для популярных монет (USDT perpetual)
# Можно обновлять через API (exchange.fetch_markets())
MIN_ORDER_SIZES = {
    'BTCUSDT': Decimal('0.001'),    # 0.001 BTC
    'ETHUSDT': Decimal('0.01'),     # 0.01 ETH
    'SOLUSDT': Decimal('0.1'),      # 0.1 SOL
    'XRPUSDT': Decimal('10'),       # 10 XRP
    # Добавляй по необходимости
}

# Дефолтные значения (если конфиг не задан или повреждён)
DEFAULT_CONFIG_VALUES = {
    'seq_len': 50,                      # Безопасное значение для телефона
    'max_coins': 5,                     # Минимум для запуска
    'prob_threshold': 0.70,             # Безопасный порог
    'default_risk_pct': 1.0,            # 1% — стандарт
    'pr_analysis_period_candles': 250,  # Базовый период PR
    'min_deals_in_pr_period': 5,        # Минимум сделок
}

# Константы для PR (Profitable Rating)
PR_DECAY_THRESHOLD_CANDLES = 150          # Сделки старше 150 свечей — вес ×0.8
PR_NORMALIZED_DIVISOR = 2.0               # log10(n+5) / 2.0
PR_MIN_DEALS_HARD_LIMIT = 3               # Абсолютный минимум (если <3 — PR=0)

# Константы для trailing stop
TRAILING_ACTIVATION_MIN_PCT = Decimal('0.3')   # Минимум для активации trailing
TRAILING_DISTANCE_MIN_PCT = Decimal('0.5')     # Минимальное расстояние

# Константы для soft entry
SOFT_ENTRY_MIN_LEVELS = 2                 # Минимум уровней для soft entry
SOFT_ENTRY_SUM_SIZES = 1.0                # Сумма soft_sizes должна быть 1.0

# Дополнительные лимиты (защита от ошибок)
MAX_RISK_PCT_HARD_LIMIT = 3.0             # Никогда не рисковать >5% на сделку
MAX_OPEN_POSITIONS_HARD_LIMIT = 30        # Максимум позиций (даже в aggressive)
MAX_LEVERAGE_HARD_LIMIT = 50              # Не выше 50x (даже если Binance позволяет 125x)

# Время (в секундах)
LIVE_LOOP_SLEEP_SECONDS = 1               # Основной цикл — проверка каждую секунду
QUIET_MODE_MIN_INTERVAL_SECONDS = 60      # Минимальный интервал тихого режима (1 минута)
PR_RECALC_MIN_INTERVAL_SECONDS = 120      # Минимум 2 минуты между пересчётами PR

# Логи
LOG_FILE_MAX_MB = 500                     # Максимум 500 МБ на лог-файл
LOG_BACKUP_COUNT = 10                      # Хранить 10 последних логов