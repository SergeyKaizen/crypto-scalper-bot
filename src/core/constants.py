"""
src/core/constants.py

=== Основной принцип работы файла ===

Этот файл содержит все константы проекта в одном месте.
Он обеспечивает централизованное управление параметрами, которые используются 
во всех модулях: периоды, пороги аномалий, размеры окон, лимиты, пути и т.д.

=== Главные группы констант и за что отвечают ===

- WINDOWS_SIZES: [24, 50, 74, 100] — окна анализа из ТЗ.
- ANOMALY: lookback=25, percentile, dominance_threshold и т.д.
- MODEL: seq_len, epochs, batch_size, hidden_size
- BACKTEST: min_trades_for_pr, min_pr_threshold
- RISK: default_risk_pct, leverage_max
- PATHS: data_dir, logs_dir
- DATABASE: db_type по hardware

=== Примечания ===
- Все значения по ТЗ или здравому смыслу для интрадей скальпинга.
"""

WINDOWS_SIZES = [24, 50, 74, 100]

LOOKBACK_ANOMALY = 25
VOLUME_PERCENTILE_MIN = 0.94
VOLUME_PERCENTILE_MAX = 0.99
VOLUME_PERCENTILE_DEFAULT = 0.97
DOMINANCE_THRESHOLD_PCT = 60.0
MIN_VOLUME_FOR_ANOMALY = 1.0

VA_PERCENT = 0.70
VA_BIN_STEP_PCT = 0.001

DEFAULT_SEQ_LEN = 100
MIN_PROB_ANOMALY = 0.60
MIN_PROB_Q = 0.75
RETRAIN_INTERVAL_CANDLES = 10000
HIDDEN_SIZE_BASE = 128

MIN_TRADES_FOR_PR = 10
MIN_PR_THRESHOLD = 0.0
PR_CALC_LOOKBACK_HOURS = 720

DEFAULT_RISK_PCT = 1.0
MAX_LEVERAGE = 50
MIN_ORDER_SIZE_USDT = 5.0

DATA_DIR = "data"
LOGS_DIR = "logs"
MODELS_DIR = "models"
CONFIG_DIR = "config"

EPSILON = 1e-8
MAX_HISTORY_CANDLES = 100000
MAX_SCENARIOS_LIMIT = 10000

DEFAULT_MAX_WORKERS = 8
DEFAULT_EPOCHS = 10
DEFAULT_BATCH_SIZE = 32

# FIX Фаза 7
DEFAULT_COMMISSION = 0.0004
SLIPPAGE_PCT = 0.0005