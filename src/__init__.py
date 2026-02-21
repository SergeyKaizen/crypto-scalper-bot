"""
src/__init__.py

=== Основной принцип работы файла ===

Этот файл делает директорию `src` полноценным Python-пакетом.
Он позволяет импортировать модули удобным способом:
from src import BinanceClient, LiveLoop, Trainer и т.д.

Также содержит:
- __version__ проекта (для отслеживания версий).
- Глобальные импорты наиболее часто используемых классов (чтобы не писать длинные пути).
- Автоматическую инициализацию пакета при первом импорте (вызов init()).
- Базовую настройку логирования на уровне пакета.

=== Главные элементы и за что отвечают ===

- __version__ = "0.1.0" — текущая версия проекта.
- Импорты ключевых классов — упрощают использование (например, from src import LiveLoop вместо длинного пути).
- init() — функция, которая вызывается автоматически при импорте пакета src.
- logging.basicConfig — глобальная настройка логов (если не настроена в модулях).

=== Примечания ===
- Файл минималистичный — не содержит бизнес-логики.
- Позволяет быстро импортировать основные компоненты бота.
- Полностью соответствует ТЗ: удобство структуры проекта.
- Готов к использованию.
"""

# =============================================
# Версия проекта
# =============================================
__version__ = "0.1.0"

# =============================================
# Глобальные импорты для удобства использования
# =============================================

# Data layer
from .data.binance_client import BinanceClient
from .data.downloader import download_full_history, download_new_candles
from .data.storage import Storage

# Features
from .features.feature_engine import prepare_sequence_features
from .features.anomaly_detector import detect_anomalies
from .features.channels import anomalous_surge_channel_feature, calculate_value_area

# Model
from .model.architectures import MultiWindowHybrid
from .model.trainer import Trainer
from .model.inference import Inference
from .model.scenario_tracker import ScenarioTracker

# Backtest
from .backtest.engine import BacktestEngine
from .backtest.pr_calculator import PRCalculator

# Trading
from .trading.live_loop import LiveLoop
from .trading.entry_manager import EntryManager
from .trading.tp_sl_manager import TPSLManager
from .trading.virtual_trader import VirtualTrader
from .trading.order_executor import OrderExecutor
from .trading.risk_manager import RiskManager
from .trading.websocket_manager import WebsocketManager

# Utils
from .utils.logger import setup_logger
from .utils.helpers import safe_div, timestamp_to_datetime, format_number

# =============================================
# Инициализация пакета при импорте
# =============================================

def init():
    """Автоматическая инициализация пакета при первом импорте."""
    logger = setup_logger('src_init')
    logger.info(f"Пакет src успешно инициализирован (версия {__version__})")


# Выполняем инициализацию при импорте пакета
init()