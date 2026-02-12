# src/init.py
"""
Crypto Scalper Bot
==================

Версия: 1.0.0 (2026)
Автор: Grok (xAI) — специализированный разработчик скальпинг-ботов на крипте
Лицензия: Private (для личного использования / подписки)

Описание:
---------
Внутридневной скальпер-бот на фьючерсах Binance (USDT perpetual).
Гибридная нейронка Conv1D + GRU, live-переобучение, динамический PR-фильтр монет,
мульти-TF (1m–15m), окна 24–100 свечей, аномалии (C/V/CV) + тихий режим (Q),
мягкие входы, trailing, shadow trading.

Поддержка 3 платформ:
- phone_tiny (Redmi Note 12 Pro): 5 монет, 3 TF, 3 окна
- colab: 30–50 монет, full TF
- server: 100+ монет, parallel, RTX 3090

Запуск:
    python scripts/run_bot.py --hardware phone_tiny --mode balanced --trading virtual

Документация: docs/ или PROJECT_STRUCTURE.md
"""

# Метаданные (доступны через import crypto_scalper_bot; crypto_scalper_bot.__version__)
__version__ = "1.0.0"
__author__ = "Grok (xAI)"
__description__ = "Adaptive intraday scalping bot for Binance futures with live-learning neural net"

# Экспорт ключевых компонентов (чтобы удобно импортировать)
# Пример: from crypto_scalper_bot import load_config, Signal, AnomalyType
from .core.config import load_config
from .core.enums import AnomalyType, Direction, TpMode, SlMode, TrailingType, RiskBase
from .core.types import Candle, Signal, Position

# Инициализация (выполняется при импорте модуля)
# Можно добавить глобальный логгер или проверку окружения
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

logger = logging.getLogger(__name__)
logger.info("Crypto Scalper Bot module initialized (version %s)", __version__)

# Опционально: авто-определение окружения (если не задан --hardware)
# from .utils.helpers import detect_hardware
# if not hasattr(sys, 'argv') or '--hardware' not in sys.argv:
#     logger.warning("Hardware profile not specified, auto-detecting: %s", detect_hardware())