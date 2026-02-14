# src/utils/logger.py
"""
Централизованное логирование для всего проекта.
Используем structlog + logging — структурированные логи в JSON для продакшена,
читаемые в консоли для разработки.

Уровни:
- DEBUG   — детальная отладка (аномалии, каждый предикт, подписки WS)
- INFO    — важные события (открытие/закрытие позиции, PR-обновление)
- WARNING — потенциальные проблемы (много аномалий, тайм-аут, низкая уверенность)
- ERROR   — критические сбои

В проде включаем INFO/WARNING, в разработке — DEBUG.
"""

import logging
import sys
import structlog

def configure_logging(level: str = "DEBUG"):
    """Вызывается один раз при старте приложения."""
    shared_processors = [
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
    ]

    if level.upper() == "DEBUG":
        # Для разработки — красивый цветной вывод в консоль
        processors = shared_processors + [structlog.dev.ConsoleRenderer()]
    else:
        # Для продакшена — JSON для парсинга (ELK, Grafana Loki и т.д.)
        processors = shared_processors + [structlog.processors.JSONRenderer()]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, level.upper())),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Базовая настройка Python logging (чтобы работали обычные logger)
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper()),
    )

    # Глобальный логгер проекта
    return structlog.get_logger("crypto_scalper")

# Инициализация при импорте модуля (можно вызвать явно в main)
logger = configure_logging("DEBUG")  # или "INFO" в проде