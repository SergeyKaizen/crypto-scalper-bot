# src/utils/logger.py
"""
Централизованное логирование проекта.

Настройки:
- Формат: 2026-02-07 14:35:22 [INFO] module_name: Сообщение
- Уровни: DEBUG / INFO / WARNING / ERROR
- Вывод: в консоль + файл (logs/bot.log)
- Ротация: max 50 МБ, хранит 5 последних файлов
- На телефоне: уровень INFO (экономия батареи и места)
- На сервере: уровень DEBUG (полная отладка)

Использование:
    logger = logging.getLogger(__name__)
    logger.info("Сообщение")
    logger.debug("Отладка")
    logger.error("Ошибка: %s", exc)

Все модули проекта используют этот логгер — единый стиль и файл.
"""

import logging
import logging.handlers
from pathlib import Path

from src.core.config import load_config

def setup_logger():
    """Инициализация глобального логгера"""
    config = load_config()  # Загружаем конфиг для определения уровня и путей

    # Базовые настройки
    log_level = config["logging"].get("level", "INFO").upper()
    log_file = config["logging"].get("file", "logs/bot.log")
    max_mb = config["logging"].get("file_max_mb", 50)
    backup_count = config["logging"].get("backup_count", 5)

    # Создаём папку logs, если нет
    log_dir = Path(log_file).parent
    log_dir.mkdir(parents=True, exist_ok=True)

    # Форматтер
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Консоль
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_level)

    # Файл с ротацией
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=max_mb * 1024 * 1024,
        backupCount=backup_count,
        encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(log_level)

    # Глобальный логгер
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    # Убираем дублирование логов (если есть другие handlers)
    root_logger.propagate = False

    logger = logging.getLogger(__name__)
    logger.info("Logger initialized: level=%s, file=%s (max %dMB, %d backups)", 
                log_level, log_file, max_mb, backup_count)

    # Специально для телефона — INFO минимум
    if config.get("low_power_mode", False):
        root_logger.setLevel(logging.INFO)
        logger.info("Low power mode: logging level forced to INFO")

    return logger


# Инициализация при импорте модуля
setup_logger()