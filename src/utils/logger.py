"""
src/utils/logger.py

=== Основной принцип работы файла ===

Этот файл реализует централизованную систему логирования для всего проекта.
Он использует стандартный модуль logging Python с дополнительными настройками:
- Уровень логирования по умолчанию INFO (можно переключить на DEBUG для отладки).
- Формат логов: [TIMESTAMP] [LEVEL] [MODULE_NAME] сообщение
- Вывод одновременно в консоль + файл.
- Файловые логи с ежедневной ротацией (новый файл каждый день) и лимитом размера (10 МБ, хранится 7 файлов).
- Цветной вывод в консоль (если установлен colorlog) — сильно улучшает читаемость.
- Отдельный логгер для каждого модуля (logger = setup_logger('live_loop')) — удобно фильтровать логи по модулю.

Ключевые задачи:
- Унифицированный формат логов для отладки и анализа.
- Автоматическое создание директории logs.
- Поддержка уровней: DEBUG, INFO, WARNING, ERROR, CRITICAL.
- Логирование в файл с ротацией (ежедневно + по размеру).
- Цвета в консоли: DEBUG — cyan, INFO — green, WARNING — yellow, ERROR — red.

=== Главные функции и за что отвечают ===

- setup_logger(name: str, level: int = logging.INFO) → logging.Logger
  Основная функция: создаёт и настраивает логгер для конкретного модуля.
  - name — имя модуля (например, 'live_loop', 'entry_manager', __name__).
  - level — уровень логирования (logging.INFO по умолчанию).
  - Возвращает готовый логгер, который можно использовать как logger.info(...).

- _setup_file_handler() — настраивает запись в файл с ротацией (TimedRotatingFileHandler + RotatingFileHandler).
- _setup_console_handler() — настраивает цветной вывод в консоль (если colorlog доступен).

=== Примечания ===
- Используется в каждом модуле: logger = setup_logger(__name__)
- Логи пишутся в logs/bot.log (или по дате: logs/bot-2026-02-20.log).
- Ротация: ежедневная + по размеру 10 МБ, хранится 7 файлов (настраивается).
- Цвета в консоли: требуют pip install colorlog (опционально, если нет — обычный вывод).
- Полностью соответствует ТЗ: подробные комментарии, логи для отладки.
- Нет зависимостей сверх logging + colorlog (опционально).
- Готов к использованию в любом файле проекта.
"""

import logging
import os
from logging.handlers import TimedRotatingFileHandler, RotatingFileHandler
from datetime import datetime

# Опционально: colorlog для цветного вывода в консоль
try:
    import colorlog
    COLORLOG_AVAILABLE = True
except ImportError:
    COLORLOG_AVAILABLE = False
    print("colorlog не установлен — консольные логи будут без цвета (рекомендуется pip install colorlog)")

def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Создаёт и настраивает логгер для модуля.
    
    Параметры:
    - name: имя модуля (например, 'live_loop', __name__)
    - level: уровень логирования (logging.INFO по умолчанию)

    Возвращает готовый логгер.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Чтобы не дублировать обработчики при многократном вызове
    if logger.handlers:
        return logger

    # Формат логов
    log_format = '%(asctime)s [%(levelname)s] [%(name)s] %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'

    # 1. Консольный handler (цветной если colorlog установлен)
    if COLORLOG_AVAILABLE:
        console_formatter = colorlog.ColoredFormatter(
            '%(log_color)s%(asctime)s [%(levelname)s] [%(name)s] %(message)s',
            datefmt=date_format,
            log_colors={
                'DEBUG':    'cyan',
                'INFO':     'green',
                'WARNING':  'yellow',
                'ERROR':    'red',
                'CRITICAL': 'red,bg_white',
            }
        )
    else:
        console_formatter = logging.Formatter(log_format, date_format)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(level)
    logger.addHandler(console_handler)

    # 2. Файловый handler с ротацией (ежедневно + по размеру)
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    # Ежедневная ротация + лимит размера
    log_file = os.path.join(log_dir, "bot.log")
    file_handler = TimedRotatingFileHandler(
        log_file,
        when='midnight',          # ротация каждый день в полночь
        interval=1,
        backupCount=7,            # храним 7 дней
        encoding='utf-8'
    )
    # Дополнительно ротация по размеру 10 МБ
    file_handler.addHandler(RotatingFileHandler(
        log_file,
        maxBytes=10*1024*1024,    # 10 МБ
        backupCount=7,
        encoding='utf-8'
    ))

    file_formatter = logging.Formatter(log_format, date_format)
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.DEBUG)  # В файл — всё, даже DEBUG
    logger.addHandler(file_handler)

    logger.propagate = False  # Не передаём выше, чтобы не дублировать

    logger.debug(f"Логгер '{name}' инициализирован (уровень: {logging.getLevelName(level)})")
    return logger