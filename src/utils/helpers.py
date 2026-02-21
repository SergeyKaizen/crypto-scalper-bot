"""
src/utils/helpers.py

=== Основной принцип работы файла ===

Этот файл содержит вспомогательные утилиты и helper-функции, которые используются в разных частях проекта.
Он не содержит бизнес-логики, а только полезные инструменты: форматирование, конвертации, математические хелперы, проверки и т.д.

Ключевые задачи:
- Форматирование чисел, дат, строк для логов и вывода.
- Конвертация timestamp <-> datetime.
- Безопасные математические операции (защита от деления на 0).
- Проверка валидности символов, TF, параметров.
- Утилиты для работы с dict/list (flatten, merge и т.д.).
- Небольшой набор констант/функций для повторяющегося кода.

Файл импортируется везде, где нужно упростить и унифицировать код.

=== Главные функции и за что отвечают ===

- format_number(value: float, decimals: int = 2, thousands_sep: bool = True) → str
  Форматирует число с разделителем тысяч и заданным количеством знаков после точки.
  Пример: 1234567.89 → "1,234,567.89" (decimals=2, thousands_sep=True)

- timestamp_to_datetime(ts: int) → datetime
  Конвертирует миллисекунды (Unix timestamp * 1000) в datetime UTC.

- safe_div(a: float, b: float, default: float = 0.0) → float
  Безопасное деление. Если b == 0 — возвращает default.

- safe_log(x: float, default: float = 0.0) → float
  Безопасный логарифм. Если x <= 0 — default.

- is_valid_symbol(symbol: str) → bool
  Проверяет, что символ в формате XXXUSDT (для фьючерсов).

- is_valid_timeframe(tf: str) → bool
  Проверяет, что TF из списка ['1m','3m','5m','10m','15m'].

- merge_dicts(dict1, dict2, overwrite: bool = True) → dict
  Глубокое слияние двух словарей. Если overwrite=True — значения из dict2 перезаписывают dict1.

- get_nested_dict(d: dict, keys: list, default=None)
  Безопасный доступ к вложенному словарю по списку ключей (без KeyError).

=== Примечания ===
- Все функции чистые, без side-effects.
- Нет зависимостей от других модулей проекта (кроме logging если нужно).
- Полностью соответствует ТЗ: вспомогательные утилиты без бизнес-логики.
- Готов к использованию в любом файле.
- Логи минимальны (только ошибки).
"""

from datetime import datetime
import math

def format_number(value: float, decimals: int = 2, thousands_sep: bool = True) -> str:
    """
    Форматирует число с разделителем тысяч и заданным количеством знаков после точки.
    Пример: 1234567.89 → "1,234,567.89" (decimals=2, thousands_sep=True)
    """
    if not isinstance(value, (int, float)):
        return str(value)
    
    fmt = f"{{:,.{decimals}f}}" if thousands_sep else f"{{:.{decimals}f}}"
    return fmt.format(value)

def timestamp_to_datetime(ts: int) -> datetime:
    """
    Конвертирует миллисекунды (Unix timestamp * 1000) в datetime UTC.
    """
    return datetime.utcfromtimestamp(ts / 1000)

def safe_div(a: float, b: float, default: float = 0.0) -> float:
    """
    Безопасное деление. Если b == 0 — возвращает default.
    """
    return a / b if b != 0 else default

def safe_log(x: float, default: float = 0.0) -> float:
    """
    Безопасный логарифм. Если x <= 0 — default.
    """
    return math.log(x) if x > 0 else default

def is_valid_symbol(symbol: str) -> bool:
    """
    Проверяет, что символ в формате XXXUSDT (для фьючерсов).
    """
    return isinstance(symbol, str) and len(symbol) >= 6 and symbol.upper().endswith('USDT')

def is_valid_timeframe(tf: str) -> bool:
    """
    Проверяет, что таймфрейм из списка разрешённых.
    """
    valid = {'1m', '3m', '5m', '10m', '15m'}
    return tf in valid

def merge_dicts(d1: dict, d2: dict, overwrite: bool = True) -> dict:
    """
    Глубокое слияние двух словарей. Если overwrite=True — значения из d2 перезаписывают d1.
    """
    result = d1.copy()
    for key, value in d2.items():
        if isinstance(value, dict) and key in result and isinstance(result[key], dict):
            result[key] = merge_dicts(result[key], value, overwrite)
        else:
            if overwrite or key not in result:
                result[key] = value
    return result

def get_nested_dict(d: dict, keys: list, default=None):
    """
    Безопасный доступ к вложенному словарю по списку ключей.
    Пример: get_nested_dict(config, ['trading', 'risk_pct'], 1.0)
    """
    current = d
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current