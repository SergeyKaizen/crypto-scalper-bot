"""
src/data/binance_client.py

=== Основной принцип работы файла ===

Единая точка взаимодействия с Binance Futures API через ccxt.
Класс BinanceClient:
- инициализирует соединение с API key/secret + поддержкой прокси
- соблюдает rate-limits (встроенный механизм ccxt + ручной retry + backoff)
- получает актуальный список активных USDT-перпетуальных фьючерсов
- скачивает OHLCV + taker_buy_base_volume
- вычисляет bid и ask по формуле из ТЗ:
  ask = taker_buy_base_volume (рыночные покупки)
  bid = volume - taker_buy_base_volume (рыночные продажи)
- фильтрует монеты по минимальному возрасту (min_age_months)
- удаляет данные delisted-монет из хранилища

Ключевые изменения:
- В update_markets_list() автоматически выявляет delisted монеты (сравнивая stored vs active)
  и вызывает storage.remove_delisted(to_remove) для автоматической очистки.
- remove_delisted теперь расширен в storage.py для удаления данных, моделей и из whitelist.

=== Главные методы ===

- __init__(proxies: dict = None)
- _check_connection()
- update_markets_list() → list[str]  # отфильтрованные активные символы, с автоматической очисткой delisted
- fetch_klines(symbol, timeframe, since=None, limit=1000) → pd.DataFrame | None

Логирование через setup_logger.
"""

import ccxt
import time
import random
import pandas as pd
from datetime import datetime, timedelta
import logging

from src.core.config import load_config
from src.utils.logger import setup_logger

logger = setup_logger('binance_client', logging.INFO)

class BinanceClient:
    def __init__(self, proxies: dict = None):
        """
        Инициализация клиента Binance Futures.

        Параметры:
            proxies (dict, optional): словарь прокси в формате ccxt
                {'http': '...', 'https': '...'}
                Если передан — переопределяет значение из конфига.
        """
        config = load_config()

        self.api_key = config['binance']['api_key']
        self.api_secret = config['binance']['api_secret']

        # Прокси: приоритет — переданный в конструктор > конфиг > None
        self.proxies = proxies if proxies is not None else config.get('proxies', None)

        self.exchange = ccxt.binance({
            'apiKey': self.api_key,
            'secret': self.api_secret,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future',           # только perpetual USDT futures
                'adjustForTimeDifference': True,
                'recvWindow': 10000,
            },
            'proxies': self.proxies,
            'verbose': False,                      # можно включить для отладки
        })

        self._check_connection()

    def _check_connection(self):
        """Проверка базового соединения с биржей"""
        try:
            self.exchange.load_markets()
            logger.info("Подключение к Binance Futures успешно установлено.")
            if self.proxies:
                logger.info(f"Используются прокси: {self.proxies}")
        except Exception as e:
            logger.error(f"Ошибка инициализации соединения с Binance: {e}")
            raise

    def update_markets_list(self) -> list[str]:
        """
        Обновляет и возвращает список активных USDT-перпетуальных фьючерсов,
        отфильтрованных по минимальному возрасту монеты (min_age_months).

        Автоматически выявляет delisted монеты и вызывает storage.remove_delisted()
        для удаления всех данных, моделей и записи из whitelist/symbols_meta.
        """
        config = load_config()
        min_age_months = config['filter'].get('min_age_months', 3)

        try:
            markets = self.exchange.load_markets(reload=True)
        except Exception as e:
            logger.error(f"Не удалось загрузить markets: {e}")
            return []

        symbols = [
            symbol for symbol, market in markets.items()
            if market['type'] == 'future'
            and market['quote'] == 'USDT'
            and market.get('active', False)
            and symbol.endswith('USDT')
        ]

        filtered_symbols = []
        now = datetime.utcnow()

        for symbol in symbols:
            info = markets[symbol].get('info', {})
            listing_str = info.get('listingDate') or info.get('deliveryDate') or info.get('onboardDate')

            if listing_str:
                try:
                    # Binance иногда возвращает строку в ms или ISO
                    if isinstance(listing_str, str) and 'T' in listing_str:
                        listing_dt = datetime.fromisoformat(listing_str.replace('Z', '+00:00'))
                    else:
                        # timestamp в ms
                        listing_dt = datetime.utcfromtimestamp(int(listing_str) / 1000)

                    age_days = (now - listing_dt).days
                    if age_days >= min_age_months * 30:
                        filtered_symbols.append(symbol)
                except (ValueError, TypeError):
                    # Если дата некорректная — пропускаем или считаем старой
                    filtered_symbols.append(symbol)
            else:
                # Нет даты листинга → считаем старую монету
                filtered_symbols.append(symbol)

        # Автоматическая очистка delisted монет
        from src.data.storage import Storage
        storage = Storage()
        all_stored = storage.get_all_symbols()
        active_now = set(filtered_symbols)  # Используем отфильтрованные активные
        to_remove = [s for s in all_stored if s not in active_now]

        if to_remove:
            storage.remove_delisted(to_remove)
            logger.info(f"Автоматически удалены данные по {len(to_remove)} delisted монетам")

        logger.info(f"Актуальный отфильтрованный список: {len(filtered_symbols)} символов")
        return filtered_symbols

    def fetch_klines(self,
                     symbol: str,
                     timeframe: str,
                     since: int = None,
                     limit: int = 1000) -> pd.DataFrame | None:
        """
        Скачивает свечи (OHLCV + taker_buy_base_volume).
        Возвращает DataFrame с колонками:
            timestamp (int ms), open, high, low, close, volume, bid, ask, buy_volume

        buy_volume = taker_buy_base_volume (рыночные покупки) — для совместимости с resampler.py
        bid = volume - buy_volume (рыночные продажи)

        timestamp остаётся в миллисекундах (int), без установки в index — чтобы не дублировать преобразования.
        """
        retries = 6
        backoff = 1

        for attempt in range(retries):
            try:
                klines = self.exchange.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    since=since,
                    limit=limit,
                    params={'recvWindow': 20000}
                )

                if not klines:
                    logger.debug(f"Пустой ответ для {symbol} {timeframe}")
                    return None

                data = []
                for candle in klines:
                    ts, o, h, l, c, v = candle[:6]
                    # taker_buy_base_asset_volume — 7-й элемент (индекс 6), если присутствует
                    taker_buy_base = candle[6] if len(candle) > 6 else 0.0
                    buy_volume = taker_buy_base
                    bid_volume = v - taker_buy_base

                    data.append([ts, o, h, l, c, v, bid_volume, buy_volume, buy_volume])

                df = pd.DataFrame(data, columns=[
                    'timestamp', 'open', 'high', 'low', 'close',
                    'volume', 'bid', 'ask', 'buy_volume'
                ])

                # timestamp остаётся int (ms) — resampler ожидает именно так
                # НЕ ставим в index, НЕ преобразуем в datetime здесь — это делает storage / resampler при необходимости

                logger.debug(f"Получено {len(df)} свечей {symbol} {timeframe}")
                return df

            except ccxt.RateLimitExceeded:
                wait = backoff + random.uniform(0, 1)
                logger.warning(f"Rate limit exceeded → ждём {wait:.2f} сек (попытка {attempt+1}/{retries})")
                time.sleep(wait)
                backoff *= 2
                if backoff > 32:
                    backoff = 32

            except ccxt.NetworkError as e:
                wait = 5 + random.uniform(0, 10)
                logger.error(f"Сетевая ошибка: {e}. Повтор через {wait:.1f} сек")
                time.sleep(wait)

            except ccxt.ExchangeError as e:
                logger.error(f"Ошибка биржи {symbol} {timeframe}: {e}")
                if attempt == retries - 1:
                    raise
                time.sleep(3 + random.uniform(0, 3))

            except Exception as e:
                logger.exception(f"Неизвестная ошибка при скачке {symbol} {timeframe}")
                if attempt == retries - 1:
                    raise
                time.sleep(2)

        logger.error(f"Не удалось скачать {symbol} {timeframe} после {retries} попыток")
        return None


if __name__ == "__main__":
    # Простой тест
    client = BinanceClient()
    symbols = client.update_markets_list()
    print(f"Найдено {len(symbols)} активных символов")
    print(symbols[:10])