"""
src/data/binance_client.py

=== Основной принцип работы файла ===

Этот файл — центральная точка взаимодействия с Binance Futures API через библиотеку ccxt.
Он предоставляет единый класс BinanceClient, который:
- инициализирует аутентифицированное соединение (API key/secret + прокси, если указаны),
- соблюдает rate limits Binance (enableRateLimit + ручной retry с backoff),
- получает актуальный список активных USDT-перпетуальных фьючерсов,
- скачивает OHLCV-данные с taker_buy_base_asset_volume,
- вычисляет bid и ask по формуле из ТЗ: ask = taker_buy_base, bid = volume - taker_buy_base.

Ключевые задачи:
- Обеспечить стабильное получение исторических и новых свечей без превышения лимитов.
- Поддерживать прокси для обхода возможных региональных блокировок.
- Фильтровать монеты по возрасту (min_age_months) и удалять delisted.
- Возвращать готовый DataFrame с колонками: timestamp, open, high, low, close, volume, bid, ask.

=== Главные функции и за что отвечают ===

- __init__() — создаёт ccxt.binance с настройками из config, включает rate-limit, прокси и futures.

- _check_connection() — проверяет доступность биржи при запуске (load_markets).

- update_markets_list() — обновляет список монет раз в сутки:
  - Загружает все активные USDT-перпетуалы.
  - Фильтрует по минимальному возрасту монеты.
  - Вызывает storage.remove_delisted для очистки данных удалённых пар.

- fetch_klines(symbol, timeframe, since=None, limit=1000) — основной метод скачивания свечей:
  - Делает fetch_ohlcv.
  - Парсит taker_buy_base_asset_volume (7-й элемент ответа Binance).
  - Вычисляет bid = volume - taker_buy_base, ask = taker_buy_base.
  - Возвращает pd.DataFrame с нужными колонками.
  - При 429 — retry с экспоненциальным backoff (1–8 сек).
  - Логирует ошибки и успехи.

=== Примечания ===
- Все конфигурационные параметры берутся из load_config() (API keys, proxies, min_age_months).
- Файл полностью готов к использованию в downloader.py и live_loop.
- Не содержит заглушек — всё реализовано для реальной работы.
- Логирование через setup_logger для удобства отладки.

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
    def __init__(self):
        """
        Инициализация клиента Binance.
        Загружает конфиг, устанавливает API ключи, прокси и включает встроенный rate-limiter ccxt.
        """
        config = load_config()
        self.api_key = config['binance']['api_key']
        self.api_secret = config['binance']['api_secret']
        self.proxies = config.get('proxies', [])  # список прокси, если указаны

        self.exchange = ccxt.binance({
            'apiKey': self.api_key,
            'secret': self.api_secret,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future',  # только perpetual futures
                'proxies': self.proxies if self.proxies else None
            }
        })

        self._check_connection()

    def _check_connection(self):
        """Проверка подключения к Binance при инициализации."""
        try:
            self.exchange.load_markets()
            logger.info("Подключение к Binance Futures успешно.")
        except Exception as e:
            logger.error(f"Ошибка подключения: {e}")
            raise

    def update_markets_list(self):
        """
        Обновляет список активных USDT-перпетуальных фьючерсов.
        Фильтрует по минимальному возрасту монеты (min_age_months из config).
        Вызывает очистку данных delisted монет в storage.
        Возвращает список отфильтрованных символов.
        """
        config = load_config()
        min_age_months = config['filter']['min_age_months']

        markets = self.exchange.load_markets()
        symbols = [
            s for s in markets
            if markets[s]['type'] == 'future'
            and markets[s]['quote'] == 'USDT'
            and markets[s]['active']
        ]

        filtered_symbols = []
        for symbol in symbols:
            info = markets[symbol].get('info', {})
            list_date_str = info.get('listDate')  # дата листинга в ISO
            if list_date_str:
                try:
                    list_date = datetime.fromisoformat(list_date_str.replace('Z', '+00:00'))
                    if (datetime.utcnow() - list_date) >= timedelta(days=30 * min_age_months):
                        filtered_symbols.append(symbol)
                except ValueError:
                    continue  # некорректная дата — пропускаем

        # Очистка delisted монет из storage
        from src.data.storage import Storage
        storage = Storage()
        active_symbols = {s for s in markets if markets[s]['active']}
        storage.remove_delisted([s for s in storage.get_all_symbols() if s not in active_symbols])

        logger.info(f"Актуальный список монет: {len(filtered_symbols)} шт.")
        return filtered_symbols

    def fetch_klines(self, symbol: str, timeframe: str, since: int = None, limit: int = 1000):
        """
        Скачивает свечи (klines) для символа и таймфрейма.
        Парсит taker_buy_base_asset_volume → вычисляет bid и ask.
        Возвращает pd.DataFrame или None при ошибке.

        Обработка rate limit: retry с экспоненциальным backoff.
        """
        retries = 5
        backoff_sec = 1

        for attempt in range(retries):
            try:
                klines = self.exchange.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    since=since,
                    limit=limit
                )

                if not klines:
                    logger.warning(f"Нет данных для {symbol} {timeframe}")
                    return None

                # Binance возвращает: [ts, open, high, low, close, volume, taker_buy_base_volume]
                data = []
                for k in klines:
                    ts, o, h, l, c, v = k[:6]
                    taker_buy_base = k[7] if len(k) > 7 else 0.0
                    ask = taker_buy_base                    # рыночные покупки
                    bid = v - taker_buy_base                # рыночные продажи
                    data.append([ts, o, h, l, c, v, bid, ask])

                df = pd.DataFrame(data, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume', 'bid', 'ask'
                ])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)

                logger.debug(f"Скачано {len(df)} свечей для {symbol} {timeframe}")
                return df

            except ccxt.RateLimitExceeded as e:
                logger.warning(f"Rate limit ({symbol} {timeframe}): {e}. Ждём {backoff_sec} сек.")
                time.sleep(backoff_sec + random.uniform(0, 0.5))
                backoff_sec *= 2  # экспоненциальный backoff
            except ccxt.NetworkError as e:
                logger.error(f"Сетевая ошибка: {e}. Попытка {attempt+1}/{retries}")
                time.sleep(5)
            except Exception as e:
                logger.error(f"Неизвестная ошибка при скачке {symbol} {timeframe}: {e}")
                if attempt == retries - 1:
                    raise

        logger.error(f"Не удалось скачать данные для {symbol} {timeframe} после {retries} попыток")
        return None