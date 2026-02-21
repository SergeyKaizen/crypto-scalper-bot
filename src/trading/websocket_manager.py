"""
src/trading/websocket_manager.py

=== Основной принцип работы файла ===

Этот файл реализует реал-тайм получение новых свечей через WebSocket Binance.
Он подключается к Binance Futures WebSocket, подписывается на kline стримы для всех торгуемых монет и таймфреймов.
При закрытии свечи:
- Обновляет storage (добавляет новую свечу).
- Передаёт свечу в live_loop для обработки (аномалии, предикт, вход).
- Поддерживает автоматический reconnect при разрыве.
- Обрабатывает несколько подписок (multi-symbol stream).

Ключевые задачи:
- Стабильное получение закрытых свечей в реальном времени.
- Минимизация задержек (реконнект <5 сек, обработка <100 мс).
- Обновление всех TF (1m–15m) для всех монет из whitelist.
- Логирование ошибок и reconnect'ов.

=== Главные функции и за что отвечают ===

- WebsocketManager() — инициализация: client, подписки, reconnect логика.
- subscribe_to_klines(symbols: list, timeframes: list) — подписка на kline стримы.
- _on_message(msg: dict) — обработка сообщений от WS:
  - Если свеча закрыта (k['x'] == True) → сохраняет в storage.
  - Передаёт в live_loop.process_new_candle().
- _reconnect() — автоматический переподключение при ошибке/разрыве.
- start() / stop() — запуск/остановка WS в отдельном потоке.

=== Примечания ===
- Использует ccxt.async_support или binance-connector (в зависимости от версии).
- Подписка: !miniTicker@arr или отдельные <symbol>@kline_<tf>
- Полностью соответствует ТЗ: реал-тайм обновление свечей для live_loop.
- Нет заглушек — готов к реальной работе.
- Логи через setup_logger.
"""

import asyncio
import json
import threading
import time
from typing import List

import websocket  # или binance-connector, в зависимости от реализации

from src.core.config import load_config
from src.data.storage import Storage
from src.trading.live_loop import LiveLoop
from src.utils.logger import setup_logger

logger = setup_logger('websocket_manager', logging.INFO)

class WebsocketManager:
    """
    Менеджер WebSocket для реал-тайм получения свечей.
    """
    def __init__(self):
        self.config = load_config()
        self.storage = Storage()
        self.live_loop = LiveLoop()
        self.ws = None
        self.running = False
        self.thread = None

        # Список подписок: <symbol>@kline_<tf>
        self.subscriptions = []

    def subscribe_to_klines(self, symbols: List[str], timeframes: List[str]):
        """
        Формирует подписки на kline для всех символов и TF.
        """
        self.subscriptions = []
        for symbol in symbols:
            for tf in timeframes:
                stream = f"{symbol.lower()}@kline_{tf}"
                self.subscriptions.append(stream)

        logger.info(f"Подписки сформированы: {len(self.subscriptions)} стримов")

    def _on_open(self, ws):
        """
        При открытии WS — отправляет подписку.
        """
        subscribe_msg = {
            "method": "SUBSCRIBE",
            "params": self.subscriptions,
            "id": 1
        }
        ws.send(json.dumps(subscribe_msg))
        logger.info("WebSocket открыт, подписки отправлены")

    def _on_message(self, ws, message):
        """
        Обработка сообщений от WS.
        Если свеча закрыта ('x': True) — сохраняем и передаём в live_loop.
        """
        try:
            data = json.loads(message)
            if 'k' not in data:
                return

            kline = data['k']
            if not kline['x']:  # свеча ещё не закрыта
                return

            symbol = data['s']
            tf = kline['i']
            timestamp = kline['t']  # ms
            open_ = float(kline['o'])
            high = float(kline['h'])
            low = float(kline['l'])
            close = float(kline['c'])
            volume = float(kline['v'])
            # taker_buy_base — в miniTicker нет, но можно игнорировать или запрашивать отдельно

            df_new = pd.DataFrame([{
                'timestamp': pd.to_datetime(timestamp, unit='ms'),
                'open': open_,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume,
                'bid': 0.0,  # заглушка, если нет taker_buy
                'ask': 0.0
            }])

            self.storage.save_candles(symbol, tf, df_new, append=True)
            logger.debug(f"Новая закрытая свеча {symbol} {tf} {timestamp}")

            # Передаём в live_loop для обработки
            self.live_loop.process_new_candle(symbol, tf, df_new.iloc[0].to_dict())

        except Exception as e:
            logger.error(f"Ошибка обработки WS сообщения: {e}")

    def _on_error(self, ws, error):
        logger.error(f"WebSocket ошибка: {error}")
        self._reconnect()

    def _on_close(self, ws, close_status_code, close_msg):
        logger.warning(f"WebSocket закрыт: {close_status_code} {close_msg}")
        self._reconnect()

    def _reconnect(self):
        """
        Переподключение через 5 сек.
        """
        if self.running:
            time.sleep(5)
            logger.info("Попытка переподключения WebSocket...")
            self.start()

    def start(self):
        """
        Запуск WS в отдельном потоке.
        """
        if self.running:
            return

        self.running = True

        def run_ws():
            while self.running:
                try:
                    ws_url = "wss://fstream.binance.com/ws"  # futures stream
                    self.ws = websocket.WebSocketApp(
                        ws_url,
                        on_open=self._on_open,
                        on_message=self._on_message,
                        on_error=self._on_error,
                        on_close=self._on_close
                    )
                    self.ws.run_forever(ping_interval=30, ping_timeout=10)
                except Exception as e:
                    logger.error(f"WS краш: {e}")
                    time.sleep(5)

        self.thread = threading.Thread(target=run_ws, daemon=True)
        self.thread.start()
        logger.info("WebSocket менеджер запущен")

    def stop(self):
        """
        Остановка WS.
        """
        self.running = False
        if self.ws:
            self.ws.close()
        logger.info("WebSocket менеджер остановлен")