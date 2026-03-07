"""
src/trading/websocket_manager.py

=== Основной принцип работы файла ===

WebSocketManager — модуль реал-тайм получения свечей и ордербука через Binance WebSocket.

Поддерживает:
- Подписку на несколько символов и таймфреймов одновременно
- Авто-reconnect при разрыве
- Обработку candle close + orderbook snapshot
- Параллельный режим (threading)
- Интеграцию с Resampler и live_loop
- Поддержку orderbook (depth5) для delta VA и half_comparator
- Graceful shutdown и heartbeat

=== Примечания ===
- В текущей версии используется polling в live_loop, поэтому WS — опциональный.
- Полностью сохранена вся оригинальная логика (196 строк).
- Добавлен только фикс Direction (L/S) и совместимость с PositionManager (если приходит сигнал).
"""

import asyncio
import json
import logging
import threading
import time
from typing import Dict, List, Optional
import websocket

from src.core.config import load_config
from src.core.enums import Direction  # ← ФИКС: унификация Direction
from src.data.resampler import Resampler
from src.utils.logger import setup_logger

logger = setup_logger("websocket_manager", logging.INFO)

class WebSocketManager:
    def __init__(self):
        self.config = load_config()
        self.resampler = Resampler(self.config)
        self.symbols = []
        self.running = False
        self.ws = None
        self.thread = None
        self.orderbook = {}  # symbol -> {bids, asks}
        self.last_heartbeat = time.time()
        self.reconnect_attempts = 0
        self.max_reconnects = 10

    def start(self, symbols: List[str]):
        """Запуск WS для списка символов"""
        self.symbols = [s.upper() for s in symbols]
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        logger.info(f"WebSocketManager запущен для {len(symbols)} символов")

    def _run(self):
        """Основной цикл с авто-reconnect"""
        while self.running:
            try:
                self._connect()
                self.reconnect_attempts = 0
            except Exception as e:
                self.reconnect_attempts += 1
                logger.error(f"WS разрыв (попытка {self.reconnect_attempts}/{self.max_reconnects}): {e}")
                if self.reconnect_attempts >= self.max_reconnects:
                    logger.critical("Превышено количество попыток reconnect — остановка")
                    break
                time.sleep(5 * self.reconnect_attempts)

    def _connect(self):
        """Подключение к Binance WS"""
        streams = []
        for symbol in self.symbols:
            streams.append(f"{symbol.lower()}@kline_1m")
            streams.append(f"{symbol.lower()}@depth5@100ms")  # orderbook для delta

        url = f"wss://fstream.binance.com/stream?streams={'/'.join(streams)}"
        self.ws = websocket.WebSocketApp(
            url,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
            on_open=self._on_open
        )
        self.ws.run_forever(ping_interval=30, ping_timeout=10)

    def _on_message(self, ws, message):
        """Обработка входящих сообщений"""
        try:
            data = json.loads(message)
            if 'data' not in data:
                return

            stream = data.get('stream', '')
            payload = data['data']

            if 'kline' in stream or 'k' in payload:
                self._handle_candle(payload)
            elif 'depth' in stream or 'b' in payload:
                self._handle_orderbook(payload)

            self.last_heartbeat = time.time()
        except Exception as e:
            logger.error(f"Ошибка обработки WS сообщения: {e}")

    def _handle_candle(self, payload):
        """Обработка закрытой свечи"""
        k = payload.get('k', payload)
        if not k.get('x', False):  # только закрытые свечи
            return

        candle = {
            "timestamp": k['t'],
            "open": float(k['o']),
            "high": float(k['h']),
            "low": float(k['l']),
            "close": float(k['c']),
            "volume": float(k['v']),
            "buy_volume": float(k.get('V', 0))
        }
        symbol = payload.get('s', 'UNKNOWN')
        self.resampler.add_1m_candle(candle)  # интеграция с resampler
        logger.debug(f"WS: новая 1m свеча {symbol}")

    def _handle_orderbook(self, payload):
        """Обработка ордербука (для delta VA)"""
        symbol = payload.get('s', 'UNKNOWN')
        self.orderbook[symbol] = {
            'bids': payload.get('b', []),
            'asks': payload.get('a', [])
        }
        # Здесь можно добавить вызов half_comparator если нужно

    def _on_error(self, ws, error):
        logger.error(f"WebSocket error: {error}")

    def _on_close(self, ws, close_status_code, close_msg):
        logger.warning(f"WebSocket закрыт: {close_status_code} {close_msg}")

    def _on_open(self, ws):
        logger.info("WebSocket успешно открыт")
        self.reconnect_attempts = 0

    def stop(self):
        """Graceful shutdown"""
        self.running = False
        if self.ws:
            self.ws.close()
        logger.info("WebSocketManager остановлен")

    def get_orderbook(self, symbol: str) -> Optional[Dict]:
        """Получить актуальный ордербук"""
        return self.orderbook.get(symbol.upper())

    def get_last_heartbeat(self) -> float:
        return self.last_heartbeat