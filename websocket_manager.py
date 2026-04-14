"""
websocket_manager.py
Persistent WebSocket connection to Twelve Data.
Feeds live price ticks into scoring engine on every tick.
Handles reconnection, heartbeat, silent drop detection.
"""

import asyncio
import json
import logging
import time

import websockets

log = logging.getLogger(__name__)

TWELVEDATA_WS_URL = "wss://ws.twelvedata.com/v1/quotes/price"
SYMBOLS = ["GLD", "DIA", "UUP", "VIXY"]

live_prices: dict = {}
last_tick_time: dict = {}


class WebSocketManager:
    def __init__(self, api_key: str, on_tick_callback):
        self.api_key = api_key
        self.on_tick = on_tick_callback
        self.ws = None
        self.running = False
        self.reconnect_delay = 5
        self.max_delay = 60
        self.last_message = time.time()
        self.watchdog_task = None

    async def start(self):
        self.running = True
        self.watchdog_task = asyncio.create_task(self._watchdog())
        while self.running:
            try:
                await self._connect_and_listen()
                self.reconnect_delay = 5
            except Exception as e:
                log.error(f"WebSocket error: {e} — reconnecting in {self.reconnect_delay}s")
                await asyncio.sleep(self.reconnect_delay)
                self.reconnect_delay = min(self.reconnect_delay * 2, self.max_delay)

    async def stop(self):
        self.running = False
        if self.ws:
            await self.ws.close()
        if self.watchdog_task:
            self.watchdog_task.cancel()

    async def _connect_and_listen(self):
        log.info("WebSocket connecting to Twelve Data...")
        async with websockets.connect(
            TWELVEDATA_WS_URL,
            ping_interval=20,
            ping_timeout=10,
            close_timeout=5,
        ) as ws:
            self.ws = ws
            self.last_message = time.time()
            log.info("WebSocket connected.")
            subscribe_msg = {
                "action": "subscribe",
                "params": {
                    "symbols": ",".join(SYMBOLS),
                    "apikey": self.api_key,
                },
            }
            await ws.send(json.dumps(subscribe_msg))
            log.info(f"Subscribed to: {SYMBOLS}")
            async for raw in ws:
                self.last_message = time.time()
                await self._handle_message(raw)

    async def _handle_message(self, raw: str):
        try:
            msg = json.loads(raw)
        except Exception:
            return
        event = msg.get("event")
        if event == "price":
            symbol = msg.get("symbol")
            price = msg.get("price")
            if symbol and price:
                try:
                    p = float(price)
                    live_prices[symbol] = p
                    last_tick_time[symbol] = time.time()
                    await self.on_tick(symbol, p)
                except (ValueError, TypeError):
                    pass
        elif event == "subscribe-status":
            log.info(f"Subscribe status: {msg}")
        elif event == "heartbeat":
            log.debug("Heartbeat received.")
        elif event == "error":
            log.error(f"WS server error: {msg}")

    async def _watchdog(self):
        while self.running:
            await asyncio.sleep(30)
            elapsed = time.time() - self.last_message
            if elapsed > 45 and self.ws:
                log.warning(f"WebSocket silent {elapsed:.0f}s — forcing reconnect.")
                try:
                    await self.ws.close()
                except Exception:
                    pass

    def get_price(self, symbol: str):
        return live_prices.get(symbol)

    def price_age(self, symbol: str) -> float:
        last = last_tick_time.get(symbol)
        return time.time() - last if last else 999.0

    def is_fresh(self, symbol: str, max_age: float = 60.0) -> bool:
        return self.price_age(symbol) < max_age
