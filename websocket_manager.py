"""
websocket_manager.py
Finnhub WebSocket — free tier, no payment required.

Symbols:
  OANDA:XAU_USD  -> Gold  (24/5)
  OANDA:EUR_USD  -> EURUSD (24/5)
  DIA            -> US30  (market hours only)

Handles reconnection, ping/pong, silent drop watchdog.
"""

import asyncio
import json
import logging
import time

import websockets

log = logging.getLogger(__name__)

FINNHUB_WS_URL = "wss://ws.finnhub.io"
SYMBOLS = ["OANDA:XAU_USD", "OANDA:EUR_USD", "DIA"]

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
        url = f"{FINNHUB_WS_URL}?token={self.api_key}"
        log.info("Connecting to Finnhub WebSocket...")
        async with websockets.connect(
            url,
            ping_interval=20,
            ping_timeout=10,
            close_timeout=5,
        ) as ws:
            self.ws = ws
            self.last_message = time.time()
            log.info("Finnhub WebSocket connected.")
            for symbol in SYMBOLS:
                await ws.send(json.dumps({"type": "subscribe", "symbol": symbol}))
                log.info(f"Subscribed: {symbol}")
            async for raw in ws:
                self.last_message = time.time()
                await self._handle_message(raw)

    async def _handle_message(self, raw: str):
        try:
            msg = json.loads(raw)
        except Exception:
            return
        msg_type = msg.get("type")
        if msg_type == "trade":
            for trade in msg.get("data", []):
                symbol = trade.get("s")
                price  = trade.get("p")
                if symbol and price:
                    try:
                        p = float(price)
                        live_prices[symbol] = p
                        last_tick_time[symbol] = time.time()
                        await self.on_tick(symbol, p)
                    except (ValueError, TypeError):
                        pass
        elif msg_type == "ping":
            try:
                await self.ws.send(json.dumps({"type": "pong"}))
                log.debug("Pong sent.")
            except Exception:
                pass
        elif msg_type == "error":
            log.error(f"Finnhub WS error: {msg.get('msg')}")
        else:
            log.debug(f"Finnhub WS msg: {msg_type}")

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
