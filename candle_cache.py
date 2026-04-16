"""
candle_cache.py
OHLCV candle history per symbol per timeframe via Twelve Data REST.
Supports forex symbols (XAU/USD, EUR/USD) and ETFs (DIA, VIXY).
Staggered refresh — each timeframe only refreshes when stale.
"""

import asyncio
import logging
import time
from datetime import datetime

import httpx

log = logging.getLogger(__name__)

REFRESH_INTERVALS = {
    "15min": 900,
    "1h":    3600,
    "4h":    14400,
    "1day":  86400,
    "1week": 604800,
}

# Twelve Data interval strings
TD_INTERVAL = {
    "15min": "15min",
    "1h":    "1h",
    "4h":    "4h",
    "1day":  "1day",
    "1week": "1week",
}

CANDLE_COUNT = 50


class CandleCache:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self._cache = {}
        self._last_refresh = {}
        self._lock = asyncio.Lock()

    def get(self, symbol: str, timeframe: str):
        return self._cache.get(symbol, {}).get(timeframe)

    def get_with_live_price(self, symbol: str, timeframe: str, live_price: float):
        data = self.get(symbol, timeframe)
        if not data:
            return None
        return {
            **data,
            "closes": data["closes"][:-1] + [live_price],
            "live": True,
            "live_price": live_price,
        }

    def is_populated(self, symbol: str) -> bool:
        return bool(self._cache.get(symbol))

    def all_populated(self, symbols) -> bool:
        return all(self.is_populated(s) for s in symbols)

    async def warm_up(self, symbols, timeframes):
        log.info(f"Cache warm-up: {len(symbols)} symbols x {len(timeframes)} timeframes...")
        async with httpx.AsyncClient() as client:
            tasks = [
                self._fetch_and_store(client, sym, tf)
                for sym in symbols
                for tf in timeframes
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            errors = [r for r in results if isinstance(r, Exception)]
            if errors:
                log.warning(f"Warm-up completed with {len(errors)} errors.")
        log.info("Cache warm-up complete.")

    async def refresh_loop(self, symbols, timeframes):
        log.info("Candle cache refresh loop started.")
        while True:
            await asyncio.sleep(60)
            now = time.time()
            async with httpx.AsyncClient() as client:
                tasks = []
                for sym in symbols:
                    for tf in timeframes:
                        interval = REFRESH_INTERVALS.get(tf, 3600)
                        last = self._last_refresh.get(sym, {}).get(tf, 0)
                        if now - last >= interval:
                            tasks.append(self._fetch_and_store(client, sym, tf))
                if tasks:
                    log.info(f"Refreshing {len(tasks)} stale timeframe(s)...")
                    await asyncio.gather(*tasks, return_exceptions=True)

    async def _fetch_and_store(self, client, symbol: str, timeframe: str):
        url = "https://api.twelvedata.com/time_series"
        params = {
            "symbol":     symbol,
            "interval":   TD_INTERVAL.get(timeframe, timeframe),
            "outputsize": CANDLE_COUNT,
            "apikey":     self.api_key,
        }
        try:
            r = await client.get(url, params=params, timeout=15)
            data = r.json()
            if data.get("status") == "error":
                log.warning(f"API error {symbol} {timeframe}: {data.get('message')}")
                return
            parsed = self._parse(data)
            if not parsed:
                return
            async with self._lock:
                if symbol not in self._cache:
                    self._cache[symbol] = {}
                self._cache[symbol][timeframe] = parsed
                if symbol not in self._last_refresh:
                    self._last_refresh[symbol] = {}
                self._last_refresh[symbol][timeframe] = time.time()
            log.debug(f"Cached: {symbol} {timeframe} ({len(parsed['closes'])} candles)")
        except Exception as e:
            log.error(f"Candle fetch failed {symbol} {timeframe}: {e}")

    def _parse(self, raw: dict):
        try:
            values = list(reversed(raw.get("values", [])))
            if not values:
                return None
            return {
                "closes":  [float(v["close"])          for v in values],
                "highs":   [float(v["high"])            for v in values],
                "lows":    [float(v["low"])             for v in values],
                "opens":   [float(v["open"])            for v in values],
                "volumes": [float(v.get("volume", 1))  for v in values],
                "updated": datetime.utcnow(),
            }
        except Exception as e:
            log.error(f"Parse error: {e}")
            return None
