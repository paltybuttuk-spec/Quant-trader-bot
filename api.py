"""
QuantRisk Bot — FastAPI Backend
Fetches live data from Twelve Data + Finnhub, caches it, serves to website.
"""

import os
import time
import math
import statistics
import hashlib
import hmac
from datetime import datetime, timezone
from typing import Optional
import httpx
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.responses import JSONResponse, FileResponse
import asyncio

app = FastAPI(title="QuantRisk API", docs_url=None, redoc_url=None)

# ── CORS (allow your Railway domain) ──────────────────────────────────────────
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET"],
    allow_headers=["*"],
)

# ── Auth ──────────────────────────────────────────────────────────────────────
security = HTTPBasic()
DASHBOARD_PASSWORD = os.getenv("DASHBOARD_PASSWORD", "changeme")
DASHBOARD_USER     = os.getenv("DASHBOARD_USER", "quant")

def verify_credentials(credentials: HTTPBasicCredentials = Depends(security)):
    ok_user = hmac.compare_digest(credentials.username.encode(), DASHBOARD_USER.encode())
    ok_pass = hmac.compare_digest(credentials.password.encode(), DASHBOARD_PASSWORD.encode())
    if not (ok_user and ok_pass):
        raise HTTPException(status_code=401, detail="Unauthorized",
                            headers={"WWW-Authenticate": "Basic"})
    return credentials.username

# ── API keys ──────────────────────────────────────────────────────────────────
TWELVE_KEY  = os.getenv("TWELVE_DATA_API_KEY", "")
FINNHUB_KEY = os.getenv("FINNHUB_API_KEY", "")

INSTRUMENTS = {
    "gold": {"symbol": "GLD",  "name": "Gold (GLD)",   "type": "Commodity"},
    "us30": {"symbol": "DIA",  "name": "US30 (DIA)",   "type": "Index"},
}

TIMEFRAMES = ["15min", "1h", "4h", "1day", "1week"]
TF_LABELS  = ["15M", "1H", "4H", "Daily", "Weekly"]

# ── In-memory cache ───────────────────────────────────────────────────────────
_cache: dict = {}
CACHE_TTL = 300  # 5 minutes

def cache_get(key: str):
    entry = _cache.get(key)
    if entry and time.time() - entry["ts"] < CACHE_TTL:
        return entry["data"]
    return None

def cache_set(key: str, data):
    _cache[key] = {"ts": time.time(), "data": data}

# ── Twelve Data fetch ─────────────────────────────────────────────────────────
async def fetch_candles(symbol: str, interval: str, outputsize: int = 50) -> list:
    url = (
        f"https://api.twelvedata.com/time_series"
        f"?symbol={symbol}&interval={interval}&outputsize={outputsize}"
        f"&apikey={TWELVE_KEY}&format=JSON"
    )
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.get(url)
        r.raise_for_status()
        data = r.json()
    if data.get("status") == "error":
        raise ValueError(data.get("message", "Twelve Data error"))
    values = data.get("values", [])
    return [
        {
            "open":   float(v["open"]),
            "high":   float(v["high"]),
            "low":    float(v["low"]),
            "close":  float(v["close"]),
            "volume": float(v.get("volume", 0)),
        }
        for v in reversed(values)  # oldest first
    ]

async def fetch_price(symbol: str) -> dict:
    url = f"https://api.twelvedata.com/quote?symbol={symbol}&apikey={TWELVE_KEY}"
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(url)
        r.raise_for_status()
        d = r.json()
    return {
        "price":  float(d.get("close", 0)),
        "open":   float(d.get("open", 0)),
        "change": float(d.get("change", 0)),
        "pct":    float(d.get("percent_change", 0)),
    }

# ── Quant engine ──────────────────────────────────────────────────────────────

def momentum_score(candles: list) -> float:
    """Time-series momentum across 5, 10, 20-bar windows."""
    closes = [c["close"] for c in candles]
    if len(closes) < 21:
        return 50.0
    def ret(n): return (closes[-1] - closes[-n]) / closes[-n] if closes[-n] else 0
    raw = ret(5)*0.5 + ret(10)*0.3 + ret(20)*0.2
    return min(100, max(0, 50 + raw * 500))

def zscore_score(candles: list) -> float:
    """Z-score mean reversion signal."""
    closes = [c["close"] for c in candles]
    if len(closes) < 21:
        return 50.0
    window = closes[-20:]
    mean = statistics.mean(window)
    std  = statistics.stdev(window)
    if std == 0:
        return 50.0
    z = (closes[-1] - mean) / std
    # z>0 → price above mean → bullish momentum; invert for extreme levels
    if abs(z) > 2:
        score = 50 - (z * 10)  # reversion signal
    else:
        score = 50 + (z * 15)
    return min(100, max(0, score))

def trend_efficiency(candles: list) -> tuple[float, str]:
    """Trend efficiency ratio → regime."""
    closes = [c["close"] for c in candles]
    if len(closes) < 11:
        return 0.5, "RANGING"
    net = abs(closes[-1] - closes[-11])
    path = sum(abs(closes[-i] - closes[-i-1]) for i in range(1, 11))
    er = net / path if path else 0
    regime = "TRENDING" if er > 0.55 else ("RANGING" if er < 0.35 else "MIXED")
    return er, regime

def vwap_score(candles: list) -> float:
    """VWAP institutional filter."""
    if len(candles) < 5:
        return 50.0
    total_vol = sum(c["volume"] for c in candles[-20:]) or 1
    vwap = sum(((c["high"]+c["low"]+c["close"])/3)*c["volume"] for c in candles[-20:]) / total_vol
    price = candles[-1]["close"]
    pct = (price - vwap) / vwap if vwap else 0
    return min(100, max(0, 50 + pct * 1000))

def volatility_regime(candles: list) -> tuple[str, float]:
    """Volatility regime classifier."""
    closes = [c["close"] for c in candles]
    if len(closes) < 31:
        return "NORMAL", 1.0
    def vol(n):
        rets = [abs(closes[-i]/closes[-i-1]-1) for i in range(1, n)]
        return statistics.mean(rets) if rets else 0
    v10 = vol(11)
    v30 = vol(31)
    ratio = v10 / v30 if v30 else 1
    if ratio > 1.8:
        return "EXTREME", 0.7
    elif ratio > 1.3:
        return "HIGH", 0.85
    elif ratio < 0.7:
        return "LOW", 1.2
    return "NORMAL", 1.0

def composite_score(candles: list) -> dict:
    """Sharpe-weighted composite of all models."""
    mom  = momentum_score(candles)
    zs   = zscore_score(candles)
    er, regime = trend_efficiency(candles)
    vwap = vwap_score(candles)
    vol_regime, vol_mult = volatility_regime(candles)

    if regime == "TRENDING":
        raw = mom*0.45 + vwap*0.25 + zs*0.15 + (er*100)*0.15
    elif regime == "RANGING":
        raw = zs*0.45 + vwap*0.25 + mom*0.20 + (er*100)*0.10
    else:
        raw = mom*0.35 + zs*0.30 + vwap*0.25 + (er*100)*0.10

    score = min(100, max(0, raw * vol_mult))

    if score >= 75:
        verdict, direction = "STRONG BUY", "LONG"
    elif score >= 60:
        verdict, direction = "BUY", "LONG"
    elif score <= 25:
        verdict, direction = "STRONG SELL", "SHORT"
    elif score <= 40:
        verdict, direction = "SELL", "SHORT"
    else:
        verdict, direction = "NEUTRAL", "FLAT"

    return {
        "score": round(score, 1),
        "verdict": verdict,
        "direction": direction,
        "regime": regime,
        "vol_regime": vol_regime,
        "momentum": round(mom, 1),
        "zscore": round(zs, 1),
        "trend_er": round(er, 3),
        "vwap": round(vwap, 1),
    }

def build_trade_setup(price: float, candles: list, score_data: dict) -> dict:
    """Generate entry/stop/target levels."""
    closes = [c["close"] for c in candles[-20:]]
    highs  = [c["high"]  for c in candles[-14:]]
    lows   = [c["low"]   for c in candles[-14:]]

    atr_vals = [c["high"] - c["low"] for c in candles[-14:]]
    atr = statistics.mean(atr_vals) if atr_vals else price * 0.01

    direction = score_data["direction"]

    if direction == "LONG":
        entry_low  = round(price * 0.9985, 2)
        entry_high = round(price * 1.0005, 2)
        stop       = round(price - atr * 1.5, 2)
        tp1        = round(price + atr * 2, 2)
        tp2        = round(price + atr * 3, 2)
        rr1 = round((tp1 - price) / (price - stop), 1) if price > stop else 0
        rr2 = round((tp2 - price) / (price - stop), 1) if price > stop else 0
    elif direction == "SHORT":
        entry_low  = round(price * 0.9995, 2)
        entry_high = round(price * 1.0015, 2)
        stop       = round(price + atr * 1.5, 2)
        tp1        = round(price - atr * 2, 2)
        tp2        = round(price - atr * 3, 2)
        rr1 = round((price - tp1) / (stop - price), 1) if stop > price else 0
        rr2 = round((price - tp2) / (stop - price), 1) if stop > price else 0
    else:
        return {"direction": "FLAT", "note": "No trade setup — market neutral"}

    confidence = min(99, max(50, int(score_data["score"])))

    return {
        "direction": direction,
        "entry_low":  entry_low,
        "entry_high": entry_high,
        "stop":  stop,
        "tp1":   tp1,
        "tp2":   tp2,
        "rr1":   rr1,
        "rr2":   rr2,
        "confidence": confidence,
        "atr": round(atr, 2),
    }

# ── Finnhub news ──────────────────────────────────────────────────────────────
async def fetch_news() -> list:
    url = f"https://finnhub.io/api/v1/news?category=forex&token={FINNHUB_KEY}"
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(url)
        r.raise_for_status()
        items = r.json()
    return [
        {
            "headline": i.get("headline", ""),
            "source":   i.get("source", ""),
            "url":      i.get("url", ""),
            "ts":       i.get("datetime", 0),
        }
        for i in items[:8]
        if i.get("headline")
    ]

# ── Main data builder ─────────────────────────────────────────────────────────
async def build_instrument_data(key: str) -> dict:
    info = INSTRUMENTS[key]
    symbol = info["symbol"]

    # Fetch price
    price_data = await fetch_price(symbol)

    # Fetch candles for all timeframes
    tf_results = []
    for tf, label in zip(TIMEFRAMES, TF_LABELS):
        try:
            candles = await fetch_candles(symbol, tf, 50)
            data = composite_score(candles)
            setup = build_trade_setup(price_data["price"], candles, data)
            tf_results.append({
                "label": label,
                "interval": tf,
                **data,
                "setup": setup,
            })
        except Exception as e:
            tf_results.append({
                "label": label,
                "interval": tf,
                "score": 50,
                "verdict": "N/A",
                "direction": "FLAT",
                "regime": "UNKNOWN",
                "error": str(e),
            })

    # Overall composite (average of all TFs, weighted toward longer)
    weights = [0.10, 0.20, 0.25, 0.30, 0.15]
    scores = [tf["score"] for tf in tf_results]
    overall = sum(s*w for s, w in zip(scores, weights))

    if overall >= 75:   overall_verdict = "STRONG BUY"
    elif overall >= 60: overall_verdict = "BUY"
    elif overall <= 25: overall_verdict = "STRONG SELL"
    elif overall <= 40: overall_verdict = "SELL"
    else:               overall_verdict = "NEUTRAL"

    # Use daily TF for main signals display
    daily = next((t for t in tf_results if t["label"] == "Daily"), tf_results[0])

    return {
        "key": key,
        "name": info["name"],
        "type": info["type"],
        "symbol": symbol,
        "price": price_data["price"],
        "change": price_data["change"],
        "pct": price_data["pct"],
        "overall_score": round(overall, 1),
        "overall_verdict": overall_verdict,
        "timeframes": tf_results,
        "signals": {
            "momentum":  daily.get("momentum", 50),
            "zscore":    daily.get("zscore", 50),
            "trend_er":  daily.get("trend_er", 0.5),
            "vwap":      daily.get("vwap", 50),
            "vol_regime": daily.get("vol_regime", "NORMAL"),
            "regime":    daily.get("regime", "MIXED"),
        },
        "setup": daily.get("setup", {}),
        "updated": datetime.now(timezone.utc).isoformat(),
    }

# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/api/scores")
async def get_scores(user: str = Depends(verify_credentials)):
    cached = cache_get("scores")
    if cached:
        return cached

    try:
        gold, us30 = await asyncio.gather(
            build_instrument_data("gold"),
            build_instrument_data("us30"),
        )
        result = {"gold": gold, "us30": us30, "ts": time.time()}
        cache_set("scores", result)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/news")
async def get_news(user: str = Depends(verify_credentials)):
    cached = cache_get("news")
    if cached:
        return cached
    try:
        news = await fetch_news()
        result = {"items": news, "ts": time.time()}
        cache_set("news", result)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok", "ts": time.time()}

@app.get("/")
async def root():
    """Serve the landing page / dashboard."""
    return FileResponse("index.html")
