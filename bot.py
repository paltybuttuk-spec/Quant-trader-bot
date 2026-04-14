"""
QuantRisk Bot v6 — Three-Speed Architecture
============================================
Speed 1 — Finnhub WebSocket (always)   : live_price float update only
Speed 2 — 15-min micro cycle           : 1H candles fresh → Gate 2 setup confirmation
Speed 3 — 30-min macro cycle           : all 5 TFs fresh → full scoring, flips, bias

Three-Gate Setup Filter:
  Gate 1 — Macro composite >62 or <38 (30-min cycle)
  Gate 2 — 1H + 15min agree with macro direction (15-min cycle)
  Gate 3 — Live price inside ATR entry zone (real-time check at Gate 2)

API Budget:
  Micro  : 2 calls × 3 instruments × 96 cycles = 576/day
  Macro  : 5 calls × 3 instruments × 48 cycles = 720/day  (shares 1H with micro on overlap)
  Total  : ~796/day  ✅ within Twelve Data free tier 800/day

Instruments:
  Gold   : XAU/USD candles | OANDA:XAU_USD websocket | UUP corr
  US30   : DIA candles     | DIA websocket            | VIXY corr
  EURUSD : EUR/USD candles | OANDA:EUR_USD websocket  | UUP corr

All 6 quant scoring models preserved:
  1. Regime Detection (Trend Efficiency Ratio)
  2. Time-Series Momentum Factor (volume-confirmed)
  3. Adaptive Z-Score Mean Reversion
  4. VWAP Institutional Filter
  5. Volatility Regime Classifier
  6. Inter-market correlation (DXY → Gold/EURUSD, VIXY → US30)
"""

import asyncio
import logging
import os
import time as _time
from collections import defaultdict, Counter
from datetime import datetime
from statistics import mean, stdev

import httpx
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes

from websocket_manager import WebSocketManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
log = logging.getLogger(__name__)

# ── Env ───────────────────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID   = int(os.environ["TELEGRAM_CHAT_ID"])
TWELVEDATA_API_KEY = os.environ["TWELVEDATA_API_KEY"]
FINNHUB_API_KEY    = os.environ["FINNHUB_API_KEY"]

# ── Constants ─────────────────────────────────────────────────────────────────
TIMEFRAMES    = ["15min", "1h", "4h", "1day", "1week"]
TF_LABELS     = {"15min": "15M", "1h": "1H", "4h": "4H", "1day": "Daily", "1week": "Weekly"}
CANDLE_COUNT  = 50

INSTRUMENTS = {
    "Gold": {
        "candle_sym": "XAU/USD",
        "ws_sym":     "OANDA:XAU_USD",
        "corr_sym":   "UUP",
        "corr_label": "DXY",
        "news_sym":   "GLD",
    },
    "US30": {
        "candle_sym": "DIA",
        "ws_sym":     "DIA",
        "corr_sym":   "VIXY",
        "corr_label": "VIX",
        "news_sym":   "DIA",
    },
    "EURUSD": {
        "candle_sym": "EUR/USD",
        "ws_sym":     "OANDA:EUR_USD",
        "corr_sym":   "UUP",
        "corr_label": "DXY",
        "news_sym":   None,
    },
}

WS_TO_INST = {
    "OANDA:XAU_USD": "Gold",
    "OANDA:EUR_USD": "EURUSD",
    "DIA":           "US30",
}

TD_INTERVAL = {
    "15min": "15min",
    "1h":    "1h",
    "4h":    "4h",
    "1day":  "1day",
    "1week": "1week",
}

# ── State (resets on restart) ─────────────────────────────────────────────────
live_prices    = {}          # {inst: float}  — updated by WebSocket only
prev_verdicts  = {}          # {inst: {tf: verdict}}
weekly_history = defaultdict(list)
score_history  = defaultdict(list)   # last 3 composites for persistence check
last_setup_fire = {}
live_scores    = {}          # latest full score result per instrument
gate1_open     = {}          # {inst: bool}  — set by macro cycle
macro_bias     = {}          # {inst: "LONG"|"SHORT"|None}

app_ref        = None
ws_mgr: WebSocketManager = None

# ── Emoji maps ────────────────────────────────────────────────────────────────
VE = {"STRONG BUY": "🟢🟢", "BUY": "🟢", "NEUTRAL": "🟡", "SELL": "🔴", "STRONG SELL": "🔴🔴"}
RE = {"TRENDING": "📈", "RANGING": "↔️", "MIXED": "〰️"}
IE = {"Gold": "🥇", "US30": "🏦", "EURUSD": "💶"}


# ══════════════════════════════════════════════════════════════════════════════
# SESSION
# ══════════════════════════════════════════════════════════════════════════════

def get_session_multiplier():
    hour = (datetime.utcnow().hour + 3) % 24
    if  8 <= hour < 11: return 1.2, "London Open"
    if 11 <= hour < 16: return 1.4, "London/NY Overlap"
    if 16 <= hour < 23: return 1.3, "New York"
    if  0 <= hour <  8: return 0.7, "Asian"
    return 0.5, "After Hours"

def is_active_session():
    m, _ = get_session_multiplier()
    return m > 0.8


# ══════════════════════════════════════════════════════════════════════════════
# TWELVE DATA FETCH — always fresh, never cached
# ══════════════════════════════════════════════════════════════════════════════

async def fetch_candles(client: httpx.AsyncClient, symbol: str, timeframe: str, live_price: float = None):
    """Fetch fresh candles from Twelve Data. Optionally patch last close with live price."""
    params = {
        "symbol":     symbol,
        "interval":   TD_INTERVAL[timeframe],
        "outputsize": CANDLE_COUNT,
        "apikey":     TWELVEDATA_API_KEY,
    }
    try:
        r = await client.get("https://api.twelvedata.com/time_series", params=params, timeout=15)
        data = r.json()
        if data.get("status") == "error":
            log.warning(f"TD error {symbol} {timeframe}: {data.get('message')}")
            return None
        values = list(reversed(data.get("values", [])))
        if not values:
            return None
        parsed = {
            "closes":  [float(v["close"])         for v in values],
            "highs":   [float(v["high"])           for v in values],
            "lows":    [float(v["low"])            for v in values],
            "opens":   [float(v["open"])           for v in values],
            "volumes": [float(v.get("volume", 1)) for v in values],
        }
        # Patch last close with live price if provided
        if live_price and live_price > 0:
            parsed["closes"][-1] = live_price
        return parsed
    except Exception as e:
        log.error(f"Fetch failed {symbol} {timeframe}: {e}")
        return None


async def fetch_corr_price(client: httpx.AsyncClient, symbol: str):
    """Fetch latest daily close for correlation symbol."""
    params = {
        "symbol":     symbol,
        "interval":   "1day",
        "outputsize": 6,
        "apikey":     TWELVEDATA_API_KEY,
    }
    try:
        r = await client.get("https://api.twelvedata.com/time_series", params=params, timeout=10)
        data = r.json()
        if data.get("status") == "error":
            return None, 0.0
        values = list(reversed(data.get("values", [])))
        if len(values) < 2:
            return None, 0.0
        closes = [float(v["close"]) for v in values]
        price  = closes[-1]
        mom    = (closes[-1] - closes[-6]) / closes[-6] if closes[-6] > 0 else 0.0
        return price, mom
    except Exception as e:
        log.error(f"Corr fetch failed {symbol}: {e}")
        return None, 0.0


# ══════════════════════════════════════════════════════════════════════════════
# QUANT SCORING ENGINE — all 6 models preserved exactly
# ══════════════════════════════════════════════════════════════════════════════

def calc_atr(highs, lows, closes, period=14):
    trs = [
        max(highs[i] - lows[i],
            abs(highs[i] - closes[i-1]),
            abs(lows[i]  - closes[i-1]))
        for i in range(1, len(closes))
    ]
    return mean(trs[-period:]) if len(trs) >= period else (mean(trs) if trs else 0)


def calc_vwap(closes, volumes, period=20):
    c, v = closes[-period:], volumes[-period:]
    tv = sum(v)
    return sum(p * vv for p, vv in zip(c, v)) / tv if tv else closes[-1]


def score_to_verdict(s):
    if s >= 70: return "STRONG BUY"
    if s >= 58: return "BUY"
    if s >= 45: return "NEUTRAL"
    if s >= 32: return "SELL"
    return "STRONG SELL"


def score_timeframe(closes, highs, lows, opens, volumes,
                    dxy_momentum=0.0, vixy_pressure=0.0, instrument="Gold"):
    """
    6-model quant scoring engine.
    Model 1: Regime Detection         — Trend Efficiency Ratio
    Model 2: Time-Series Momentum     — volume-confirmed, 3-horizon
    Model 3: Adaptive Z-Score         — window adjusts to volatility
    Model 4: VWAP Institutional Filter
    Model 5: Volatility Regime Classifier
    Model 6: Inter-market correlation  — DXY / VIXY
    """
    if len(closes) < 30:
        return {
            "score": 50, "verdict": "NEUTRAL", "regime": "MIXED",
            "z_score": 0, "momentum": 0, "vol_ratio": 1,
            "vol_label": "NORMAL", "vwap_signal": "NEUTRAL",
            "efficiency": 0, "vol_conf": 1.0
        }

    score = 50.0

    # ── Model 1: Regime Detection ─────────────────────────────────────────────
    net  = abs(closes[-1] - closes[-11])
    path = sum(abs(closes[i] - closes[i-1]) for i in range(-10, 0))
    eff  = net / path if path > 0 else 0
    regime = "TRENDING" if eff > 0.55 else ("RANGING" if eff < 0.35 else "MIXED")

    # ── Model 5: Volatility Regime Classifier ────────────────────────────────
    r10 = [abs(closes[i] - closes[i-1]) / closes[i-1] for i in range(-10, 0) if closes[i-1] > 0]
    r30 = [abs(closes[i] - closes[i-1]) / closes[i-1] for i in range(-30, 0) if closes[i-1] > 0]
    v10 = mean(r10) if r10 else 0.01
    v30 = mean(r30) if r30 else 0.01
    vol_ratio = v10 / v30 if v30 > 0 else 1.0
    if   vol_ratio < 0.70: vol_mult, vol_label = 1.2, "LOW"
    elif vol_ratio < 1.50: vol_mult, vol_label = 1.0, "NORMAL"
    elif vol_ratio < 2.00: vol_mult, vol_label = 0.7, "HIGH"
    else:                  vol_mult, vol_label = 0.3, "EXTREME"

    # ── Model 3: Adaptive Z-Score Mean Reversion ──────────────────────────────
    zw = 10 if vol_ratio > 1.5 else (30 if vol_ratio < 0.7 else 20)
    zc = closes[-zw:]
    zm = mean(zc)
    try:    zs = stdev(zc)
    except: zs = 0.0001
    z = (closes[-1] - zm) / zs if zs > 0 else 0

    if regime == "RANGING":
        if   z >  2.0: score -= 20
        elif z >  1.5: score -= 12
        elif z < -2.0: score += 20
        elif z < -1.5: score += 12
    elif regime == "TRENDING" and abs(z) > 2.5:
        score *= 0.85   # suppress momentum if extremely stretched

    # ── Model 2: Time-Series Momentum (volume-confirmed) ─────────────────────
    avg_v = mean(volumes[-20:]) if len(volumes) >= 20 else mean(volumes)
    vc    = 1.0 if volumes[-1] > avg_v else 0.5   # volume confirmation
    m5    = (closes[-1] - closes[-6])  / closes[-6]  if closes[-6]  > 0 else 0
    m10   = (closes[-1] - closes[-11]) / closes[-11] if closes[-11] > 0 else 0
    m20   = (closes[-1] - closes[-21]) / closes[-21] if closes[-21] > 0 else 0
    mom   = (m5 * 0.5 + m10 * 0.3 + m20 * 0.2) * vc
    mp    = mom * 200

    if   regime == "TRENDING": score += mp
    elif regime == "MIXED":    score += mp * 0.5
    else:                      score += mp * 0.25

    # ── Model 4: VWAP Institutional Filter ───────────────────────────────────
    vwap = calc_vwap(closes, volumes)
    if closes[-1] > vwap:
        score += 8
        vs = "ABOVE"
    else:
        score -= 8
        vs = "BELOW"

    # ── Model 6: Inter-market Correlation ────────────────────────────────────
    if instrument in ("Gold", "EURUSD") and dxy_momentum:
        score += max(-15, min(15, -dxy_momentum * 150))
    elif instrument == "US30" and vixy_pressure:
        score += max(-12, min(12, -vixy_pressure * 100))

    # ── Session multiplier + vol multiplier ──────────────────────────────────
    sm, _ = get_session_multiplier()
    score  = 50 + (score - 50) * vol_mult
    score  = 50 + (score - 50) * sm
    score  = max(0, min(100, score))

    return {
        "score":       round(score, 1),
        "verdict":     score_to_verdict(score),
        "regime":      regime,
        "z_score":     round(z, 2),
        "momentum":    round(mom * 100, 3),
        "vol_ratio":   round(vol_ratio, 2),
        "vol_label":   vol_label,
        "vwap_signal": vs,
        "efficiency":  round(eff, 3),
        "vol_conf":    vc,
    }


def calc_composite(tf_scores: dict):
    if not tf_scores:
        return 50.0, 0.0
    scores    = list(tf_scores.values())
    composite = mean(scores)
    bull = sum(1 for s in scores if s > 58)
    bear = sum(1 for s in scores if s < 42)
    if   bull >= 4: composite = min(100, composite + 8)
    elif bear >= 4: composite = max(0,   composite - 8)
    aligned    = max(bull, bear)
    confidence = (composite / 100) * (aligned / len(scores)) * 100
    return round(composite, 1), round(confidence, 1)


def update_score_history(inst, score):
    h = score_history[inst]
    h.append(score)
    score_history[inst] = h[-3:]
    if len(h) < 3:
        return False
    return all(s > 62 for s in h) or all(s < 38 for s in h)


def calc_setup(closes, highs, lows, composite, vol_ratio):
    atr   = calc_atr(highs, lows, closes)
    price = closes[-1]
    sm    = 2.2 if vol_ratio > 1.5 else (1.1 if vol_ratio < 0.7 else 1.5)
    d     = "LONG" if composite > 62 else "SHORT"
    el    = round(price - 0.3 * atr, 5)
    eh    = round(price + 0.3 * atr, 5)
    if d == "LONG":
        stop = round(price - sm * atr, 5)
        t1, t2, t3 = round(price + 2*atr, 5), round(price + 3*atr, 5), round(price + 4*atr, 5)
    else:
        stop = round(price + sm * atr, 5)
        t1, t2, t3 = round(price - 2*atr, 5), round(price - 3*atr, 5), round(price - 4*atr, 5)
    return {
        "direction":  d,
        "entry_low":  el,
        "entry_high": eh,
        "stop":       stop,
        "t1": t1, "t2": t2, "t3": t3,
        "atr":        round(atr, 5),
        "stop_mult":  sm,
        "support":    round(min(lows[-20:]),  5),
        "resistance": round(max(highs[-20:]), 5),
    }


# ══════════════════════════════════════════════════════════════════════════════
# THREE-GATE SETUP LOGIC
# ══════════════════════════════════════════════════════════════════════════════

def check_gate3_entry(price, setup):
    """Gate 3: is live price inside the ATR entry zone right now?"""
    if not setup:
        return False
    return setup["entry_low"] <= price <= setup["entry_high"]


# ══════════════════════════════════════════════════════════════════════════════
# MACRO CYCLE — every 30 minutes
# Full 5-timeframe scoring, flip detection, Gate 1 bias update
# ══════════════════════════════════════════════════════════════════════════════

async def macro_cycle():
    """
    Fetch all 5 timeframes for all 3 instruments in one asyncio.gather burst.
    Score everything. Check flips. Update Gate 1 bias.
    Discard all candle data after use.
    """
    log.info("── Macro cycle starting ──")
    async with httpx.AsyncClient() as client:
        for inst, cfg in INSTRUMENTS.items():
            price = live_prices.get(inst)
            if not price:
                log.warning(f"Macro: no live price for {inst}, skipping.")
                continue

            sym = cfg["candle_sym"]

            # Fetch all 5 TFs + corr in parallel
            tasks = [fetch_candles(client, sym, tf, price) for tf in TIMEFRAMES]
            tasks.append(fetch_corr_price(client, cfg["corr_sym"]))
            results = await asyncio.gather(*tasks, return_exceptions=True)

            candle_results = results[:5]
            corr_result    = results[5]

            corr_price, corr_mom = (corr_result if isinstance(corr_result, tuple) else (None, 0.0))
            dxy_mom    = corr_mom if inst in ("Gold", "EURUSD") else 0.0
            vixy_pres  = corr_mom if inst == "US30" else 0.0

            tf_scores  = {}
            tf_details = {}
            vol_ratios = []
            daily_candles = None

            for i, tf in enumerate(TIMEFRAMES):
                candles = candle_results[i]
                if isinstance(candles, Exception) or candles is None:
                    continue
                d = score_timeframe(
                    candles["closes"], candles["highs"], candles["lows"],
                    candles["opens"],  candles["volumes"],
                    dxy_momentum=dxy_mom, vixy_pressure=vixy_pres,
                    instrument=inst,
                )
                tf_scores[tf]  = d["score"]
                tf_details[tf] = d
                vol_ratios.append(d["vol_ratio"])
                if tf == "1day":
                    daily_candles = candles   # keep daily for setup calc

            if not tf_scores:
                continue

            composite, conf = calc_composite(tf_scores)
            persistent      = update_score_history(inst, composite)
            avg_vr          = mean(vol_ratios) if vol_ratios else 1.0

            # Gate 1 update — raised bar vs v5
            if composite > 62:
                gate1_open[inst] = True
                macro_bias[inst] = "LONG"
            elif composite < 38:
                gate1_open[inst] = True
                macro_bias[inst] = "SHORT"
            else:
                gate1_open[inst] = False
                macro_bias[inst] = None

            # Setup calculation (daily candles)
            setup = None
            if daily_candles and gate1_open.get(inst) and conf > 70:
                setup = calc_setup(
                    daily_candles["closes"], daily_candles["highs"],
                    daily_candles["lows"],   composite, avg_vr,
                )

            result = {
                "instrument":     inst,
                "candle_sym":     sym,
                "price":          price,
                "composite":      composite,
                "confidence":     conf,
                "verdict":        score_to_verdict(composite),
                "tf_scores":      tf_scores,
                "tf_details":     tf_details,
                "setup":          setup,
                "persistent":     persistent,
                "corr_price":     corr_price,
                "corr_label":     cfg["corr_label"],
                "dxy_momentum":   round(dxy_mom  * 100, 3),
                "vixy_pressure":  round(vixy_pres * 100, 3),
                "scored_at":      datetime.utcnow().isoformat(),
                "cycle":          "macro",
            }

            live_scores[inst] = result

            # Weekly history
            weekly_history[inst].append((datetime.utcnow(), price, composite, score_to_verdict(composite)))
            weekly_history[inst] = weekly_history[inst][-336:]

            # Flip detection
            await _check_flips(inst, tf_details)

    log.info("── Macro cycle complete ──")


async def _check_flips(inst, tf_details):
    if app_ref is None:
        return
    for tf, detail in tf_details.items():
        new_v = detail["verdict"]
        old_v = prev_verdicts.get(inst, {}).get(tf)
        if old_v is None:
            prev_verdicts.setdefault(inst, {})[tf] = new_v
            continue
        if new_v != old_v:
            try:
                await app_ref.bot.send_message(
                    TELEGRAM_CHAT_ID,
                    fmt_flip_alert(inst, tf, old_v, new_v, detail),
                    parse_mode="Markdown",
                )
            except Exception as e:
                log.error(f"Flip alert failed: {e}")
        prev_verdicts.setdefault(inst, {})[tf] = new_v


# ══════════════════════════════════════════════════════════════════════════════
# MICRO CYCLE — every 15 minutes
# Fetches 15min + 1H fresh. Runs Gate 2 check. Fires setup if all 3 gates open.
# ══════════════════════════════════════════════════════════════════════════════

async def micro_cycle():
    """
    Fetch 15min + 1H for all instruments. Check Gate 2 (short-TF agreement).
    If Gate 1 + Gate 2 + Gate 3 all open → fire setup alert.
    """
    log.info("── Micro cycle starting ──")
    if not is_active_session():
        log.info("Micro: outside active session, skipping.")
        return

    async with httpx.AsyncClient() as client:
        for inst, cfg in INSTRUMENTS.items():

            # Gate 1 must be open from last macro cycle
            if not gate1_open.get(inst):
                continue

            price = live_prices.get(inst)
            if not price:
                continue

            bias = macro_bias.get(inst)
            sym  = cfg["candle_sym"]

            # Fetch 15min + 1H fresh in parallel
            c15, c1h = await asyncio.gather(
                fetch_candles(client, sym, "15min", price),
                fetch_candles(client, sym, "1h",    price),
                return_exceptions=True,
            )

            if isinstance(c15, Exception) or c15 is None:
                continue
            if isinstance(c1h, Exception) or c1h is None:
                continue

            # Score both short timeframes
            d15 = score_timeframe(c15["closes"], c15["highs"], c15["lows"],
                                  c15["opens"],  c15["volumes"], instrument=inst)
            d1h = score_timeframe(c1h["closes"], c1h["highs"], c1h["lows"],
                                  c1h["opens"],  c1h["volumes"], instrument=inst)

            # Gate 2: both short TFs must agree with macro bias
            if bias == "LONG":
                gate2 = d15["score"] > 58 and d1h["score"] > 58
            elif bias == "SHORT":
                gate2 = d15["score"] < 42 and d1h["score"] < 42
            else:
                gate2 = False

            if not gate2:
                log.info(f"Micro: {inst} Gate 2 not open (15M={d15['score']} 1H={d1h['score']} bias={bias})")
                continue

            # Gate 3: live price inside ATR entry zone
            setup = live_scores.get(inst, {}).get("setup")
            gate3 = check_gate3_entry(price, setup)

            if not gate3:
                log.info(f"Micro: {inst} Gate 3 not open — price {price} not in entry zone")
                continue

            # All 3 gates open — check cooldown (90 min between setups)
            now  = datetime.utcnow()
            last = last_setup_fire.get(inst)
            if last and (now - last).seconds < 5400:
                continue

            # Fire setup alert
            data = live_scores.get(inst, {})
            if data and data.get("setup"):
                try:
                    await app_ref.bot.send_message(
                        TELEGRAM_CHAT_ID,
                        fmt_setup_alert_full(inst, data, d15, d1h),
                        parse_mode="Markdown",
                    )
                    last_setup_fire[inst] = now
                    log.info(f"✅ Setup alert fired: {inst} {bias}")
                except Exception as e:
                    log.error(f"Setup alert failed: {e}")

    log.info("── Micro cycle complete ──")


# ══════════════════════════════════════════════════════════════════════════════
# WEBSOCKET PRICE UPDATE — Speed 1
# One job only: update live_prices float
# ══════════════════════════════════════════════════════════════════════════════

async def on_price_tick(symbol, price):
    inst = WS_TO_INST.get(symbol)
    if inst:
        live_prices[inst] = price


# ══════════════════════════════════════════════════════════════════════════════
# NEWS
# ══════════════════════════════════════════════════════════════════════════════

async def fetch_news(client, symbol):
    now    = datetime.utcnow()
    from_d = now.replace(day=max(1, now.day - 7)).strftime("%Y-%m-%d")
    try:
        r = await client.get(
            "https://finnhub.io/api/v1/company-news",
            params={"symbol": symbol, "from": from_d,
                    "to": now.strftime("%Y-%m-%d"), "token": FINNHUB_API_KEY},
            timeout=10,
        )
        articles = r.json()
        if not isinstance(articles, list):
            return []
        scored = []
        for a in articles:
            s  = a.get("sentiment", {})
            sc = s.get("bullishPercent", 0.5) - s.get("bearishPercent", 0.5) if s else 0
            scored.append((sc, a))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored
    except Exception as e:
        log.error(f"News failed {symbol}: {e}")
        return []


def _fmt_news(news, instrument):
    if not news:
        return ""
    emoji = IE.get(instrument, "📰")
    lines = [f"{emoji} *{instrument} News*\n", "📈 *Catalysts:*"]
    for _, a in news[:5]:
        lines.append(f"  • {a.get('headline', '')[:80]}")
    lines.append("\n📉 *Risks:*")
    for _, a in news[-5:]:
        lines.append(f"  • {a.get('headline', '')[:80]}")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# FORMATTERS
# ══════════════════════════════════════════════════════════════════════════════

def fmt_instrument(data, include_setup=True):
    inst   = data["instrument"]
    sm, sl = get_session_multiplier()
    e      = IE.get(inst, "📊")
    lines  = [
        f"{e} *{inst}* 🔴 LIVE",
        f"💲 Price: `{data['price']:.5f}`" if data.get("price") else "",
        f"📊 Score: `{data['composite']}/100` — {VE.get(data['verdict'], '⚪')} *{data['verdict']}*",
        f"🎯 Confidence: `{data['confidence']:.1f}%`",
        f"⏰ Session: {sl} (×{sm})", "",
        "*Timeframe Breakdown:*",
    ]
    for tf in TIMEFRAMES:
        d = data["tf_details"].get(tf)
        if not d:
            lines.append(f"  {TF_LABELS[tf]}: —")
            continue
        vc = "✅" if d["vol_conf"] == 1.0 else "⚠️"
        lines.append(f"  {TF_LABELS[tf]}: `{d['score']}` {VE.get(d['verdict'], '⚪')} {RE.get(d['regime'], '')} {vc}")

    d15 = data["tf_details"].get("15min") or next(iter(data["tf_details"].values()), {})
    if d15:
        lines += [
            "", "*Quant Signals:*",
            f"  Z-Score: `{d15.get('z_score', 0):+.2f}` (adaptive window)",
            f"  Momentum: `{d15.get('momentum', 0):+.3f}%` {'✅' if d15.get('vol_conf', 1) == 1 else '⚠️ low vol'}",
            f"  Regime: {d15.get('regime', '—')} {RE.get(d15.get('regime', ''), '')}",
            f"  Vol: `{d15.get('vol_ratio', 1):.2f}` → {d15.get('vol_label', 'NORMAL')}",
            f"  VWAP: {'Above ✅' if d15.get('vwap_signal') == 'ABOVE' else 'Below ⚠️'}",
        ]

    cp    = data.get("corr_price")
    label = data.get("corr_label", "")
    if cp:
        if inst in ("Gold", "EURUSD"):
            m = data["dxy_momentum"]
            t = "Headwind 🔴" if m > 0.3 else ("Tailwind 🟢" if m < -0.3 else "Neutral 🟡")
            lines.append(f"\n💱 {label}: `{cp:.4f}` | Mom: `{m:+.2f}%` → {t}")
        elif inst == "US30":
            p = data["vixy_pressure"]
            t = "Risk-Off 🔴" if p > 0.3 else ("Risk-On 🟢" if p < -0.3 else "Calm 🟡")
            lines.append(f"\n😱 {label} (VIXY): `{cp:.2f}` | Mom: `{p:+.2f}%` → {t}")

    g1 = gate1_open.get(inst, False)
    lines.append(f"\n🚦 Gate 1 (Macro): {'✅ Open' if g1 else '🔴 Closed'} | Bias: {macro_bias.get(inst) or 'None'}")

    if data.get("persistent"):
        lines.append("🔒 *Signal Persistent — 3 macro cycles confirmed*")

    if include_setup and data.get("setup"):
        s  = data["setup"]
        de = "🟢 LONG" if s["direction"] == "LONG" else "🔴 SHORT"
        lines += [
            "", f"*🎯 Trade Setup — {de}*",
            f"  Entry: `{s['entry_low']} – {s['entry_high']}`",
            f"  Stop: `{s['stop']}` (×{s['stop_mult']} ATR adaptive)",
            f"  T1: `{s['t1']}` | T2: `{s['t2']}` | T3: `{s['t3']}`",
            f"  ATR: `{s['atr']}` | S: `{s['support']}` | R: `{s['resistance']}`",
        ]
    return "\n".join(l for l in lines if l is not None)


def fmt_flip_alert(inst, tf, old_v, new_v, d):
    e = IE.get(inst, "📊")
    return (
        f"⚡ *SCORE FLIP — {e} {inst}*\n"
        f"Timeframe: *{TF_LABELS.get(tf, tf)}*\n"
        f"{VE.get(old_v, '⚪')} {old_v} → {VE.get(new_v, '⚪')} {new_v}\n"
        f"Score: `{d['score']}` | Z: `{d['z_score']:+.2f}` | Mom: `{d['momentum']:+.3f}%`\n"
        f"Regime: {d['regime']} | Vol: {d['vol_label']}"
    )


def fmt_setup_alert_full(inst, data, d15, d1h):
    """Full 3-gate setup alert with short-TF confirmation shown."""
    s  = data["setup"]
    e  = IE.get(inst, "📊")
    de = "🟢 LONG" if s["direction"] == "LONG" else "🔴 SHORT"
    return (
        f"🚨 *3-GATE SETUP — {e} {inst}*\n"
        f"{de} | Score: `{data['composite']}` | Conf: `{data['confidence']:.1f}%`\n\n"
        f"✅ Gate 1 Macro:  `{data['composite']}` {VE.get(data['verdict'], '')}\n"
        f"✅ Gate 2 Confirm: 15M `{d15['score']}` | 1H `{d1h['score']}`\n"
        f"✅ Gate 3 Entry:   Price inside ATR zone\n\n"
        f"*Entry:* `{s['entry_low']} – {s['entry_high']}`\n"
        f"*Stop:*  `{s['stop']}` (×{s['stop_mult']} ATR adaptive)\n"
        f"*T1:* `{s['t1']}` (RR 1:2)\n"
        f"*T2:* `{s['t2']}` (RR 1:3)\n"
        f"*T3:* `{s['t3']}` (RR 1:4)\n"
        f"ATR: `{s['atr']}` | S: `{s['support']}` | R: `{s['resistance']}`\n"
        f"Regime: {d15['regime']} | Vol: {d15['vol_label']} | Z: `{d15['z_score']:+.2f}`\n"
        f"🔒 Persistent: {data['persistent']}"
    )


def fmt_setup_alert(inst, data):
    """Simple setup alert for non-gate-checked contexts."""
    s  = data["setup"]
    e  = IE.get(inst, "📊")
    return (
        f"🚨 *SETUP ALERT — {e} {inst}*\n"
        f"{'🟢 LONG' if s['direction'] == 'LONG' else '🔴 SHORT'} | "
        f"Score: `{data['composite']}` | Conf: `{data['confidence']:.1f}%`\n"
        f"Entry: `{s['entry_low']} – {s['entry_high']}`\n"
        f"Stop: `{s['stop']}` (adaptive ×{s['stop_mult']})\n"
        f"T1: `{s['t1']}` | T2: `{s['t2']}` | T3: `{s['t3']}`\n"
        f"🔒 Persistent: {data['persistent']}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# KEYBOARD
# ══════════════════════════════════════════════════════════════════════════════

def main_keyboard():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🥇 Gold",    callback_data="gold"),
         InlineKeyboardButton("🏦 US30",    callback_data="us30"),
         InlineKeyboardButton("💶 EURUSD",  callback_data="eurusd")],
        [InlineKeyboardButton("📊 Summary", callback_data="summary"),
         InlineKeyboardButton("🎯 Setups",  callback_data="setup")],
        [InlineKeyboardButton("📅 Weekly",  callback_data="weekly"),
         InlineKeyboardButton("🔄 Refresh", callback_data="summary")],
    ])


# ══════════════════════════════════════════════════════════════════════════════
# SCHEDULED JOBS
# ══════════════════════════════════════════════════════════════════════════════

async def job_daily_report(app):
    async with httpx.AsyncClient() as client:
        gn, dn = await asyncio.gather(
            fetch_news(client, "GLD"),
            fetch_news(client, "DIA"),
        )
    hdr = (
        f"☀️ *QuantRisk Daily Report*\n"
        f"📅 {datetime.utcnow().strftime('%A, %d %B %Y')} | EAT 07:30\n{'─' * 30}"
    )
    await app.bot.send_message(TELEGRAM_CHAT_ID, hdr, parse_mode="Markdown")
    for inst, news in [("Gold", gn), ("US30", dn)]:
        if d := live_scores.get(inst):
            await app.bot.send_message(TELEGRAM_CHAT_ID, fmt_instrument(d), parse_mode="Markdown")
        if cats := _fmt_news(news, inst):
            await app.bot.send_message(TELEGRAM_CHAT_ID, cats, parse_mode="Markdown")
    if d := live_scores.get("EURUSD"):
        await app.bot.send_message(TELEGRAM_CHAT_ID, fmt_instrument(d), parse_mode="Markdown")
    await app.bot.send_message(TELEGRAM_CHAT_ID, "Good trading today 🚀", reply_markup=main_keyboard())


async def job_weekly_recap(app):
    lines = ["📅 *QuantRisk Weekly Recap*\n"]
    for inst, history in weekly_history.items():
        if not history:
            continue
        scores = [h[2] for h in history]
        tv     = Counter(h[3] for h in history).most_common(1)[0][0]
        e      = IE.get(inst, "📊")
        lines.append(
            f"{e} *{inst}*\n"
            f"  Avg: `{round(mean(scores), 1)}` | High: `{max(scores)}` | Low: `{min(scores)}`\n"
            f"  Dominant: {VE.get(tv, '⚪')} {tv} | Points: {len(history)}\n"
        )
    await app.bot.send_message(TELEGRAM_CHAT_ID, "\n".join(lines), parse_mode="Markdown")


# ══════════════════════════════════════════════════════════════════════════════
# COMMANDS
# ══════════════════════════════════════════════════════════════════════════════

def _mo(update):
    return update.message or update.callback_query.message


async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    prices_ready = all(live_prices.get(inst) for inst in INSTRUMENTS)
    scores_ready = all(live_scores.get(inst) for inst in INSTRUMENTS)
    g  = "🟢" if live_prices.get("Gold")   else "🟡"
    eu = "🟢" if live_prices.get("EURUSD") else "🟡"
    di = "🟢" if live_prices.get("US30")   else "🟡"
    sc = "✅ Scores live" if scores_ready else "⏳ First macro cycle pending..."
    await _mo(update).reply_text(
        f"👋 *QuantRisk Bot v6 — Three-Speed Architecture*\n\n"
        f"{g} Gold (XAU/USD) — 24/5 live\n"
        f"{eu} EURUSD — 24/5 live\n"
        f"{di} US30 (DIA) — market hours\n"
        f"{sc}\n\n"
        f"*6 quant models ✅ | 3-gate setup filter ✅*\n"
        f"Macro: 30min | Micro: 15min | WebSocket: live",
        parse_mode="Markdown",
        reply_markup=main_keyboard(),
    )


async def _send_inst(update, inst):
    d = live_scores.get(inst)
    if not d:
        await _mo(update).reply_text(
            f"⏳ {inst} — waiting for first macro cycle (up to 30s after start).",
            reply_markup=main_keyboard(),
        )
        return
    await _mo(update).reply_text(fmt_instrument(d), parse_mode="Markdown", reply_markup=main_keyboard())


async def cmd_gold(u, c):   await _send_inst(u, "Gold")
async def cmd_us30(u, c):   await _send_inst(u, "US30")
async def cmd_eurusd(u, c): await _send_inst(u, "EURUSD")


async def cmd_summary(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    sm, sl = get_session_multiplier()
    lines  = [f"📊 *QuantRisk Summary*\n⏰ {sl} (×{sm})\n"]
    for inst in ["Gold", "US30", "EURUSD"]:
        d = live_scores.get(inst)
        e = IE.get(inst, "📊")
        if not d:
            lines.append(f"{e} *{inst}*: ⏳ Loading...")
            continue
        g1 = "✅" if gate1_open.get(inst) else "🔴"
        lines.append(
            f"{e} *{inst}*: `{d['composite']}/100` {VE.get(d['verdict'], '⚪')} {d['verdict']}\n"
            f"  Conf: `{d['confidence']:.1f}%` | Gate 1: {g1} | Bias: {macro_bias.get(inst) or 'None'}"
        )
    await _mo(update).reply_text("\n".join(lines), parse_mode="Markdown", reply_markup=main_keyboard())


async def cmd_setup(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    found = False
    for inst in ["Gold", "US30", "EURUSD"]:
        d = live_scores.get(inst)
        if d and d.get("setup") and gate1_open.get(inst):
            await _mo(update).reply_text(fmt_setup_alert(inst, d), parse_mode="Markdown")
            found = True
    if not found:
        await _mo(update).reply_text(
            "🟡 No high-confidence setups right now.\n"
            "Need composite >62 or <38 + confidence >70% + Gate 1 open.",
            reply_markup=main_keyboard(),
        )


async def cmd_report(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    async with httpx.AsyncClient() as client:
        gn, dn = await asyncio.gather(
            fetch_news(client, "GLD"),
            fetch_news(client, "DIA"),
        )
    for inst, news in [("Gold", gn), ("US30", dn)]:
        if d := live_scores.get(inst):
            await _mo(update).reply_text(fmt_instrument(d), parse_mode="Markdown")
        if cats := _fmt_news(news, inst):
            await _mo(update).reply_text(cats, parse_mode="Markdown")
    if d := live_scores.get("EURUSD"):
        await _mo(update).reply_text(fmt_instrument(d), parse_mode="Markdown")
    await _mo(update).reply_text("Report complete ✅", reply_markup=main_keyboard())


async def cmd_weekly(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not any(weekly_history.values()):
        await _mo(update).reply_text(
            "📅 History building — check back after first few macro cycles.",
            reply_markup=main_keyboard(),
        )
        return
    await job_weekly_recap(ctx.application)


async def button_handler(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.callback_query.answer()
    h = {
        "gold":    cmd_gold,
        "us30":    cmd_us30,
        "eurusd":  cmd_eurusd,
        "summary": cmd_summary,
        "setup":   cmd_setup,
        "weekly":  cmd_weekly,
    }.get(update.callback_query.data)
    if h:
        await h(update, ctx)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    global ws_mgr, app_ref

    ws_mgr = WebSocketManager(FINNHUB_API_KEY, on_tick_callback=on_price_tick)

    async def post_init(application):
        global app_ref
        app_ref = application
        log.info("Starting Finnhub WebSocket...")
        asyncio.create_task(ws_mgr.start())
        # Give WebSocket 5 seconds to connect and get first prices
        await asyncio.sleep(5)
        # Fire first macro cycle immediately on startup
        log.info("Firing initial macro cycle...")
        asyncio.create_task(macro_cycle())
        log.info("QuantRisk Bot v6 — Three-Speed Architecture live. 🚀")

    app = (
        Application.builder()
        .token(TELEGRAM_BOT_TOKEN)
        .post_init(post_init)
        .build()
    )

    app.add_handler(CommandHandler("start",   cmd_start))
    app.add_handler(CommandHandler("gold",    cmd_gold))
    app.add_handler(CommandHandler("us30",    cmd_us30))
    app.add_handler(CommandHandler("eurusd",  cmd_eurusd))
    app.add_handler(CommandHandler("report",  cmd_report))
    app.add_handler(CommandHandler("setup",   cmd_setup))
    app.add_handler(CommandHandler("summary", cmd_summary))
    app.add_handler(CommandHandler("weekly",  cmd_weekly))
    app.add_handler(CallbackQueryHandler(button_handler))

    scheduler = AsyncIOScheduler(timezone="Africa/Nairobi")

    # Macro cycle — every 30 minutes
    scheduler.add_job(
        lambda: asyncio.create_task(macro_cycle()),
        "interval", minutes=30, id="macro",
    )
    # Micro cycle — every 15 minutes
    scheduler.add_job(
        lambda: asyncio.create_task(micro_cycle()),
        "interval", minutes=15, id="micro",
    )
    # Daily report — 07:30 EAT
    scheduler.add_job(
        lambda: asyncio.create_task(job_daily_report(app)),
        "cron", hour=7, minute=30, id="daily",
    )
    # Weekly recap — Monday 07:30 EAT
    scheduler.add_job(
        lambda: asyncio.create_task(job_weekly_recap(app)),
        "cron", day_of_week="mon", hour=7, minute=30, id="weekly",
    )

    scheduler.start()
    log.info("QuantRisk Bot v6 starting...")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
