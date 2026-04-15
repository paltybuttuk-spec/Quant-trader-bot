"""
QuantRisk Bot v7 — Pure Price Action Engine
============================================
Speed 1 — Finnhub WebSocket (always)   : live_price float update only
Speed 2 — 15-min micro cycle           : Gate B + Gate C check, setup fire
Speed 3 — 30-min macro cycle           : all 5 TFs → structure + scoring + flips

Six Pure Price Action Models (zero lagging indicators):
  1. Swing Structure         — HH/HL or LH/LL on raw swing points
  2. Candle Close Position   — where price closes inside its own range
  3. Inside Bar + Range Contraction — compression before expansion
  4. Wick vs Body Dominance  — raw buying/selling pressure per candle
  5. True Range Percentile   — vol regime from raw ranges, no smoothing
  6. Swing Momentum Divergence — price vs move-size, no RSI needed

Three Sequential Gates:
  Gate A — STRUCTURE  : Swing structure aligned on 4H + Daily + Weekly
  Gate B — LOCATION   : Price at wick-defined level + range compression present
  Gate C — TRIGGER    : 3 consecutive 15M closes confirm + no divergence

API Budget:
  Macro  : 6 calls × 3 instruments × 48 cycles = 864 — offset by Gate A skip
  Micro  : 2 calls × 3 instruments × 96 cycles = 576 — only when Gate A open
  Real   : Gate A fails ~40% → actual ~700/day ✅ within Twelve Data free 800/day
"""

import asyncio
import logging
import os
from collections import defaultdict, Counter
from datetime import datetime, timedelta
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
TIMEFRAMES   = ["15min", "1h", "4h", "1day", "1week"]
TF_LABELS    = {"15min": "15M", "1h": "1H", "4h": "4H", "1day": "Daily", "1week": "Weekly"}
CANDLE_COUNT = 50

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

# ── State ─────────────────────────────────────────────────────────────────────
live_prices     = {}                   # {inst: float}
prev_verdicts   = {}                   # {inst: {tf: verdict}}
weekly_history  = defaultdict(list)    # {inst: [(dt, price, score, verdict)]}
last_setup_fire = defaultdict(dict)    # {inst: {"LONG": dt, "SHORT": dt}}
live_scores     = {}                   # {inst: full result dict}
gate_a_open     = {}                   # {inst: bool}
structure_bias  = {}                   # {inst: "LONG"|"SHORT"|None}

app_ref         = None
ws_mgr: WebSocketManager = None

# ── Emoji maps ────────────────────────────────────────────────────────────────
VE = {"STRONG BUY": "🟢🟢", "BUY": "🟢", "NEUTRAL": "🟡", "SELL": "🔴", "STRONG SELL": "🔴🔴"}
SE = {"LONG": "🟢", "SHORT": "🔴", None: "🟡"}
IE = {"Gold": "🥇", "US30": "🏦", "EURUSD": "💶"}
RE = {"TRENDING": "📈", "RANGING": "↔️", "MIXED": "〰️"}


# ══════════════════════════════════════════════════════════════════════════════
# SESSION
# ══════════════════════════════════════════════════════════════════════════════

def get_session():
    """Returns (multiplier, label, is_active) in EAT (UTC+3)."""
    hour = (datetime.utcnow().hour + 3) % 24
    if  8 <= hour < 11:  return 1.2, "London Open", True
    if 11 <= hour < 16:  return 1.4, "London/NY Overlap", True
    if 16 <= hour < 23:  return 1.3, "New York", True
    if  0 <= hour <  8:  return 0.7, "Asian", False
    return 0.5, "After Hours", False

def is_active_session():
    _, _, active = get_session()
    return active


# ══════════════════════════════════════════════════════════════════════════════
# TWELVE DATA FETCH
# ══════════════════════════════════════════════════════════════════════════════

async def fetch_candles(client: httpx.AsyncClient, symbol: str, timeframe: str, live_price: float = None):
    """Fetch fresh candles. Optionally patch last close with live price."""
    params = {
        "symbol":     symbol,
        "interval":   TD_INTERVAL[timeframe],
        "outputsize": CANDLE_COUNT,
        "apikey":     TWELVEDATA_API_KEY,
    }
    try:
        r    = await client.get("https://api.twelvedata.com/time_series", params=params, timeout=15)
        data = r.json()
        if data.get("status") == "error":
            log.warning(f"TD error {symbol} {timeframe}: {data.get('message')}")
            return None
        values = list(reversed(data.get("values", [])))
        if not values:
            return None
        parsed = {
            "closes":  [float(v["close"])          for v in values],
            "highs":   [float(v["high"])            for v in values],
            "lows":    [float(v["low"])             for v in values],
            "opens":   [float(v["open"])            for v in values],
            "volumes": [float(v.get("volume", 1))  for v in values],
        }
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
        r      = await client.get("https://api.twelvedata.com/time_series", params=params, timeout=10)
        data   = r.json()
        if data.get("status") == "error":
            return None, 0.0
        values = list(reversed(data.get("values", [])))
        if len(values) < 2:
            return None, 0.0
        closes = [float(v["close"]) for v in values]
        mom    = (closes[-1] - closes[-6]) / closes[-6] if closes[-6] > 0 else 0.0
        return closes[-1], mom
    except Exception as e:
        log.error(f"Corr fetch failed {symbol}: {e}")
        return None, 0.0


# ══════════════════════════════════════════════════════════════════════════════
# PURE PRICE ACTION ENGINE — 6 models, zero lagging indicators
# ══════════════════════════════════════════════════════════════════════════════

# ── Model 1: Swing Structure ──────────────────────────────────────────────────

def find_swing_points(highs: list, lows: list, lookback: int = 2):
    """
    Find swing highs and lows from raw candle data.
    Swing high = candle with lower highs on both sides (lookback bars each side).
    Swing low  = candle with higher lows on both sides.
    Returns last 4 swing highs and 4 swing lows as price values.
    """
    swing_highs = []
    swing_lows  = []
    n = len(highs)
    for i in range(lookback, n - lookback):
        # Swing high: higher than all surrounding bars
        if all(highs[i] > highs[i - j] for j in range(1, lookback + 1)) and \
           all(highs[i] > highs[i + j] for j in range(1, lookback + 1)):
            swing_highs.append(highs[i])
        # Swing low: lower than all surrounding bars
        if all(lows[i] < lows[i - j] for j in range(1, lookback + 1)) and \
           all(lows[i] < lows[i + j] for j in range(1, lookback + 1)):
            swing_lows.append(lows[i])
    return swing_highs[-4:], swing_lows[-4:]


def detect_swing_structure(highs: list, lows: list):
    """
    Model 1: Pure swing structure detection.
    Uptrend   = consecutive higher swing highs AND higher swing lows.
    Downtrend = consecutive lower swing highs AND lower swing lows.
    Returns: ("BULLISH"|"BEARISH"|"MIXED", strength 0-3, last_swing_low, last_swing_high)
    """
    sh, sl = find_swing_points(highs, lows)
    if len(sh) < 2 or len(sl) < 2:
        return "MIXED", 0, lows[-1], highs[-1]

    # Count consecutive HH and HL
    hh_count = sum(1 for i in range(1, len(sh)) if sh[i] > sh[i - 1])
    hl_count = sum(1 for i in range(1, len(sl)) if sl[i] > sl[i - 1])
    lh_count = sum(1 for i in range(1, len(sh)) if sh[i] < sh[i - 1])
    ll_count = sum(1 for i in range(1, len(sl)) if sl[i] < sl[i - 1])

    bull_strength = hh_count + hl_count
    bear_strength = lh_count + ll_count

    if bull_strength > bear_strength and bull_strength >= 2:
        return "BULLISH", bull_strength, sl[-1], sh[-1]
    if bear_strength > bull_strength and bear_strength >= 2:
        return "BEARISH", bear_strength, sl[-1], sh[-1]
    return "MIXED", 0, sl[-1] if sl else lows[-1], sh[-1] if sh else highs[-1]


# ── Model 2: Candle Close Position ───────────────────────────────────────────

def candle_close_position(closes: list, highs: list, lows: list, lookback: int = 10):
    """
    Model 2: Where price closes inside its own range over last N candles.
    Top 25% = strong bullish. Bottom 25% = strong bearish.
    Returns: score contribution (-15 to +15), label, consecutive_confirm count
    """
    if len(closes) < lookback:
        return 0, "NEUTRAL", 0

    range_high = max(highs[-lookback:])
    range_low  = min(lows[-lookback:])
    range_size = range_high - range_low
    if range_size == 0:
        return 0, "NEUTRAL", 0

    # Position of last close inside the range
    position = (closes[-1] - range_low) / range_size  # 0.0 = bottom, 1.0 = top

    # Count consecutive closes in same zone (last 3 candles)
    consecutive = 0
    for i in range(-3, 0):
        bar_range = highs[i] - lows[i]
        if bar_range == 0:
            continue
        bar_pos = (closes[i] - lows[i]) / bar_range
        if position > 0.5 and bar_pos > 0.5:
            consecutive += 1
        elif position < 0.5 and bar_pos < 0.5:
            consecutive += 1

    if   position > 0.75: return 15, "STRONG_BULL", consecutive
    elif position > 0.60: return 8,  "BULL",        consecutive
    elif position < 0.25: return -15, "STRONG_BEAR", consecutive
    elif position < 0.40: return -8,  "BEAR",        consecutive
    return 0, "NEUTRAL", consecutive


# ── Model 3: Inside Bar + Range Contraction ───────────────────────────────────

def detect_compression(highs: list, lows: list, closes: list, lookback: int = 5):
    """
    Model 3: Inside bar sequence and range contraction detection.
    Inside bar = current high < prev high AND current low > prev low.
    Contraction = each bar's true range smaller than previous.
    Returns: (is_compressed bool, compression_bars int, expansion_confirmed bool, regime)
    """
    if len(highs) < lookback + 2:
        return False, 0, False, "UNKNOWN"

    # True ranges for last N bars
    true_ranges = []
    for i in range(-lookback - 1, 0):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i]  - closes[i - 1])
        )
        true_ranges.append(tr)

    # Count contracting bars
    compression_count = 0
    for i in range(1, len(true_ranges)):
        if true_ranges[i] < true_ranges[i - 1]:
            compression_count += 1
        else:
            break  # contraction must be consecutive from the end

    # Inside bars (last 3)
    inside_count = 0
    for i in range(-3, -1):
        if highs[i] < highs[i - 1] and lows[i] > lows[i - 1]:
            inside_count += 1

    is_compressed = compression_count >= 3 or inside_count >= 2

    # Expansion: latest bar range > 1.5x average of compressed bars
    avg_compressed = mean(true_ranges[:-1]) if len(true_ranges) > 1 else true_ranges[0]
    expansion_confirmed = true_ranges[-1] > 1.5 * avg_compressed

    if is_compressed and not expansion_confirmed:
        regime = "RANGING"   # coiling, waiting
    elif expansion_confirmed:
        regime = "TRENDING"  # breakout in progress
    else:
        regime = "MIXED"

    return is_compressed, compression_count, expansion_confirmed, regime


# ── Model 4: Wick vs Body Dominance ──────────────────────────────────────────

def wick_body_analysis(opens: list, closes: list, highs: list, lows: list, lookback: int = 10):
    """
    Model 4: Raw buying and selling pressure from candle structure.
    Bullish body  = close > open AND body > 60% of range AND lower wick < upper wick.
    Rejection wick = wick > 2x body (price rejected this level hard).
    Divergence    = price new high but last 3 candles showing upper rejection wicks.
    Returns: (score_delta -20 to +20, pressure_label, key_level, divergence_warning)
    """
    if len(opens) < lookback:
        return 0, "NEUTRAL", closes[-1], False

    bull_count = 0
    bear_count = 0
    rejection_wicks = []

    for i in range(-lookback, 0):
        body       = abs(closes[i] - opens[i])
        rng        = highs[i] - lows[i]
        if rng == 0:
            continue
        upper_wick = highs[i] - max(opens[i], closes[i])
        lower_wick = min(opens[i], closes[i]) - lows[i]
        body_ratio = body / rng

        is_bull = closes[i] > opens[i]
        is_bear = closes[i] < opens[i]

        if body_ratio > 0.6:
            if is_bull and lower_wick <= upper_wick:
                bull_count += 1
            elif is_bear and upper_wick <= lower_wick:
                bear_count += 1

        # Rejection wick detection
        if body > 0 and max(upper_wick, lower_wick) > 2 * body:
            rejection_wicks.append(("upper" if upper_wick > lower_wick else "lower", i))

    # Find key level: candle with largest lower wick (buyers defended hard)
    best_support_idx = max(range(-lookback, 0),
                           key=lambda i: (min(opens[i], closes[i]) - lows[i]))
    key_level = lows[best_support_idx]  # strongest buyer defense level

    # Divergence: price near high but recent rejection upper wicks
    recent_rejections = [w for w in rejection_wicks if w[1] >= -3]
    upper_rejections  = sum(1 for w in recent_rejections if w[0] == "upper")
    price_near_high   = closes[-1] > max(highs[-lookback:]) * 0.995
    divergence        = price_near_high and upper_rejections >= 2

    net = bull_count - bear_count
    if   net >= 5:  return 20, "STRONG_BULL", key_level, divergence
    elif net >= 3:  return 12, "BULL",        key_level, divergence
    elif net <= -5: return -20, "STRONG_BEAR", key_level, divergence
    elif net <= -3: return -12, "BEAR",        key_level, divergence
    return 0, "NEUTRAL", key_level, divergence


# ── Model 5: True Range Percentile ───────────────────────────────────────────

def true_range_percentile(highs: list, lows: list, closes: list):
    """
    Model 5: Vol regime from raw true range percentile, no smoothing.
    Returns: (multiplier, label, percentile, current_tr, skip_entry)
    skip_entry = True if current bar is itself an extreme range bar (already moved).
    """
    if len(closes) < 2:
        return 1.0, "NORMAL", 50, 0, False

    true_ranges = []
    for i in range(1, len(closes)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i]  - closes[i - 1])
        )
        true_ranges.append(tr)

    if not true_ranges:
        return 1.0, "NORMAL", 50, 0, False

    current_tr  = true_ranges[-1]
    n_below     = sum(1 for t in true_ranges[:-1] if t < current_tr)
    percentile  = round(n_below / max(len(true_ranges) - 1, 1) * 100, 1)
    skip_entry  = percentile > 80  # already an extreme move candle

    if   percentile < 20:  return 1.3, "LOW",     percentile, current_tr, skip_entry
    elif percentile < 50:  return 1.0, "NORMAL",  percentile, current_tr, skip_entry
    elif percentile < 80:  return 0.8, "ELEVATED", percentile, current_tr, skip_entry
    else:                  return 0.4, "EXTREME", percentile, current_tr, skip_entry


# ── Model 6: Swing Momentum Divergence ───────────────────────────────────────

def swing_momentum_divergence(highs: list, lows: list, closes: list):
    """
    Model 6: Direct price vs move-size divergence. No RSI.
    Compares last 2 swing highs (or lows) and the magnitude of the move that created them.
    Bearish div: new price high but smaller move = sellers absorbing.
    Bullish div: new price low but smaller move = buyers absorbing.
    Returns: (score_delta, divergence_type, strength)
    """
    sh, sl = find_swing_points(highs, lows, lookback=2)

    if len(sh) < 2 and len(sl) < 2:
        return 0, "NONE", 0

    score_delta = 0
    div_type    = "NONE"
    strength    = 0

    # Bearish divergence: new high with smaller move
    if len(sh) >= 2 and len(sl) >= 2:
        move_to_high1 = sh[-2] - sl[-2] if len(sl) >= 2 else sh[-2] - min(lows)
        move_to_high2 = sh[-1] - sl[-1]

        if sh[-1] > sh[-2] and move_to_high2 < move_to_high1 * 0.7:
            shrink = 1 - (move_to_high2 / move_to_high1) if move_to_high1 > 0 else 0
            strength    = round(shrink * 100, 1)
            score_delta = -18
            div_type    = "BEARISH"

        # Bullish divergence: new low with smaller move
        elif len(sl) >= 2:
            move_to_low1 = sh[-2] - sl[-2] if len(sh) >= 2 else max(highs) - sl[-2]
            move_to_low2 = sh[-1] - sl[-1] if len(sh) >= 1 else max(highs) - sl[-1]

            if sl[-1] < sl[-2] and move_to_low2 < move_to_low1 * 0.7:
                shrink      = 1 - (move_to_low2 / move_to_low1) if move_to_low1 > 0 else 0
                strength    = round(shrink * 100, 1)
                score_delta = 18
                div_type    = "BULLISH"

    return score_delta, div_type, strength


# ══════════════════════════════════════════════════════════════════════════════
# SCORE ASSEMBLY — all 6 models → 0-100 composite per timeframe
# ══════════════════════════════════════════════════════════════════════════════

def score_to_verdict(s: float) -> str:
    if s >= 70: return "STRONG BUY"
    if s >= 58: return "BUY"
    if s >= 45: return "NEUTRAL"
    if s >= 32: return "SELL"
    return "STRONG SELL"


def score_timeframe(closes: list, highs: list, lows: list, opens: list, volumes: list):
    """
    Assemble all 6 pure price action models into a 0-100 score.
    No indicators. Every signal derived directly from OHLC candle data.
    """
    if len(closes) < 15:
        return {
            "score": 50, "verdict": "NEUTRAL", "regime": "MIXED",
            "structure": "MIXED", "structure_strength": 0,
            "close_position": "NEUTRAL", "consecutive_closes": 0,
            "compression": False, "expansion": False,
            "wick_pressure": "NEUTRAL", "divergence_warning": False,
            "div_type": "NONE", "vol_label": "NORMAL", "vol_pct": 50,
            "skip_entry": False, "key_level": closes[-1],
        }

    score = 50.0

    # ── Model 1: Swing Structure ──────────────────────────────────────────────
    structure, s_strength, last_sw_low, last_sw_high = detect_swing_structure(highs, lows)
    if structure == "BULLISH":
        score += min(12, s_strength * 4)
    elif structure == "BEARISH":
        score -= min(12, s_strength * 4)

    # ── Model 2: Candle Close Position ────────────────────────────────────────
    cp_delta, cp_label, consec = candle_close_position(closes, highs, lows)
    score += cp_delta
    # Bonus: 3 consecutive closes in same zone = momentum building
    if consec == 3:
        score += 5 if cp_delta > 0 else -5

    # ── Model 3: Inside Bar + Range Contraction ───────────────────────────────
    is_compressed, comp_bars, expansion, regime = detect_compression(highs, lows, closes)
    # In expansion after compression = trend starting, boost current direction signal
    if expansion and structure == "BULLISH":
        score += 10
    elif expansion and structure == "BEARISH":
        score -= 10
    # Compression alone = reduce score toward neutral (indecision)
    if is_compressed and not expansion:
        score = 50 + (score - 50) * 0.6

    # ── Model 4: Wick vs Body Dominance ──────────────────────────────────────
    wick_delta, wick_label, key_level, divergence_warning = wick_body_analysis(
        opens, closes, highs, lows
    )
    score += wick_delta
    # Divergence warning: suppress signal hard
    if divergence_warning:
        score = 50 + (score - 50) * 0.5

    # ── Model 5: True Range Percentile ───────────────────────────────────────
    vol_mult, vol_label, vol_pct, current_tr, skip_entry = true_range_percentile(
        highs, lows, closes
    )
    score = 50 + (score - 50) * vol_mult

    # ── Model 6: Swing Momentum Divergence ───────────────────────────────────
    div_delta, div_type, div_strength = swing_momentum_divergence(highs, lows, closes)
    score += div_delta

    # ── Session multiplier ────────────────────────────────────────────────────
    sm, _, _ = get_session()
    score = 50 + (score - 50) * sm

    score = max(0.0, min(100.0, score))

    return {
        "score":               round(score, 1),
        "verdict":             score_to_verdict(score),
        "regime":              regime,
        "structure":           structure,
        "structure_strength":  s_strength,
        "close_position":      cp_label,
        "consecutive_closes":  consec,
        "compression":         is_compressed,
        "expansion":           expansion,
        "wick_pressure":       wick_label,
        "divergence_warning":  divergence_warning,
        "div_type":            div_type,
        "div_strength":        div_strength,
        "vol_label":           vol_label,
        "vol_pct":             vol_pct,
        "skip_entry":          skip_entry,
        "key_level":           round(key_level, 5),
    }


def calc_composite(tf_scores: dict):
    """
    Weighted composite — structure TFs anchor, short TFs time entry.
    Weights: Weekly 15% | Daily 25% | 4H 30% | 1H 20% | 15M 10%
    Alignment bonus: 4+ TFs in same direction → ±6
    Weekly/Daily conflict → confidence halved
    """
    TF_WEIGHTS = {"15min": 0.10, "1h": 0.20, "4h": 0.30, "1day": 0.25, "1week": 0.15}

    if not tf_scores:
        return 50.0, 0.0

    total_w   = sum(TF_WEIGHTS.get(tf, 0.2) for tf in tf_scores)
    composite = sum(tf_scores[tf] * TF_WEIGHTS.get(tf, 0.2) for tf in tf_scores) / total_w

    scores = list(tf_scores.values())
    bull   = sum(1 for s in scores if s > 58)
    bear   = sum(1 for s in scores if s < 42)

    if   bull >= 4: composite = min(100, composite + 6)
    elif bear >= 4: composite = max(0,   composite - 6)

    w_score = tf_scores.get("1week", 50)
    d_score = tf_scores.get("1day",  50)
    weekly_daily_conflict = (w_score > 58 and d_score < 42) or (w_score < 42 and d_score > 58)

    aligned    = max(bull, bear)
    confidence = (composite / 100) * (aligned / max(len(scores), 1)) * 100
    if weekly_daily_conflict:
        confidence *= 0.5

    return round(composite, 1), round(confidence, 1)


# ══════════════════════════════════════════════════════════════════════════════
# THREE SEQUENTIAL GATES
# ══════════════════════════════════════════════════════════════════════════════

def check_gate_a(tf_details: dict):
    """
    Gate A — STRUCTURE
    4H + Daily + Weekly swing structures must all agree in same direction.
    This uses the raw structure field from score_timeframe, not the score.
    Returns: (passed bool, direction "LONG"|"SHORT"|None, description str)
    """
    key_tfs = ["4h", "1day", "1week"]
    structures = {}
    for tf in key_tfs:
        d = tf_details.get(tf)
        if d:
            structures[tf] = d.get("structure", "MIXED")

    if len(structures) < 2:
        return False, None, "Insufficient data"

    bull_tfs = [tf for tf, s in structures.items() if s == "BULLISH"]
    bear_tfs = [tf for tf, s in structures.items() if s == "BEARISH"]

    if len(bull_tfs) >= 2 and "BEARISH" not in structures.values():
        desc = f"Bullish: {', '.join(TF_LABELS.get(tf, tf) for tf in bull_tfs)}"
        return True, "LONG", desc
    if len(bear_tfs) >= 2 and "BULLISH" not in structures.values():
        desc = f"Bearish: {', '.join(TF_LABELS.get(tf, tf) for tf in bear_tfs)}"
        return True, "SHORT", desc

    return False, None, "Structure conflict or mixed"


def check_gate_b(price: float, tf_details: dict, direction: str):
    """
    Gate B — LOCATION
    Price must be at a meaningful level defined by wick analysis.
    AND range must be compressed (not already in middle of a big move).
    AND current bar must not be an extreme range candle (already moved).
    Returns: (passed bool, level float, location_type str)
    """
    d1h = tf_details.get("1h") or tf_details.get("4h")
    if not d1h:
        return False, price, "No 1H data"

    # Skip if current candle is already an extreme move
    if d1h.get("skip_entry", False):
        return False, price, "Extreme range candle — wait"

    key_level = d1h.get("key_level", price)

    # Get ATR approximation from 1H data (we don't store it separately)
    # Use vol_pct to infer range size relative to key level
    # A 0.5% buffer covers most instruments
    buffer = price * 0.005

    at_key_level = abs(price - key_level) < buffer

    # Check compression — better to enter into compressed zone not extended
    compressed = d1h.get("compression", False)
    expansion  = d1h.get("expansion",  False)

    if direction == "LONG":
        # Want price near the buyer defense level and market not already expanding
        near_support = price <= key_level * 1.003  # within 0.3% of key level
        if at_key_level or (near_support and compressed):
            return True, key_level, "At buyer wick level"
        if near_support and not expansion:
            return True, key_level, "Near support, compressed"

    elif direction == "SHORT":
        near_resistance = price >= key_level * 0.997
        if at_key_level or (near_resistance and compressed):
            return True, key_level, "At seller wick level"
        if near_resistance and not expansion:
            return True, key_level, "Near resistance, compressed"

    return False, key_level, f"Price not at level (level={key_level:.5f}, price={price:.5f})"


def check_gate_c(tf_details: dict, direction: str):
    """
    Gate C — TRIGGER
    3 consecutive 15M closes confirm direction.
    No divergence on 15M.
    Expansion candle breaking compression = bonus confirmation.
    Returns: (passed bool, trigger_type str, strength int)
    """
    d15 = tf_details.get("15min")
    d1h = tf_details.get("1h")
    if not d15:
        return False, "No 15M data", 0

    consec    = d15.get("consecutive_closes", 0)
    cp_label  = d15.get("close_position", "NEUTRAL")
    div_warn  = d15.get("divergence_warning", False)
    expansion = d15.get("expansion", False)
    div_type  = d15.get("div_type", "NONE")

    # Divergence cancels trigger
    if div_warn:
        return False, "Divergence warning — skip", 0
    if direction == "LONG"  and div_type == "BEARISH":
        return False, "Bearish divergence active", 0
    if direction == "SHORT" and div_type == "BULLISH":
        return False, "Bullish divergence active", 0

    # Strength calculation
    strength = 0

    if direction == "LONG":
        if cp_label in ("STRONG_BULL", "BULL") and consec >= 2:
            strength += 2
        if consec == 3:
            strength += 1
        if expansion and d15.get("structure") == "BULLISH":
            strength += 1
        # 1H agreement adds conviction
        if d1h and d1h.get("close_position") in ("STRONG_BULL", "BULL"):
            strength += 1

    elif direction == "SHORT":
        if cp_label in ("STRONG_BEAR", "BEAR") and consec >= 2:
            strength += 2
        if consec == 3:
            strength += 1
        if expansion and d15.get("structure") == "BEARISH":
            strength += 1
        if d1h and d1h.get("close_position") in ("STRONG_BEAR", "BEAR"):
            strength += 1

    trigger_type = "STRONG" if strength >= 3 else ("MODERATE" if strength >= 2 else "WEAK")
    passed       = strength >= 2  # require at least moderate

    return passed, trigger_type, strength


# ══════════════════════════════════════════════════════════════════════════════
# SETUP CALCULATION
# ══════════════════════════════════════════════════════════════════════════════

def calc_setup(closes: list, highs: list, lows: list, key_level: float,
               direction: str, tf_details: dict):
    """
    Calculate trade setup anchored to structure levels, not current price.
    Entry zone = around key level (wick-defined buyer/seller area).
    Stop = beyond the key level (if buyers defended here, stop is below here).
    Targets = structure-anchored (swing points, not fixed ATR multiples).
    """
    # ATR from daily candles (true range average)
    trs = []
    for i in range(1, min(15, len(closes))):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i]  - closes[i - 1])
        )
        trs.append(tr)
    atr = mean(trs) if trs else (highs[-1] - lows[-1])

    # Vol regime for adaptive stop
    _, vol_label, vol_pct, _, _ = true_range_percentile(highs, lows, closes)
    stop_mult = 2.2 if vol_pct > 70 else (1.1 if vol_pct < 30 else 1.5)

    price = closes[-1]

    # Structure levels
    support    = round(min(lows[-20:]),  5)
    resistance = round(max(highs[-20:]), 5)

    if direction == "LONG":
        entry_low  = round(key_level - 0.3 * atr, 5)
        entry_high = round(key_level + 0.3 * atr, 5)
        stop       = round(key_level - stop_mult * atr, 5)
        t1         = round(price + 1.5 * atr, 5)
        t2         = round(price + 2.5 * atr, 5)
        t3         = resistance                      # actual structure target
    else:
        entry_low  = round(key_level - 0.3 * atr, 5)
        entry_high = round(key_level + 0.3 * atr, 5)
        stop       = round(key_level + stop_mult * atr, 5)
        t1         = round(price - 1.5 * atr, 5)
        t2         = round(price - 2.5 * atr, 5)
        t3         = support                         # actual structure target

    rr1 = round(abs(t1 - price) / max(abs(price - stop), 0.00001), 1)
    rr2 = round(abs(t2 - price) / max(abs(price - stop), 0.00001), 1)

    return {
        "direction":  direction,
        "entry_low":  entry_low,
        "entry_high": entry_high,
        "stop":       stop,
        "stop_mult":  stop_mult,
        "t1":         t1, "t2": t2, "t3": t3,
        "rr1":        rr1, "rr2": rr2,
        "atr":        round(atr, 5),
        "support":    support,
        "resistance": resistance,
        "key_level":  round(key_level, 5),
        "vol_label":  vol_label,
    }


# ══════════════════════════════════════════════════════════════════════════════
# WEBSOCKET — Speed 1
# ══════════════════════════════════════════════════════════════════════════════

async def on_price_tick(symbol, price):
    inst = WS_TO_INST.get(symbol)
    if inst:
        live_prices[inst] = price


# ══════════════════════════════════════════════════════════════════════════════
# NEWS — FINNHUB
# ══════════════════════════════════════════════════════════════════════════════

async def fetch_news(client: httpx.AsyncClient, symbol: str):
    now    = datetime.utcnow()
    from_d = (now - timedelta(days=7)).strftime("%Y-%m-%d")
    try:
        r        = await client.get(
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
# MACRO CYCLE — every 30 min, Speed 3
# Full 5-TF scoring, flip detection, Gate A evaluation
# ══════════════════════════════════════════════════════════════════════════════

async def macro_cycle():
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
            corr_price, corr_mom = (
                corr_result if isinstance(corr_result, tuple) else (None, 0.0)
            )

            tf_scores  = {}
            tf_details = {}

            for i, tf in enumerate(TIMEFRAMES):
                candles = candle_results[i]
                if isinstance(candles, Exception) or candles is None:
                    continue
                d = score_timeframe(
                    candles["closes"], candles["highs"], candles["lows"],
                    candles["opens"],  candles["volumes"],
                )
                tf_scores[tf]  = d["score"]
                tf_details[tf] = d

            if not tf_scores:
                continue

            composite, conf = calc_composite(tf_scores)

            # Gate A — structural direction
            ga_passed, direction, ga_desc = check_gate_a(tf_details)
            gate_a_open[inst]    = ga_passed
            structure_bias[inst] = direction

            log.info(f"Macro {inst}: composite={composite} Gate A={'✅' if ga_passed else '❌'} dir={direction}")

            # Weekly history
            weekly_history[inst].append(
                (datetime.utcnow(), price, composite, score_to_verdict(composite))
            )
            weekly_history[inst] = weekly_history[inst][-336:]

            # Flip detection
            await _check_flips(inst, tf_details)

            # Build and store result
            result = {
                "instrument":    inst,
                "candle_sym":    sym,
                "price":         price,
                "composite":     composite,
                "confidence":    conf,
                "verdict":       score_to_verdict(composite),
                "tf_scores":     tf_scores,
                "tf_details":    tf_details,
                "gate_a":        ga_passed,
                "gate_a_desc":   ga_desc,
                "direction":     direction,
                "corr_price":    corr_price,
                "corr_label":    cfg["corr_label"],
                "corr_mom":      round(corr_mom * 100, 3),
                "scored_at":     datetime.utcnow().isoformat(),
                "cycle":         "macro",
                "setup":         None,   # setup set in micro cycle
            }
            live_scores[inst] = result

    log.info("── Macro cycle complete ──")


async def _check_flips(inst: str, tf_details: dict):
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
# MICRO CYCLE — every 15 min, Speed 2
# Gate B + Gate C check on fresh 15M + 1H data. Fire setup if all pass.
# ══════════════════════════════════════════════════════════════════════════════

async def micro_cycle():
    log.info("── Micro cycle starting ──")
    if not is_active_session():
        log.info("Micro: outside active session, skipping.")
        return

    async with httpx.AsyncClient() as client:
        for inst, cfg in INSTRUMENTS.items():

            # Gate A must be open from last macro cycle
            if not gate_a_open.get(inst):
                log.info(f"Micro: {inst} Gate A closed, skip.")
                continue

            price = live_prices.get(inst)
            if not price:
                continue

            direction = structure_bias.get(inst)
            if not direction:
                continue

            sym = cfg["candle_sym"]

            # Fetch 15M + 1H fresh in parallel
            c15, c1h = await asyncio.gather(
                fetch_candles(client, sym, "15min", price),
                fetch_candles(client, sym, "1h",    price),
                return_exceptions=True,
            )

            if isinstance(c15, Exception) or c15 is None:
                continue
            if isinstance(c1h, Exception) or c1h is None:
                continue

            d15 = score_timeframe(c15["closes"], c15["highs"], c15["lows"],
                                  c15["opens"],  c15["volumes"])
            d1h = score_timeframe(c1h["closes"], c1h["highs"], c1h["lows"],
                                  c1h["opens"],  c1h["volumes"])

            # Merge short TF details with macro details for gate checks
            full_tf = dict(live_scores.get(inst, {}).get("tf_details", {}))
            full_tf["15min"] = d15
            full_tf["1h"]    = d1h

            # Gate B — location check
            gb_passed, key_level, gb_desc = check_gate_b(price, full_tf, direction)
            if not gb_passed:
                log.info(f"Micro: {inst} Gate B failed: {gb_desc}")
                continue

            # Gate C — trigger check
            gc_passed, trigger_type, trigger_strength = check_gate_c(full_tf, direction)
            if not gc_passed:
                log.info(f"Micro: {inst} Gate C failed: {trigger_type}")
                continue

            # Per-direction cooldown: 90 min between same-direction setups
            now       = datetime.utcnow()
            last_fire = last_setup_fire[inst].get(direction)
            if last_fire and (now - last_fire).total_seconds() < 5400:
                remaining = int((5400 - (now - last_fire).total_seconds()) / 60)
                log.info(f"Micro: {inst} {direction} cooldown — {remaining}min remaining")
                continue

            # All gates passed — build setup from daily candles
            macro_data   = live_scores.get(inst, {})
            daily_tf     = macro_data.get("tf_details", {}).get("1day")
            setup_closes = c1h["closes"]  # fallback to 1H if no daily
            setup_highs  = c1h["highs"]
            setup_lows   = c1h["lows"]

            # Use daily candles if available in macro_data tf_details
            # We don't store raw candles, so we use the 1H data for setup calc
            setup = calc_setup(
                setup_closes, setup_highs, setup_lows,
                key_level, direction, full_tf
            )

            # Update live_scores with setup
            if inst in live_scores:
                live_scores[inst]["setup"]     = setup
                live_scores[inst]["gate_b"]    = True
                live_scores[inst]["gate_b_desc"] = gb_desc
                live_scores[inst]["gate_c"]    = True
                live_scores[inst]["trigger"]   = trigger_type

            # Fire setup alert
            composite = macro_data.get("composite", 50)
            conf      = macro_data.get("confidence", 0)
            try:
                await app_ref.bot.send_message(
                    TELEGRAM_CHAT_ID,
                    fmt_setup_alert_full(inst, price, direction, setup,
                                         composite, conf, full_tf,
                                         trigger_type, trigger_strength, gb_desc),
                    parse_mode="Markdown",
                )
                last_setup_fire[inst][direction] = now
                log.info(f"✅ Setup fired: {inst} {direction} {trigger_type}")
            except Exception as e:
                log.error(f"Setup alert failed: {e}")

    log.info("── Micro cycle complete ──")


# ══════════════════════════════════════════════════════════════════════════════
# FORMATTERS
# ══════════════════════════════════════════════════════════════════════════════

def fmt_instrument(data: dict, include_setup: bool = True) -> str:
    inst   = data["instrument"]
    sm, sl, _ = get_session()
    e      = IE.get(inst, "📊")
    d      = data.get("direction")
    ga     = data.get("gate_a", False)

    lines = [
        f"{e} *{inst}* 🔴 LIVE",
        f"💲 Price: `{data['price']:.5f}`" if data.get("price") else "",
        f"📊 Score: `{data['composite']}/100` — {VE.get(data['verdict'], '⚪')} *{data['verdict']}*",
        f"🎯 Confidence: `{data['confidence']:.1f}%`",
        f"⏰ Session: {sl} (×{sm})",
        "",
        "*Timeframe Breakdown:*",
    ]

    for tf in TIMEFRAMES:
        det = data.get("tf_details", {}).get(tf)
        if not det:
            lines.append(f"  {TF_LABELS[tf]}: —")
            continue
        struct_e = "📈" if det["structure"] == "BULLISH" else ("📉" if det["structure"] == "BEARISH" else "〰️")
        div_e    = " ⚠️div" if det["divergence_warning"] else ""
        comp_e   = " 🔄comp" if det["compression"] and not det["expansion"] else ""
        exp_e    = " 💥exp" if det["expansion"] else ""
        lines.append(
            f"  {TF_LABELS[tf]}: `{det['score']}` {VE.get(det['verdict'], '⚪')} "
            f"{struct_e} {det['wick_pressure']}{div_e}{comp_e}{exp_e}"
        )

    # Quant signals from 15M
    d15 = data.get("tf_details", {}).get("15min") or next(iter(data.get("tf_details", {}).values()), {})
    if d15:
        lines += [
            "",
            "*Price Action Signals:*",
            f"  Structure: {d15.get('structure','—')} (strength {d15.get('structure_strength',0)})",
            f"  Close pos: {d15.get('close_position','—')} ({d15.get('consecutive_closes',0)} consec)",
            f"  Wick pressure: {d15.get('wick_pressure','—')}",
            f"  Compression: {'Yes 🔄' if d15.get('compression') else 'No'} | "
            f"Expansion: {'Yes 💥' if d15.get('expansion') else 'No'}",
            f"  Vol: {d15.get('vol_pct',50):.0f}th %ile → {d15.get('vol_label','—')}",
            f"  Divergence: {d15.get('div_type','NONE')} "
            f"{'⚠️' if d15.get('divergence_warning') else '✅'}",
        ]

    # Gate status
    ga_e = "✅" if ga else "🔴"
    lines.append(f"\n🚦 Gate A: {ga_e} | Direction: {SE.get(d,'🟡')} {d or 'None'}")
    lines.append(f"   {data.get('gate_a_desc','')}")

    # Correlation context
    cp    = data.get("corr_price")
    label = data.get("corr_label", "")
    mom   = data.get("corr_mom", 0)
    if cp:
        if inst in ("Gold", "EURUSD"):
            tone = "Headwind 🔴" if mom > 0.3 else ("Tailwind 🟢" if mom < -0.3 else "Neutral 🟡")
            lines.append(f"\n💱 {label}: `{cp:.4f}` | Mom: `{mom:+.2f}%` → {tone}")
        elif inst == "US30":
            tone = "Risk-Off 🔴" if mom > 0.3 else ("Risk-On 🟢" if mom < -0.3 else "Calm 🟡")
            lines.append(f"\n😱 {label} (VIXY): `{cp:.2f}` | Mom: `{mom:+.2f}%` → {tone}")

    # Setup if present
    if include_setup and data.get("setup"):
        s  = data["setup"]
        de = "🟢 LONG" if s["direction"] == "LONG" else "🔴 SHORT"
        lines += [
            "", f"*🎯 Trade Setup — {de}*",
            f"  Watch zone: `{s['entry_low']} – {s['entry_high']}`",
            f"  Stop: `{s['stop']}` (×{s['stop_mult']} ATR adaptive)",
            f"  T1: `{s['t1']}` (RR 1:{s['rr1']}) | T2: `{s['t2']}` (RR 1:{s['rr2']})",
            f"  T3: `{s['t3']}` (structure target)",
            f"  Key level: `{s['key_level']}` | ATR: `{s['atr']}`",
        ]

    return "\n".join(l for l in lines if l is not None)


def fmt_flip_alert(inst: str, tf: str, old_v: str, new_v: str, d: dict) -> str:
    e = IE.get(inst, "📊")
    return (
        f"⚡ *STRUCTURE FLIP — {e} {inst}*\n"
        f"Timeframe: *{TF_LABELS.get(tf, tf)}*\n"
        f"{VE.get(old_v, '⚪')} {old_v} → {VE.get(new_v, '⚪')} {new_v}\n"
        f"Score: `{d['score']}` | Structure: {d['structure']}\n"
        f"Wick: {d['wick_pressure']} | Vol: {d['vol_label']} ({d['vol_pct']:.0f}th %ile)\n"
        f"Divergence: {d['div_type']} {'⚠️' if d['divergence_warning'] else '✅'}"
    )


def fmt_setup_alert_full(inst: str, price: float, direction: str, setup: dict,
                          composite: float, conf: float, tf_details: dict,
                          trigger_type: str, trigger_strength: int, gate_b_desc: str) -> str:
    e  = IE.get(inst, "📊")
    de = "🟢 LONG" if direction == "LONG" else "🔴 SHORT"
    s  = setup
    _, sl, _ = get_session()

    d15 = tf_details.get("15min", {})
    d1h = tf_details.get("1h",    {})
    d4h = tf_details.get("4h",    {})
    ddy = tf_details.get("1day",  {})
    dwk = tf_details.get("1week", {})

    struct_line = (
        f"Daily {ddy.get('structure','—')} | "
        f"4H {d4h.get('structure','—')} | "
        f"Weekly {dwk.get('structure','—')}"
    )
    trigger_line = (
        f"15M closes: {d15.get('consecutive_closes',0)}/3 | "
        f"Wick: {d15.get('wick_pressure','—')} | "
        f"Vol: {d15.get('vol_pct',50):.0f}th %ile"
    )

    return (
        f"🎯 *{trigger_type} SETUP — {e} {inst}*\n"
        f"{de} | Score: `{composite}` | Conf: `{conf:.1f}%`\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"✅ *Gate A Structure:*  {struct_line}\n"
        f"✅ *Gate B Location:*   {gate_b_desc}\n"
        f"✅ *Gate C Trigger:*    {trigger_line}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"*Watch zone:*  `{s['entry_low']} – {s['entry_high']}`\n"
        f"*Stop:*        `{s['stop']}` (×{s['stop_mult']} ATR)\n"
        f"*T1:*          `{s['t1']}` (RR 1:{s['rr1']})\n"
        f"*T2:*          `{s['t2']}` (RR 1:{s['rr2']})\n"
        f"*T3:*          `{s['t3']}` (structure target)\n"
        f"Key level: `{s['key_level']}` | ATR: `{s['atr']}`\n"
        f"S: `{s['support']}` | R: `{s['resistance']}`\n"
        f"⏰ {sl} | Trigger strength: {trigger_strength}/5"
    )


def fmt_setup_alert(inst: str, data: dict) -> str:
    """Simple setup alert for /setup command."""
    s  = data.get("setup")
    if not s:
        return f"{IE.get(inst,'📊')} *{inst}*: No setup currently active."
    e  = IE.get(inst, "📊")
    de = "🟢 LONG" if s["direction"] == "LONG" else "🔴 SHORT"
    return (
        f"🎯 *SETUP — {e} {inst}*\n"
        f"{de} | Score: `{data['composite']}` | Conf: `{data['confidence']:.1f}%`\n"
        f"Watch: `{s['entry_low']} – {s['entry_high']}`\n"
        f"Stop: `{s['stop']}` (×{s['stop_mult']} ATR)\n"
        f"T1: `{s['t1']}` (RR 1:{s['rr1']}) | T2: `{s['t2']}` (RR 1:{s['rr2']})\n"
        f"T3: `{s['t3']}` | Key level: `{s['key_level']}`"
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
    _, sl, _ = get_session()
    hdr = (
        f"☀️ *QuantRisk Daily Report*\n"
        f"📅 {datetime.utcnow().strftime('%A, %d %B %Y')} | EAT 07:30\n"
        f"{'─' * 30}"
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
            f"  Avg: `{round(mean(scores), 1)}` | "
            f"High: `{max(scores)}` | Low: `{min(scores)}`\n"
            f"  Dominant: {VE.get(tv, '⚪')} {tv} | Points: {len(history)}\n"
        )
    await app.bot.send_message(TELEGRAM_CHAT_ID, "\n".join(lines), parse_mode="Markdown")


# ══════════════════════════════════════════════════════════════════════════════
# COMMANDS
# ══════════════════════════════════════════════════════════════════════════════

def _mo(update: Update):
    return update.message or update.callback_query.message


async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    g  = "🟢" if live_prices.get("Gold")   else "🟡"
    eu = "🟢" if live_prices.get("EURUSD") else "🟡"
    di = "🟢" if live_prices.get("US30")   else "🟡"
    sc = "✅ Scores live" if live_scores else "⏳ First macro cycle pending..."
    await _mo(update).reply_text(
        f"👋 *QuantRisk Bot v7 — Pure Price Action*\n\n"
        f"{g} Gold (XAU/USD) — 24/5 live\n"
        f"{eu} EURUSD — 24/5 live\n"
        f"{di} US30 (DIA) — market hours\n"
        f"{sc}\n\n"
        f"*6 pure price action models ✅*\n"
        f"*3 sequential gates ✅ | Zero lagging indicators ✅*\n"
        f"Macro: 30min | Micro: 15min | WebSocket: live",
        parse_mode="Markdown",
        reply_markup=main_keyboard(),
    )


async def _send_inst(update: Update, inst: str):
    d = live_scores.get(inst)
    if not d:
        await _mo(update).reply_text(
            f"⏳ {inst} — waiting for first macro cycle (up to 30s after start).",
            reply_markup=main_keyboard(),
        )
        return
    await _mo(update).reply_text(
        fmt_instrument(d), parse_mode="Markdown", reply_markup=main_keyboard()
    )


async def cmd_gold(u, c):   await _send_inst(u, "Gold")
async def cmd_us30(u, c):   await _send_inst(u, "US30")
async def cmd_eurusd(u, c): await _send_inst(u, "EURUSD")


async def cmd_summary(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    sm, sl, _ = get_session()
    lines     = [f"📊 *QuantRisk Summary*\n⏰ {sl} (×{sm})\n"]
    for inst in ["Gold", "US30", "EURUSD"]:
        d = live_scores.get(inst)
        e = IE.get(inst, "📊")
        if not d:
            lines.append(f"{e} *{inst}*: ⏳ Loading...")
            continue
        ga = "✅" if gate_a_open.get(inst) else "🔴"
        bias = structure_bias.get(inst) or "None"
        lines.append(
            f"{e} *{inst}*: `{d['composite']}/100` {VE.get(d['verdict'], '⚪')} {d['verdict']}\n"
            f"  Conf: `{d['confidence']:.1f}%` | Gate A: {ga} | Direction: {bias}"
        )
    await _mo(update).reply_text(
        "\n".join(lines), parse_mode="Markdown", reply_markup=main_keyboard()
    )


async def cmd_setup(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    found = False
    for inst in ["Gold", "US30", "EURUSD"]:
        d = live_scores.get(inst)
        if d and d.get("setup"):
            await _mo(update).reply_text(fmt_setup_alert(inst, d), parse_mode="Markdown")
            found = True
    if not found:
        await _mo(update).reply_text(
            "🟡 No active setups right now.\n"
            "Need Gate A (structure) + Gate B (location) + Gate C (trigger) all open.\n"
            "Setups fire automatically during active sessions.",
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
        await asyncio.sleep(5)
        log.info("Firing initial macro cycle...")
        asyncio.create_task(macro_cycle())
        log.info("QuantRisk Bot v7 — Pure Price Action live 🚀")

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

    scheduler.add_job(
        lambda: asyncio.create_task(macro_cycle()),
        "interval", minutes=30, id="macro",
    )
    scheduler.add_job(
        lambda: asyncio.create_task(micro_cycle()),
        "interval", minutes=15, id="micro",
    )
    scheduler.add_job(
        lambda: asyncio.create_task(job_daily_report(app)),
        "cron", hour=7, minute=30, id="daily",
    )
    scheduler.add_job(
        lambda: asyncio.create_task(job_weekly_recap(app)),
        "cron", day_of_week="mon", hour=7, minute=30, id="weekly",
    )

    scheduler.start()
    log.info("QuantRisk Bot v7 starting...")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
