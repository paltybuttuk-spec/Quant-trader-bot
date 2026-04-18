"""
QuantRisk Bot v5 — Three Instruments
- Gold:   OANDA:XAU_USD WebSocket (24/5) + XAU/USD Twelve Data candles
- US30:   DIA WebSocket (market hours) + DIA Twelve Data candles
- EURUSD: OANDA:EUR_USD WebSocket (24/5) + EUR/USD Twelve Data candles
- DXY:    OANDA:USD_DXY WebSocket — correlation for Gold + EURUSD
- VIX:    VIXY Twelve Data — correlation for US30
All 6 scoring improvements active.
"""

import asyncio
import logging
import os
import time as _time
from collections import defaultdict
from datetime import datetime
from statistics import mean, stdev

import httpx
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes

from candle_cache import CandleCache
from websocket_manager import WebSocketManager
from monte_carlo import run_monte_carlo, run_prospective_scan
from ict import ict_score_delta, enrich_setup_with_ict, find_fvg, find_order_blocks

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
log = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID   = int(os.environ["TELEGRAM_CHAT_ID"])
TWELVEDATA_API_KEY = os.environ["TWELVEDATA_API_KEY"]
FINNHUB_API_KEY    = os.environ["FINNHUB_API_KEY"]

TIMEFRAMES = ["15min", "1h", "4h", "1day", "1week"]
TF_LABELS  = {"15min": "15M", "1h": "1H", "4h": "4H", "1day": "Daily", "1week": "Weekly"}

# Candle count per timeframe — more history on short TFs for dense S/R
CANDLE_COUNTS = {
    "15min": 500,
    "1h":    500,
    "4h":    500,
    "1day":  50,
    "1week": 50,
}

INSTRUMENTS = {
    "Gold": {
        "candle_sym":  "XAU/USD",
        "ws_sym":      "OANDA:XAU_USD",
        "corr_candle": "UUP",
        "corr_ws":     "OANDA:USD_DXY",
        "corr_label":  "DXY",
    },
    "US30": {
        "candle_sym":  "DIA",
        "ws_sym":      "DIA",
        "corr_candle": "VIXY",
        "corr_ws":     "VIXY",
        "corr_label":  "VIX",
    },
    "EURUSD": {
        "candle_sym":  "EUR/USD",
        "ws_sym":      "OANDA:EUR_USD",
        "corr_candle": "UUP",
        "corr_ws":     "OANDA:USD_DXY",
        "corr_label":  "DXY",
    },
}

CANDLE_SYMBOLS = ["XAU/USD", "EUR/USD", "DIA", "UUP", "VIXY"]

WS_TO_INST = {
    "OANDA:XAU_USD": "Gold",
    "OANDA:EUR_USD": "EURUSD",
    "DIA":           "US30",
}

WS_TO_CORR = {
    "OANDA:USD_DXY": ["Gold", "EURUSD"],
    "VIXY":          ["US30"],
}

prev_verdicts       = {}   # last CONFIRMED verdict per inst/tf
verdict_candidates  = {}   # pending new verdict per inst/tf: {inst: {tf: [verdict, count, score]}}
VERDICT_CONFIRM_N   = 3    # must see same new verdict N consecutive checks to confirm a flip
VERDICT_MIN_GAP     = 6    # score must be >= this many points clear of boundary to confirm

weekly_history  = defaultdict(list)
score_history   = defaultdict(list)
last_setup_fire    = {}
prospective_scans  = {}      # {inst: latest prospective scan result}
last_prospect_fire = {}      # {inst: datetime} — cooldown for prospect alerts
live_scores     = {}
_last_score_ts  = {}
cache: CandleCache       = None
ws_mgr: WebSocketManager = None
app_ref = None

VE = {"STRONG BUY":"🟢🟢","BUY":"🟢","NEUTRAL":"🟡","SELL":"🔴","STRONG SELL":"🔴🔴"}
RE = {"TRENDING":"📈","RANGING":"↔️","MIXED":"〰️"}
IE = {"Gold":"🥇","US30":"🏦","EURUSD":"💶"}


def get_session_multiplier():
    hour = (datetime.utcnow().hour + 3) % 24
    if 8  <= hour < 11: return 1.2, "London Open"
    if 11 <= hour < 16: return 1.4, "London/NY Overlap"
    if 16 <= hour < 23: return 1.3, "New York"
    if 0  <= hour < 8:  return 0.7, "Asian"
    return 0.5, "After Hours"

def is_active_session():
    m, _ = get_session_multiplier()
    return m > 0.8

def calc_atr(highs, lows, closes, period=14):
    trs = [max(highs[i]-lows[i], abs(highs[i]-closes[i-1]), abs(lows[i]-closes[i-1]))
           for i in range(1, len(closes))]
    return mean(trs[-period:]) if len(trs) >= period else (mean(trs) if trs else 0)

def calc_vwap(closes, volumes, period=20):
    c, v = closes[-period:], volumes[-period:]
    tv = sum(v)
    return sum(p*vv for p, vv in zip(c, v)) / tv if tv else closes[-1]

# ─── Support / Resistance Engine ────────────────────────────────────────────

SR_SWING_WING   = 3      # wider wing on 500-candle history — reduces noise pivots
SR_CLUSTER_ATR  = 0.20   # tighter cluster radius — keeps distinct levels distinct
SR_MAX_LEVELS   = 5      # more levels available with deeper history
SR_PROX_ATR     = 1.5    # "near a level" threshold as fraction of ATR
SR_SCORE_STRONG = 12     # score adjustment at a STRONG level
SR_SCORE_MOD    = 7      # score adjustment at a MODERATE level
SR_SCORE_WEAK   = 3      # score adjustment at a WEAK level
# Touch thresholds scale up with 500-candle history — more data = higher bar for STRONG
SR_TOUCH_STRONG = 7      # touches needed for STRONG  (was 5)
SR_TOUCH_MOD    = 4      # touches needed for MODERATE (was 3)

def _find_swings(highs, lows):
    """Return lists of (index, price, 'high'|'low') swing pivots."""
    w = SR_SWING_WING
    pivots = []
    n = len(highs)
    for i in range(w, n - w):
        if all(highs[i] >= highs[j] for j in range(i-w, i+w+1) if j != i):
            pivots.append((i, highs[i], "high"))
        if all(lows[i]  <= lows[j]  for j in range(i-w, i+w+1) if j != i):
            pivots.append((i, lows[i],  "low"))
    return pivots

def _cluster_pivots(pivots, atr, n_bars):
    """
    Merge pivots within SR_CLUSTER_ATR * atr of each other into zones.
    Returns list of dicts: {price, touches, recency_weight, kind}.
    recency_weight = avg(bar_index / n_bars) so recent pivots score higher.
    """
    radius = SR_CLUSTER_ATR * atr
    zones = []
    for idx, price, kind in sorted(pivots, key=lambda x: x[1]):
        merged = False
        for z in zones:
            if abs(price - z["price"]) <= radius:
                # Merge into existing zone
                total = z["touches"] + 1
                z["price"]          = (z["price"] * z["touches"] + price) / total
                z["touches"]        = total
                z["recency_weight"] = (z["recency_weight"] * (total-1) + idx/n_bars) / total
                if kind != z["kind"]: z["kind"] = "both"
                merged = True
                break
        if not merged:
            zones.append({"price": price, "touches": 1,
                          "recency_weight": idx/n_bars, "kind": kind})
    return zones

def _zone_strength(z):
    """STRONG / MODERATE / WEAK based on touch count.
    Thresholds scale with 500-candle history on short timeframes."""
    t = z["touches"]
    if t >= SR_TOUCH_STRONG: return "STRONG"
    if t >= SR_TOUCH_MOD:    return "MODERATE"
    return "WEAK"

def calc_sr_levels(highs, lows, closes, atr):
    """
    Main entry point. Returns:
      {
        "support":    [{"price", "touches", "strength", "dist_atr"}, ...],
        "resistance": [{"price", "touches", "strength", "dist_atr"}, ...],
        "nearest_support":    dict or None,
        "nearest_resistance": dict or None,
      }
    All lists sorted by strength then recency, capped at SR_MAX_LEVELS.
    """
    if atr <= 0 or len(highs) < 10:
        return {"support": [], "resistance": [],
                "nearest_support": None, "nearest_resistance": None}

    price   = closes[-1]
    pivots  = _find_swings(highs, lows)
    if not pivots:
        return {"support": [], "resistance": [],
                "nearest_support": None, "nearest_resistance": None}

    zones   = _cluster_pivots(pivots, atr, len(highs))

    supports    = [z for z in zones if z["price"] < price and z["kind"] in ("low",  "both")]
    resistances = [z for z in zones if z["price"] > price and z["kind"] in ("high", "both")]

    def _rank(lst):
        # Sort: STRONG first, then by recency (higher = more recent)
        order = {"STRONG": 0, "MODERATE": 1, "WEAK": 2}
        return sorted(lst,
                      key=lambda z: (order[_zone_strength(z)], -z["recency_weight"]))

    supports    = _rank(supports)[:SR_MAX_LEVELS]
    resistances = _rank(resistances)[:SR_MAX_LEVELS]

    def _enrich(z):
        return {
            "price":    round(z["price"], 5),
            "touches":  z["touches"],
            "strength": _zone_strength(z),
            "dist_atr": round(abs(z["price"] - price) / atr, 2),
        }

    sup_e = [_enrich(z) for z in supports]
    res_e = [_enrich(z) for z in resistances]

    nearest_sup = min(sup_e, key=lambda z: z["dist_atr"]) if sup_e else None
    nearest_res = min(res_e, key=lambda z: z["dist_atr"]) if res_e else None

    return {
        "support":            sup_e,
        "resistance":         res_e,
        "nearest_support":    nearest_sup,
        "nearest_resistance": nearest_res,
    }

def sr_score_adjustment(sr, atr, score, direction_bias):
    """
    Adjust score based on proximity and strength of nearest S/R.
    direction_bias: +1 = bullish signal active, -1 = bearish, 0 = neutral.
    Returns (adjustment: float, context_str: str).
    """
    adj  = 0.0
    tags = []

    ns = sr.get("nearest_support")
    nr = sr.get("nearest_resistance")

    strength_adj = {
        "STRONG":   SR_SCORE_STRONG,
        "MODERATE": SR_SCORE_MOD,
        "WEAK":     SR_SCORE_WEAK,
    }

    if nr and nr["dist_atr"] <= SR_PROX_ATR:
        # Price near resistance
        mag = strength_adj[nr["strength"]]
        if direction_bias >= 0:
            # Bullish signal running into resistance → suppress
            adj -= mag
            tags.append(f"Near {nr['strength']} R ({nr['dist_atr']:.1f}×ATR) ⚠️")
        else:
            # Bearish — resistance confirms direction → boost
            adj -= mag * 0.4
            tags.append(f"{nr['strength']} R overhead ({nr['dist_atr']:.1f}×ATR) 🔴")

    if ns and ns["dist_atr"] <= SR_PROX_ATR:
        # Price near support
        mag = strength_adj[ns["strength"]]
        if direction_bias <= 0:
            # Bearish signal at support → suppress
            adj += mag
            tags.append(f"Near {ns['strength']} S ({ns['dist_atr']:.1f}×ATR) ⚠️")
        else:
            # Bullish — support confirms direction → boost
            adj += mag * 0.4
            tags.append(f"{ns['strength']} S below ({ns['dist_atr']:.1f}×ATR) 🟢")

    context = " | ".join(tags) if tags else ""
    return adj, context

def snap_stop_to_sr(stop, direction, sr, atr):
    """
    If a stronger S/R level sits just beyond the raw ATR stop,
    snap the stop to that level (adding a small buffer).
    Never moves stop closer to entry.
    """
    buffer = atr * 0.15
    if direction == "LONG":
        candidates = [z["price"] - buffer
                      for z in sr["support"]
                      if z["price"] < stop + atr * 0.5
                      and z["strength"] in ("STRONG", "MODERATE")]
        if candidates:
            best = max(candidates)           # closest strong support below
            if best < stop:                  # only widen, never tighten
                return round(best, 5), True
    else:
        candidates = [z["price"] + buffer
                      for z in sr["resistance"]
                      if z["price"] > stop - atr * 0.5
                      and z["strength"] in ("STRONG", "MODERATE")]
        if candidates:
            best = min(candidates)
            if best > stop:
                return round(best, 5), True
    return stop, False

def snap_targets_to_sr(t1, t2, t3, direction, sr):
    """
    If a strong/moderate S/R level sits just before a target,
    pull the target back to that level (avoids setting targets inside walls).
    """
    def _snap(target, levels, direction):
        buffer = 0.0
        for z in levels:
            p = z["price"]
            if z["strength"] not in ("STRONG", "MODERATE"): continue
            if direction == "LONG"  and p < target and p > target * 0.998:
                return round(p - buffer, 5), z["strength"]
            if direction == "SHORT" and p > target and p < target * 1.002:
                return round(p + buffer, 5), z["strength"]
        return target, None

    res_levels = sr.get("resistance", [])
    sup_levels = sr.get("support", [])
    walls      = res_levels if direction == "LONG" else sup_levels

    t1_new, t1_tag = _snap(t1, walls, direction)
    t2_new, t2_tag = _snap(t2, walls, direction)
    t3_new, t3_tag = _snap(t3, walls, direction)
    return t1_new, t2_new, t3_new, any([t1_tag, t2_tag, t3_tag])

# ─────────────────────────────────────────────────────────────────────────────

def score_to_verdict(s):
    if s >= 70: return "STRONG BUY"
    if s >= 58: return "BUY"
    if s >= 45: return "NEUTRAL"
    if s >= 32: return "SELL"
    return "STRONG SELL"

def score_timeframe(closes, highs, lows, opens, volumes,
                    dxy_momentum=0.0, vixy_pressure=0.0, instrument="Gold"):
    if len(closes) < 30:
        return {"score": 50, "verdict": "NEUTRAL", "regime": "MIXED",
                "z_score": 0, "momentum": 0, "vol_ratio": 1,
                "vol_label": "NORMAL", "vwap_signal": "NEUTRAL",
                "efficiency": 0, "vol_conf": 1.0}
    score = 50.0
    net   = abs(closes[-1] - closes[-11])
    path  = sum(abs(closes[i] - closes[i-1]) for i in range(-10, 0))
    eff   = net / path if path > 0 else 0
    regime = "TRENDING" if eff > 0.55 else ("RANGING" if eff < 0.35 else "MIXED")
    r10 = [abs(closes[i]-closes[i-1])/closes[i-1] for i in range(-10, 0) if closes[i-1] > 0]
    r30 = [abs(closes[i]-closes[i-1])/closes[i-1] for i in range(-30, 0) if closes[i-1] > 0]
    v10 = mean(r10) if r10 else 0.01
    v30 = mean(r30) if r30 else 0.01
    vol_ratio = v10 / v30 if v30 > 0 else 1.0
    if   vol_ratio < 0.70: vol_mult, vol_label = 1.2, "LOW"
    elif vol_ratio < 1.50: vol_mult, vol_label = 1.0, "NORMAL"
    elif vol_ratio < 2.00: vol_mult, vol_label = 0.7, "HIGH"
    else:                  vol_mult, vol_label = 0.3, "EXTREME"
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
        score *= 0.85
    avg_v = mean(volumes[-20:]) if len(volumes) >= 20 else mean(volumes)
    vc    = 1.0 if volumes[-1] > avg_v else 0.5
    m5    = (closes[-1]-closes[-6])  / closes[-6]  if closes[-6]  > 0 else 0
    m10   = (closes[-1]-closes[-11]) / closes[-11] if closes[-11] > 0 else 0
    m20   = (closes[-1]-closes[-21]) / closes[-21] if closes[-21] > 0 else 0
    mom   = (m5*0.5 + m10*0.3 + m20*0.2) * vc
    mp    = mom * 200
    if   regime == "TRENDING": score += mp
    elif regime == "MIXED":    score += mp * 0.5
    else:                      score += mp * 0.25
    vwap = calc_vwap(closes, volumes)
    if closes[-1] > vwap: score += 8; vs = "ABOVE"
    else:                 score -= 8; vs = "BELOW"
    if instrument in ("Gold", "EURUSD") and dxy_momentum:
        score += max(-15, min(15, -dxy_momentum * 150))
    elif instrument == "US30" and vixy_pressure:
        score += max(-12, min(12, -vixy_pressure * 100))
    sm, _ = get_session_multiplier()
    score  = 50 + (score - 50) * vol_mult
    score  = 50 + (score - 50) * sm
    score  = max(0, min(100, score))

    # ── S/R integration ──────────────────────────────────────────────────────
    atr_val = calc_atr(highs, lows, closes)
    sr      = calc_sr_levels(highs, lows, closes, atr_val)
    bias    = 1 if score > 55 else (-1 if score < 45 else 0)
    sr_adj, sr_ctx = sr_score_adjustment(sr, atr_val, score, bias)
    score   = max(0, min(100, score + sr_adj))

    # ── ICT integration ───────────────────────────────────────────────────────
    ict_delta, ict_data = ict_score_delta(closes, highs, lows, opens, bias)
    score = max(0, min(100, score + ict_delta))
    # ─────────────────────────────────────────────────────────────────────────

    return {"score": round(score,1), "verdict": score_to_verdict(score),
            "regime": regime, "z_score": round(z,2), "momentum": round(mom*100,3),
            "vol_ratio": round(vol_ratio,2), "vol_label": vol_label,
            "vwap_signal": vs, "efficiency": round(eff,3), "vol_conf": vc,
            "sr": sr, "sr_context": sr_ctx, "atr": round(atr_val, 5),
            "ict": ict_data}

def calc_composite(tf_scores):
    if not tf_scores: return 50.0, 0.0
    scores    = list(tf_scores.values())
    composite = mean(scores)
    bull = sum(1 for s in scores if s > 58)
    bear = sum(1 for s in scores if s < 42)
    if bull >= 4:   composite = min(100, composite + 8)
    elif bear >= 4: composite = max(0,   composite - 8)
    aligned    = max(bull, bear)
    confidence = (composite / 100) * (aligned / len(scores)) * 100
    return round(composite, 1), round(confidence, 1)

def update_score_history(inst, score):
    h = score_history[inst]
    h.append(score)
    score_history[inst] = h[-3:]
    if len(h) < 3: return False
    return all(s > 58 for s in h) or all(s < 42 for s in h)

# ─── S/R Entry Engine ────────────────────────────────────────────────────────

SR_FLIP_WINDOW  = 3      # candles back to check if price crossed level (flip detection)
SR_FLIP_BUFFER  = 0.25   # level must be within this many ATRs of current price to be a flip
SR_PB_BUFFER    = 1.0    # pullback: price must be within this many ATRs of level to be "at it"

def detect_sr_entry(closes, highs, lows, sr, atr, direction):
    """
    Detect S/R-based entry type and anchor price.
    Returns dict: {type, level, level_price, strength, touches} or None.

    Priority:
      1. S/R FLIP — former resistance now support (LONG) or former support now resistance (SHORT)
         Detected when price recently crossed a level and is now retesting from the other side.
      2. PULLBACK — price has retraced to nearest support (LONG) or resistance (SHORT)
         Detected when current price is within SR_PB_BUFFER × ATR of the level.
    """
    if not sr or atr <= 0: return None
    price = closes[-1]

    # ── 1. Flip detection ─────────────────────────────────────────────────────
    # For LONG: look for a resistance level that price broke above recently
    #           and is now trading just above (former resistance = new support)
    # For SHORT: look for a support level that price broke below recently
    #            and is now trading just below (former support = new resistance)
    if direction == "LONG":
        flip_candidates = sr.get("resistance", [])   # levels above recent lows
        for z in flip_candidates:
            lp = z["price"]
            if z["strength"] == "WEAK": continue
            # Price must be above the level now (broken through)
            if price <= lp: continue
            # Must be close — don't pick up distant broken levels
            if (price - lp) / atr > SR_FLIP_BUFFER * 2: continue
            # Check that price was BELOW this level within the last SR_FLIP_WINDOW candles
            recent_lows = lows[-(SR_FLIP_WINDOW + 1):-1]
            if any(l < lp for l in recent_lows):
                return {
                    "type":        "FLIP",
                    "level":       "Resistance→Support",
                    "level_price": lp,
                    "strength":    z["strength"],
                    "touches":     z["touches"],
                }
    else:  # SHORT
        flip_candidates = sr.get("support", [])
        for z in flip_candidates:
            lp = z["price"]
            if z["strength"] == "WEAK": continue
            if price >= lp: continue
            if (lp - price) / atr > SR_FLIP_BUFFER * 2: continue
            recent_highs = highs[-(SR_FLIP_WINDOW + 1):-1]
            if any(h > lp for h in recent_highs):
                return {
                    "type":        "FLIP",
                    "level":       "Support→Resistance",
                    "level_price": lp,
                    "strength":    z["strength"],
                    "touches":     z["touches"],
                }

    # ── 2. Pullback to nearest S/R ────────────────────────────────────────────
    if direction == "LONG":
        ns = sr.get("nearest_support")
        if ns and ns["dist_atr"] <= SR_PB_BUFFER:
            return {
                "type":        "PULLBACK",
                "level":       "Support",
                "level_price": ns["price"],
                "strength":    ns["strength"],
                "touches":     ns["touches"],
            }
    else:
        nr = sr.get("nearest_resistance")
        if nr and nr["dist_atr"] <= SR_PB_BUFFER:
            return {
                "type":        "PULLBACK",
                "level":       "Resistance",
                "level_price": nr["price"],
                "strength":    nr["strength"],
                "touches":     nr["touches"],
            }

    return None

# ─────────────────────────────────────────────────────────────────────────────

def calc_setup(closes, highs, lows, composite, vol_ratio, sr=None):
    atr   = calc_atr(highs, lows, closes)
    price = closes[-1]
    d     = "LONG" if composite > 58 else "SHORT"
    stop_buffer = atr * 0.15   # just beyond the S/R level

    # ── Detect S/R entry type ─────────────────────────────────────────────────
    sr_entry = detect_sr_entry(closes, highs, lows, sr or {}, atr, d) if sr else None

    if sr_entry:
        # Entry zone anchored to the S/R level (±0.1×ATR band around level)
        lp   = sr_entry["level_price"]
        el   = round(lp - 0.1 * atr, 5)
        eh   = round(lp + 0.1 * atr, 5)
        # Stop just beyond the level
        if d == "LONG":
            stop = round(lp - stop_buffer, 5)
            t1   = round(lp + 2 * atr, 5)
            t2   = round(lp + 3 * atr, 5)
            t3   = round(lp + 4 * atr, 5)
        else:
            stop = round(lp + stop_buffer, 5)
            t1   = round(lp - 2 * atr, 5)
            t2   = round(lp - 3 * atr, 5)
            t3   = round(lp - 4 * atr, 5)
        stop_mult    = round((abs(lp - stop)) / atr, 2)
        stop_snapped = True
    else:
        # Fallback: ATR-based entry (no S/R anchor found)
        el, eh = round(price - 0.3 * atr, 5), round(price + 0.3 * atr, 5)
        sm     = 2.2 if vol_ratio > 1.5 else (1.1 if vol_ratio < 0.7 else 1.5)
        if d == "LONG":
            stop = round(price - sm * atr, 5)
            t1, t2, t3 = round(price+2*atr,5), round(price+3*atr,5), round(price+4*atr,5)
        else:
            stop = round(price + sm * atr, 5)
            t1, t2, t3 = round(price-2*atr,5), round(price-3*atr,5), round(price-4*atr,5)
        stop_mult    = sm
        stop_snapped = False

    # ── Snap targets to S/R walls ─────────────────────────────────────────────
    targets_adjusted = False
    if sr:
        t1, t2, t3, targets_adjusted = snap_targets_to_sr(t1, t2, t3, d, sr)

    # ── Quality gate: reject if strong S/R sits between entry and T1 ──────────
    setup_blocked = False
    if sr:
        walls = sr["resistance"] if d == "LONG" else sr["support"]
        for z in walls:
            if z["strength"] == "STRONG":
                p       = z["price"]
                between = (price < p < t1) if d == "LONG" else (t1 < p < price)
                if between and z["dist_atr"] < 0.5:
                    setup_blocked = True
                    break

    return {
        "direction":          d,
        "entry_low":          el,
        "entry_high":         eh,
        "stop":               stop,
        "t1":                 t1,
        "t2":                 t2,
        "t3":                 t3,
        "atr":                round(atr, 5),
        "stop_mult":          stop_mult,
        "support":            round(min(lows[-20:]), 5),
        "resistance":         round(max(highs[-20:]), 5),
        "stop_snapped":       stop_snapped,
        "targets_adjusted":   targets_adjusted,
        "setup_blocked":      setup_blocked,
        "sr_entry":           sr_entry,   # None = ATR fallback, dict = S/R anchored
        "sr":                 sr or {},
    }

async def score_instrument_live(instrument, live_price):
    cfg = INSTRUMENTS[instrument]
    dxy_mom = vixy_pres = 0.0
    corr_cd = cache.get(cfg["corr_candle"], "1day")
    if corr_cd and len(corr_cd["closes"]) >= 6:
        cc = corr_cd["closes"]
        m  = (cc[-1]-cc[-6])/cc[-6] if cc[-6] > 0 else 0
        if instrument in ("Gold","EURUSD"): dxy_mom   = m
        elif instrument == "US30":          vixy_pres = m
    corr_live  = ws_mgr.get_price(cfg["corr_ws"]) if ws_mgr else None
    tf_scores  = {}
    tf_details = {}
    daily_data = None
    daily_sr   = None
    vol_ratios = []
    for tf in TIMEFRAMES:
        candles = cache.get_with_live_price(cfg["candle_sym"], tf, live_price)
        if not candles: continue
        d = score_timeframe(candles["closes"],candles["highs"],candles["lows"],
                            candles["opens"],candles["volumes"],
                            dxy_momentum=dxy_mom,vixy_pressure=vixy_pres,
                            instrument=instrument)
        tf_scores[tf]  = d["score"]
        tf_details[tf] = d
        vol_ratios.append(d["vol_ratio"])
        if tf == "1day":
            daily_data = (candles["closes"],candles["highs"],candles["lows"])
            daily_sr   = d.get("sr")
    if not tf_scores: return
    composite, conf = calc_composite(tf_scores)
    persistent  = update_score_history(instrument, composite)
    avg_vr      = mean(vol_ratios) if vol_ratios else 1.0
    setup = None
    if daily_data and (composite > 58 or composite < 42) and conf > 70:
        setup = calc_setup(*daily_data, composite, avg_vr, sr=daily_sr)
        if setup and setup.get("setup_blocked"):
            setup = None
        elif setup:
            if daily_data:
                closes_d, highs_d, lows_d = daily_data
                opens_d  = cache.get(cfg["candle_sym"], "1day")
                opens_d  = opens_d["opens"] if opens_d else closes_d
                setup    = enrich_setup_with_ict(setup, closes_d, highs_d, lows_d, opens_d, tf="1day")
            mc = await run_monte_carlo(setup, daily_data[0])
            setup["mc"] = mc

    # ── Prospective scan — multi-TF, day-trading configured ─────────────────
    # Uses 15min/1H/4H candles (NOT daily) — correct horizons for day trading.
    # Each TF gets its own GARCH sigma, S/R levels, FVGs, and OBs.
    # Results synthesised into a single cross-TF probable setup list.
    prospect = None
    scan_tfs = ["15min", "1h", "4h"]
    candles_by_tf = {}
    sr_by_tf      = {}
    fvgs_by_tf    = {}
    obs_by_tf     = {}
    atrs_by_tf    = {}

    for stf in scan_tfs:
        c = cache.get_with_live_price(cfg["candle_sym"], stf, live_price)
        if not c:
            continue
        candles_by_tf[stf] = c
        td = tf_details.get(stf, {})
        sr_by_tf[stf]   = td.get("sr", {})
        atrs_by_tf[stf] = td.get("atr", 0.0)
        op = cache.get(cfg["candle_sym"], stf)
        op = op["opens"] if op else c["closes"]
        fvgs_by_tf[stf] = find_fvg(c["closes"], c["highs"], c["lows"], lookback=40)
        obs_by_tf[stf]  = find_order_blocks(c["closes"], c["highs"], c["lows"], op, lookback=40)

    if candles_by_tf:
        prospect = await run_prospective_scan(
            current_price = live_price,
            candles_by_tf = candles_by_tf,
            sr_by_tf      = sr_by_tf,
            fvgs_by_tf    = fvgs_by_tf,
            obs_by_tf     = obs_by_tf,
            atrs          = atrs_by_tf,
        )
        prospective_scans[instrument] = prospect
    result = {
        "instrument": instrument, "candle_sym": cfg["candle_sym"],
        "price": live_price, "composite": composite, "confidence": conf,
        "verdict": score_to_verdict(composite), "tf_scores": tf_scores,
        "tf_details": tf_details, "setup": setup, "persistent": persistent,
        "corr_price": corr_live, "corr_label": cfg["corr_label"],
        "dxy_momentum": round(dxy_mom*100,3), "vixy_pressure": round(vixy_pres*100,3),
        "live": True, "scored_at": datetime.utcnow().isoformat(),
        "prospect": prospect,
    }
    live_scores[instrument] = result
    await check_flips_and_alerts(result)

def _score_clear_of_boundary(score: float, verdict: str) -> bool:
    """Return True only if score is comfortably inside its verdict band."""
    g = VERDICT_MIN_GAP
    if verdict == "STRONG BUY":   return score >= 70 + g
    if verdict == "BUY":          return score >= 58 + g and score < 70
    if verdict == "NEUTRAL":      return score >= 45 + g and score < 58 - g
    if verdict == "SELL":         return score >= 32 + g and score < 45 - g
    if verdict == "STRONG SELL":  return score < 32 - g
    return False

async def check_flips_and_alerts(data):
    if app_ref is None: return
    inst = data["instrument"]
    for tf, detail in data["tf_details"].items():
        new_v = detail["verdict"]
        score = detail["score"]
        old_v = prev_verdicts.get(inst, {}).get(tf)

        # First time seen — seed confirmed verdict, no alert
        if old_v is None:
            prev_verdicts.setdefault(inst, {})[tf] = new_v
            verdict_candidates.setdefault(inst, {})[tf] = [new_v, 1, score]
            continue

        if new_v == old_v:
            # Reset any pending candidate for this tf
            verdict_candidates.setdefault(inst, {})[tf] = [new_v, 1, score]
            continue

        # Score is proposing a new verdict — check gap first
        if not _score_clear_of_boundary(score, new_v):
            # Too close to boundary — ignore, don't update candidate
            continue

        # Accumulate consecutive confirmations
        cand = verdict_candidates.setdefault(inst, {}).get(tf, [new_v, 0, score])
        if cand[0] == new_v:
            cand[1] += 1
            cand[2]  = score
        else:
            cand = [new_v, 1, score]
        verdict_candidates[inst][tf] = cand

        if cand[1] >= VERDICT_CONFIRM_N:
            # Confirmed flip — fire alert and update state
            try:
                await app_ref.bot.send_message(TELEGRAM_CHAT_ID,
                    fmt_flip_alert(inst, tf, old_v, new_v, detail), parse_mode="Markdown")
            except Exception as e:
                log.error(f"Flip alert failed: {e}")
            prev_verdicts.setdefault(inst, {})[tf] = new_v
            verdict_candidates[inst][tf] = [new_v, 1, score]
    if is_active_session() and data["setup"] and data["confidence"]>70 and data["persistent"]:
        last = last_setup_fire.get(inst)
        now  = datetime.utcnow()
        if last is None or (now-last).seconds > 5400:
            try:
                await app_ref.bot.send_message(TELEGRAM_CHAT_ID,
                    fmt_setup_alert(inst,data), parse_mode="Markdown")
                last_setup_fire[inst] = now
            except Exception as e:
                log.error(f"Setup alert failed: {e}")

    # ── Prospective setup alert — fires when simulation finds probable setups ─
    prospect = data.get("prospect")
    if prospect and prospect.get("probable_setups") and is_active_session():
        now  = datetime.utcnow()
        last = last_prospect_fire.get(inst)
        # Cooldown: 2 hours between prospect alerts per instrument
        if last is None or (now - last).seconds > 7200:
            try:
                msg = fmt_prospect_alert(inst, data, prospect)
                if msg:
                    await app_ref.bot.send_message(
                        TELEGRAM_CHAT_ID, msg, parse_mode="Markdown")
                    last_prospect_fire[inst] = now
            except Exception as e:
                log.error(f"Prospect alert failed: {e}")
    weekly_history[inst].append((datetime.utcnow(),data["price"],data["composite"],data["verdict"]))
    weekly_history[inst] = weekly_history[inst][-336:]

async def on_price_tick(symbol, price):
    inst = WS_TO_INST.get(symbol)
    if inst:
        now = _time.time()
        if now - _last_score_ts.get(inst,0) < 3: return
        _last_score_ts[inst] = now
        if not cache.is_populated(INSTRUMENTS[inst]["candle_sym"]): return
        try: await score_instrument_live(inst, price)
        except Exception as e: log.error(f"Live score error {inst}: {e}")
        return
    for inst in WS_TO_CORR.get(symbol,[]):
        now = _time.time()
        ck  = f"corr_{inst}"
        if now - _last_score_ts.get(ck,0) < 10: continue
        _last_score_ts[ck] = now
        inst_price = ws_mgr.get_price(INSTRUMENTS[inst]["ws_sym"]) if ws_mgr else None
        if inst_price and cache.is_populated(INSTRUMENTS[inst]["candle_sym"]):
            try: await score_instrument_live(inst, inst_price)
            except Exception as e: log.error(f"Corr rescore error {inst}: {e}")

async def fetch_news(client, symbol):
    now   = datetime.utcnow()
    from_d = now.replace(day=max(1,now.day-7)).strftime("%Y-%m-%d")
    try:
        r = await client.get("https://finnhub.io/api/v1/company-news",
            params={"symbol":symbol,"from":from_d,
                    "to":now.strftime("%Y-%m-%d"),"token":FINNHUB_API_KEY},timeout=10)
        articles = r.json()
        if not isinstance(articles, list): return []
        scored = []
        for a in articles:
            s  = a.get("sentiment",{})
            sc = s.get("bullishPercent",0.5)-s.get("bearishPercent",0.5) if s else 0
            scored.append((sc,a))
        scored.sort(key=lambda x:x[0],reverse=True)
        return scored
    except Exception as e:
        log.error(f"News failed {symbol}: {e}"); return []

def _fmt_news(news, instrument):
    if not news: return ""
    emoji = {"Gold":"🥇","US30":"🏦","EURUSD":"💶"}.get(instrument,"📰")
    lines = [f"{emoji} *{instrument} News*\n","📈 *Catalysts:*"]
    for _,a in news[:5]: lines.append(f"  • {a.get('headline','')[:80]}")
    lines.append("\n📉 *Risks:*")
    for _,a in news[-5:]: lines.append(f"  • {a.get('headline','')[:80]}")
    return "\n".join(lines)

def fmt_instrument(data, include_setup=True):
    inst   = data["instrument"]
    sm, sl = get_session_multiplier()
    e      = IE.get(inst,"📊")
    lines  = [
        f"{e} *{inst}* 🔴 LIVE",
        f"💲 Price: `{data['price']:.5f}`" if data.get("price") else "",
        f"📊 Score: `{data['composite']}/100` — {VE.get(data['verdict'],'⚪')} *{data['verdict']}*",
        f"🎯 Confidence: `{data['confidence']:.1f}%`",
        f"⏰ Session: {sl} (×{sm})", "", "*Timeframe Breakdown:*",
    ]
    for tf in TIMEFRAMES:
        d = data["tf_details"].get(tf)
        if not d: lines.append(f"  {TF_LABELS[tf]}: —"); continue
        vc = "✅" if d["vol_conf"]==1.0 else "⚠️"
        lines.append(f"  {TF_LABELS[tf]}: `{d['score']}` {VE.get(d['verdict'],'⚪')} {RE.get(d['regime'],'')} {vc}")
    d15 = data["tf_details"].get("15min") or next(iter(data["tf_details"].values()),{})
    if d15:
        lines += ["","*Quant Signals:*",
            f"  Z-Score: `{d15.get('z_score',0):+.2f}` (adaptive window)",
            f"  Momentum: `{d15.get('momentum',0):+.3f}%` {'✅' if d15.get('vol_conf',1)==1 else '⚠️ low vol'}",
            f"  Regime: {d15.get('regime','—')} {RE.get(d15.get('regime',''),'')}",
            f"  Vol: `{d15.get('vol_ratio',1):.2f}` → {d15.get('vol_label','NORMAL')}",
            f"  VWAP: {'Above ✅' if d15.get('vwap_signal')=='ABOVE' else 'Below ⚠️'}"]
    cp    = data.get("corr_price")
    label = data.get("corr_label","")
    if cp:
        if inst in ("Gold","EURUSD"):
            m = data["dxy_momentum"]
            t = "Headwind 🔴" if m>0.3 else ("Tailwind 🟢" if m<-0.3 else "Neutral 🟡")
            lines.append(f"\n💱 {label}: `{cp:.4f}` | Mom: `{m:+.2f}%` → {t}")
        elif inst == "US30":
            p = data["vixy_pressure"]
            t = "Risk-Off 🔴" if p>0.3 else ("Risk-On 🟢" if p<-0.3 else "Calm 🟡")
            lines.append(f"\n😱 {label} (VIXY): `{cp:.2f}` | Mom: `{p:+.2f}%` → {t}")
    if data.get("persistent"): lines.append("\n🔒 *Signal Persistent — 3 checks confirmed*")

    # ── Prospective simulation summary ────────────────────────────────────────
    prospect = data.get("prospect")
    if prospect:
        lines.append(fmt_prospect_summary(data["instrument"], prospect))

    # ── S/R levels from daily timeframe ─────────────────────────────────────
    daily_d = data["tf_details"].get("1day", {})
    sr      = daily_d.get("sr", {})
    res_z   = sr.get("resistance", [])
    sup_z   = sr.get("support",    [])
    if res_z or sup_z:
        lines.append("\n*📐 Key S/R Levels (Daily):*")
        for z in res_z[:2]:
            lines.append(f"  🔴 R `{z['price']}` — {z['strength']} ({z['touches']}T, {z['dist_atr']:.1f}×ATR)")
        for z in sup_z[:2]:
            lines.append(f"  🟢 S `{z['price']}` — {z['strength']} ({z['touches']}T, {z['dist_atr']:.1f}×ATR)")
        ctx = daily_d.get("sr_context","")
        if ctx: lines.append(f"  ⚡ {ctx}")
    # ─────────────────────────────────────────────────────────────────────────

    if include_setup and data.get("setup"):
        s  = data["setup"]
        de = "🟢 LONG" if s["direction"]=="LONG" else "🔴 SHORT"
        sr_entry = s.get("sr_entry")
        if sr_entry:
            et = sr_entry["type"]
            el = sr_entry["level"]
            es = sr_entry["strength"]
            entry_tag = f"📍 {et} — {el} ({es})"
        else:
            entry_tag = "📍 ATR-based"
        lines += ["", f"*🎯 Trade Setup — {de}*",
            f"  {entry_tag}",
            f"  Entry: `{s['entry_low']} – {s['entry_high']}`",
            f"  Stop: `{s['stop']}` (×{s['stop_mult']} ATR)",
            f"  T1: `{s['t1']}` | T2: `{s['t2']}` | T3: `{s['t3']}`",
            f"  ATR: `{s['atr']}` | S: `{s['support']}` | R: `{s['resistance']}`"]
        mc = s.get("mc") or {}
        if mc and mc.get("n_sims", 0) > 0:
            ev_sign = "+" if mc["ev"] >= 0 else ""
            lines += [
                f"  ┄┄ MC ({mc['vol_model']}, {mc['n_sims']:,} sims) ┄┄",
                f"  Win: `{mc['win_rate']}%` | EV: `{ev_sign}{mc['ev']}R` | Sharpe: `{mc['sharpe']:.2f}`",
                f"  P(T1/T2/T3): `{mc['p_t1']}` / `{mc['p_t2']}` / `{mc['p_t3']}`%",
                f"  P(Stop): `{mc['p_stop']}%` | σ/yr: `{mc['sigma_daily']}%`",
            ]
    return "\n".join(l for l in lines if l is not None)

def fmt_flip_alert(inst, tf, old_v, new_v, d):
    e = IE.get(inst,"📊")
    sr_line = ""
    ctx = d.get("sr_context","")
    ns  = (d.get("sr") or {}).get("nearest_support")
    nr  = (d.get("sr") or {}).get("nearest_resistance")
    parts = []
    if nr: parts.append(f"R: `{nr['price']}` ({nr['strength']}, {nr['dist_atr']:.1f}×ATR)")
    if ns: parts.append(f"S: `{ns['price']}` ({ns['strength']}, {ns['dist_atr']:.1f}×ATR)")
    if parts: sr_line = "\n📐 " + " | ".join(parts)
    if ctx:   sr_line += f"\n⚡ {ctx}"
    return (f"⚡ *SCORE FLIP — {e} {inst}*\nTimeframe: *{TF_LABELS.get(tf,tf)}*\n"
            f"{VE.get(old_v,'⚪')} {old_v} → {VE.get(new_v,'⚪')} {new_v}\n"
            f"Score: `{d['score']}` | Z: `{d['z_score']:+.2f}` | Mom: `{d['momentum']:+.3f}%`\n"
            f"Regime: {d['regime']} | Vol: {d['vol_label']}"
            f"{sr_line}")

def fmt_setup_alert(inst, data):
    s = data["setup"]; e = IE.get(inst,"📊")
    sr      = s.get("sr", {})
    sup_z   = sr.get("support",   [])
    res_z   = sr.get("resistance",[])
    sr_entry = s.get("sr_entry")

    if sr_entry:
        et = sr_entry["type"]; el = sr_entry["level"]
        es = sr_entry["strength"]; ec = sr_entry["touches"]
        entry_label = f"📍 *{et}* — {el} ({es}, {ec}T)" if et=="FLIP" else f"📍 *{et}* to {el} ({es}, {ec}T)"
    else:
        entry_label = "📍 *ATR-based* (no S/R anchor)"

    sr_lines = [entry_label]
    if res_z:
        top = res_z[0]
        sr_lines.append(f"🔴 R: `{top['price']}` ({top['strength']}, {top['touches']}T, {top['dist_atr']:.1f}×ATR)")
    if sup_z:
        bot = sup_z[0]
        sr_lines.append(f"🟢 S: `{bot['price']}` ({bot['strength']}, {bot['touches']}T, {bot['dist_atr']:.1f}×ATR)")
    if s.get("stop_snapped"):     sr_lines.append("📌 Stop at S/R level")
    if s.get("targets_adjusted"): sr_lines.append("🎯 Target(s) adjusted for S/R wall")
    sr_block = ("\n" + "\n".join(sr_lines)) if sr_lines else ""

    # ── ICT block ─────────────────────────────────────────────────────────────
    ict      = s.get("ict") or {}
    ict_tags = ict.get("tags", [])
    ict_pd   = ict.get("premium_discount", {})
    ict_bos  = ict.get("bos")
    ict_chch = ict.get("choch")
    ict_fvg  = ict.get("nearest_fvg")
    ict_ob   = ict.get("nearest_ob")
    ict_str  = ict.get("structure", "")

    ict_lines = []
    if ict_str:           ict_lines.append(f"Structure: *{ict_str}*")
    if ict_pd.get("zone"): ict_lines.append(f"Zone: {ict_pd['zone']} ({ict_pd.get('ratio','?')})")
    if ict_bos:            ict_lines.append(f"BOS {ict_bos['direction']} @ `{ict_bos['level']}`")
    if ict_chch:           ict_lines.append(f"CHOCH {ict_chch['direction']} @ `{ict_chch['level']}`")
    if ict_fvg:            ict_lines.append(f"Nearest FVG: {ict_fvg['type']} `{ict_fvg['bottom']}–{ict_fvg['top']}`")
    if ict_ob:             ict_lines.append(f"Nearest OB: {ict_ob['type']} `{ict_ob['bottom']}–{ict_ob['top']}`")
    for tag in ict_tags[:3]: ict_lines.append(f"• {tag}")

    ict_block = ("\n\n🧠 *ICT Analysis:*\n" + "\n".join(ict_lines)) if ict_lines else ""

    # ── Monte Carlo block ─────────────────────────────────────────────────────
    mc = s.get("mc") or {}
    if mc and mc.get("n_sims", 0) > 0:
        ev_sign  = "+" if mc["ev"] >= 0 else ""
        mc_block = (
            f"\n\n📊 *Monte Carlo — {mc['n_sims']:,} sims ({mc['vol_model']})*\n"
            f"Win Rate: `{mc['win_rate']}%` | EV: `{ev_sign}{mc['ev']}R` | Sharpe: `{mc['sharpe']:.2f}`\n"
            f"P(T1): `{mc['p_t1']}%` | P(T2): `{mc['p_t2']}%` | P(T3): `{mc['p_t3']}%`\n"
            f"P(Stop): `{mc['p_stop']}%` | σ/yr: `{mc['sigma_daily']}%`"
        )
    else:
        mc_block = ""

    return (f"🚨 *SETUP ALERT — {e} {inst}*\n"
            f"{'🟢 LONG' if s['direction']=='LONG' else '🔴 SHORT'} | "
            f"Score: `{data['composite']}` | Conf: `{data['confidence']:.1f}%`\n"
            f"Entry: `{s['entry_low']} – {s['entry_high']}`\n"
            f"Stop: `{s['stop']}` (×{s['stop_mult']} ATR)\n"
            f"T1: `{s['t1']}` | T2: `{s['t2']}` | T3: `{s['t3']}`\n"
            f"🔒 Persistent: {data['persistent']}"
            f"{sr_block}"
            f"{ict_block}"
            f"{mc_block}")

def _fmt_p(price: float) -> str:
    """2dp for Gold/indices (price>10), 5dp for forex pairs."""
    return f"{price:.2f}" if price > 10 else f"{price:.5f}"


def fmt_prospect_alert(inst, data, prospect):
    """
    Multi-timeframe prospective radar alert — day-trading format.

    Structure:
      Line 1 : instrument + price + cross-TF bias
      Line 2 : vol context (sigma, model, sims) — one line, not a paragraph
      Section A : WATCH THESE LEVELS — probable setup zones, leading with
                  the highest-confluence zone. Each zone shows:
                  - price (2dp Gold, 5dp forex)
                  - % of paths reaching that zone
                  - confluence tags (S/R strength+touches, FVG, OB)
                  - which TFs agree (e.g. "1H+4H")
      Section B : INTRADAY RANGES — IQR (P25–P75) per TF horizon.
                  No delta vs current — that is noise on zero-drift model.
                  Shows the zone price will trade within, not where it ends up.
    """
    if not prospect or not prospect.get("probable_setups"):
        return None

    e      = IE.get(inst, "📊")
    price  = data["price"]
    bias   = prospect.get("bias", "NO CLEAR BIAS")
    p_up   = prospect.get("p_up", 50.0)
    sigma  = prospect.get("sigma_daily", 0.0)
    vm     = prospect.get("vol_model", "?")
    n_sims = prospect.get("n_sims", 0)
    tf_res = prospect.get("tf_results", {})

    # Bias — only show directional label when meaningful
    if bias == "BULLISH":
        bias_str = f"🟢 BULLISH bias ({p_up:.0f}% paths up)"
    elif bias == "BEARISH":
        bias_str = f"🔴 BEARISH bias ({p_up:.0f}% paths up)"
    else:
        bias_str = f"🟡 Split ({p_up:.0f}/{100-p_up:.0f}) — no edge"

    # ── Section A: Watch these levels ────────────────────────────────────────
    setup_lines = []
    for ps in prospect["probable_setups"][:3]:
        d_arrow  = "⬆️" if ps["direction"] == "LONG" else "⬇️"
        zp       = _fmt_p(ps["zone_price"])
        pct      = ps["density_pct"]
        n_conf   = ps["n_levels"]
        tf_src   = ps.get("tf", "")
        tc       = ps.get("tf_count", 1)
        tf_label = f"{'✅✅' if tc >= 2 else '✅'} {tf_src.upper()}" if tf_src else ""

        # First line: price + density + TF agreement
        setup_lines.append(f"  {d_arrow} *{zp}* — {pct}% of paths  {tf_label}")

        # Confluence tags — max 2, compact
        for tag in ps.get("confluence", [])[:2]:
            setup_lines.append(f"    {tag}")

    if not setup_lines:
        return None

    # ── Section B: Intraday ranges (IQR = P25–P75) ───────────────────────────
    # Show 1H and 15min only — most relevant for day trading entry timing
    range_lines = []
    tf_labels = {"1h": "1H range", "15min": "15M range", "4h": "4H range"}
    for tf in ["15min", "1h", "4h"]:
        tres = tf_res.get(tf, {})
        summ = tres.get("summary", {})
        if not summ:
            continue
        # Use shortest horizon for tightest range
        h0 = min(TF_HORIZONS_DISPLAY.get(tf, [4]))
        hd = summ.get(h0)
        if not hd:
            continue
        lo  = _fmt_p(hd["p25"])
        hi  = _fmt_p(hd["p75"])
        lbl = tf_labels.get(tf, tf)
        range_lines.append(f"  {lbl}: `{lo} — {hi}`")

    lines = [
        f"🔭 *SETUP RADAR — {e} {inst}*",
        f"`{_fmt_p(price)}` | {bias_str}",
        f"σ/yr `{sigma}%` ({vm}, {n_sims:,} sims, 3 TFs)",
        "",
        "*🎯 Watch These Levels:*",
        *setup_lines,
    ]
    if range_lines:
        lines += ["", "*📐 Intraday IQR (50% of paths):*"] + range_lines
    lines += ["", "_Entry not confirmed — monitor these levels._"]
    return "\n".join(lines)


# Horizon display map used by formatter (mirrors TF_HORIZONS in monte_carlo.py)
TF_HORIZONS_DISPLAY = {"4h": [2, 4, 6], "1h": [4, 8], "15min": [4, 8]}


def fmt_prospect_summary(inst, prospect):
    """
    Compact 2-line prospect block appended to /gold, /us30, /eurusd.
    Shows bias + top 2 watch levels with clean price formatting.
    """
    if not prospect or not prospect.get("probable_setups"):
        return ""
    bias   = prospect.get("bias", "NO CLEAR BIAS")
    p_up   = prospect.get("p_up", 50.0)
    if bias == "BULLISH":
        bias_e = f"🟢 MTF Sim: BULLISH ({p_up:.0f}% paths up)"
    elif bias == "BEARISH":
        bias_e = f"🔴 MTF Sim: BEARISH ({p_up:.0f}% paths up)"
    else:
        bias_e = f"🟡 MTF Sim: No edge ({p_up:.0f}/{100-p_up:.0f})"
    lines = [f"\n{bias_e}"]
    for ps in prospect["probable_setups"][:2]:
        d_arrow = "⬆️" if ps["direction"] == "LONG" else "⬇️"
        zp      = _fmt_p(ps["zone_price"])
        tc      = ps.get("tf_count", 1)
        star    = "★" if tc >= 2 else " "
        lines.append(f"  {d_arrow}{star} `{zp}` — {ps['density_pct']}% of paths")
    return "\n".join(lines)


def main_keyboard():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🥇 Gold",   callback_data="gold"),
         InlineKeyboardButton("🏦 US30",   callback_data="us30"),
         InlineKeyboardButton("💶 EURUSD", callback_data="eurusd")],
        [InlineKeyboardButton("📊 Summary",callback_data="summary"),
         InlineKeyboardButton("🎯 Setups", callback_data="setup")],
        [InlineKeyboardButton("📅 Weekly", callback_data="weekly"),
         InlineKeyboardButton("🔄 Refresh",callback_data="summary")],
    ])

async def job_daily_report(app):
    async with httpx.AsyncClient() as client:
        gn, dn = await asyncio.gather(fetch_news(client,"GLD"),fetch_news(client,"DIA"))
    hdr = (f"☀️ *QuantRisk Daily Report*\n"
           f"📅 {datetime.utcnow().strftime('%A, %d %B %Y')} | EAT 07:30\n{'─'*30}")
    await app.bot.send_message(TELEGRAM_CHAT_ID, hdr, parse_mode="Markdown")
    for inst, news in [("Gold",gn),("US30",dn)]:
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
        if not history: continue
        scores = [h[2] for h in history]
        from collections import Counter
        tv = Counter(h[3] for h in history).most_common(1)[0][0]
        e  = IE.get(inst,"📊")
        lines.append(f"{e} *{inst}*\n  Avg: `{round(mean(scores),1)}` | "
                     f"High: `{max(scores)}` | Low: `{min(scores)}`\n"
                     f"  Dominant: {VE.get(tv,'⚪')} {tv} | Points: {len(history)}\n")
    await app.bot.send_message(TELEGRAM_CHAT_ID, "\n".join(lines), parse_mode="Markdown")

def _mo(update):
    return update.message or update.callback_query.message

async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    g = "🟢" if (ws_mgr and ws_mgr.is_fresh("OANDA:XAU_USD")) else "🟡"
    eu= "🟢" if (ws_mgr and ws_mgr.is_fresh("OANDA:EUR_USD")) else "🟡"
    di= "🟢" if (ws_mgr and ws_mgr.is_fresh("DIA"))           else "🟡"
    ck= "✅ Cache warm" if (cache and cache.all_populated(CANDLE_SYMBOLS)) else "⏳ Warming..."
    await _mo(update).reply_text(
        f"👋 *QuantRisk Bot v5 — Three Instruments*\n\n"
        f"{g} Gold (XAU/USD) — 24/5 live\n{eu} EURUSD — 24/5 live\n"
        f"{di} US30 (DIA) — market hours\n{ck}\n\n*6 scoring models active ✅*",
        parse_mode="Markdown", reply_markup=main_keyboard())

async def _send_inst(update, inst):
    d = live_scores.get(inst)
    if not d:
        await _mo(update).reply_text(f"⏳ {inst} — warming up, try again in 30s.", reply_markup=main_keyboard()); return
    await _mo(update).reply_text(fmt_instrument(d), parse_mode="Markdown", reply_markup=main_keyboard())

async def cmd_gold(u,c):   await _send_inst(u,"Gold")
async def cmd_us30(u,c):   await _send_inst(u,"US30")
async def cmd_eurusd(u,c): await _send_inst(u,"EURUSD")

async def cmd_summary(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    sm, sl = get_session_multiplier()
    lines  = [f"📊 *QuantRisk Summary*\n⏰ {sl} (×{sm})\n"]
    for inst in ["Gold","US30","EURUSD"]:
        d = live_scores.get(inst); e = IE.get(inst,"📊")
        if not d: lines.append(f"{e} *{inst}*: ⏳ Loading..."); continue
        lines.append(f"{e} *{inst}*: `{d['composite']}/100` {VE.get(d['verdict'],'⚪')} {d['verdict']}\n"
                     f"  Conf: `{d['confidence']:.1f}%` | Persistent: {'✅' if d['persistent'] else '⏳'} 🔴 LIVE")
    await _mo(update).reply_text("\n".join(lines), parse_mode="Markdown", reply_markup=main_keyboard())

async def cmd_setup(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    found = False
    for inst in ["Gold","US30","EURUSD"]:
        d = live_scores.get(inst)
        if d and d.get("setup"):
            await _mo(update).reply_text(fmt_setup_alert(inst,d), parse_mode="Markdown"); found=True
    if not found:
        await _mo(update).reply_text("🟡 No high-confidence setups right now.\nNeed score >58 or <42 + confidence >70%.",
                                     reply_markup=main_keyboard())

async def cmd_report(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    async with httpx.AsyncClient() as client:
        gn,dn = await asyncio.gather(fetch_news(client,"GLD"),fetch_news(client,"DIA"))
    for inst,news in [("Gold",gn),("US30",dn)]:
        if d := live_scores.get(inst):
            await _mo(update).reply_text(fmt_instrument(d), parse_mode="Markdown")
        if cats := _fmt_news(news,inst):
            await _mo(update).reply_text(cats, parse_mode="Markdown")
    if d := live_scores.get("EURUSD"):
        await _mo(update).reply_text(fmt_instrument(d), parse_mode="Markdown")
    await _mo(update).reply_text("Report complete ✅", reply_markup=main_keyboard())

async def cmd_weekly(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not any(weekly_history.values()):
        await _mo(update).reply_text("📅 History building — check back later.", reply_markup=main_keyboard()); return
    await job_weekly_recap(ctx.application)

async def button_handler(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.callback_query.answer()
    h = {"gold":cmd_gold,"us30":cmd_us30,"eurusd":cmd_eurusd,
         "summary":cmd_summary,"setup":cmd_setup,"weekly":cmd_weekly}.get(update.callback_query.data)
    if h: await h(update, ctx)

def main():
    global cache, ws_mgr, app_ref
    cache  = CandleCache(TWELVEDATA_API_KEY, candle_counts=CANDLE_COUNTS)
    ws_mgr = WebSocketManager(FINNHUB_API_KEY, on_tick_callback=on_price_tick)

    async def post_init(application):
        global app_ref
        app_ref = application
        log.info("Warming candle cache...")
        await cache.warm_up(CANDLE_SYMBOLS, TIMEFRAMES)
        log.info("Cache warm. Starting Finnhub WebSocket...")
        asyncio.create_task(ws_mgr.start())
        asyncio.create_task(cache.refresh_loop(CANDLE_SYMBOLS, TIMEFRAMES))
        log.info("QuantRisk Bot v5 — Gold + US30 + EURUSD — fully live. 🚀")

    app = (Application.builder().token(TELEGRAM_BOT_TOKEN).post_init(post_init).build())
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
    scheduler.add_job(lambda: asyncio.create_task(job_daily_report(app)),
                      "cron", hour=7, minute=30, id="daily")
    scheduler.add_job(lambda: asyncio.create_task(job_weekly_recap(app)),
                      "cron", day_of_week="mon", hour=7, minute=30, id="weekly")
    scheduler.start()
    log.info("QuantRisk Bot v5 starting...")
    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()
