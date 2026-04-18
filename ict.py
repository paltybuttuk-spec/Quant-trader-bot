"""
ict.py — ICT Concepts + Support/Resistance
============================================
All functions operate on standard OHLCV lists (oldest → newest index).
No external dependencies — pure Python stdlib only.

Public API:
  find_fvg(closes, highs, lows, lookback)             → list[dict]
  find_order_blocks(closes, highs, lows, opens, ...)  → list[dict]
  detect_bos_choch(closes, highs, lows)               → dict
  premium_discount(closes, highs, lows, lookback)     → dict
  ict_score_delta(closes, highs, lows, opens, bias)   → (float, dict)
  enrich_setup_with_ict(setup, closes, highs, lows, opens, tf) → dict

ict_score_delta() is the main entry point for score_timeframe() in bot.py.
Returns a score adjustment clamped to [-20, +20] and a breakdown dict.
"""

from statistics import mean
from typing import Optional

# ── Tuning constants ──────────────────────────────────────────────────────────
FVG_LOOKBACK      = 30
OB_LOOKBACK       = 30
OB_IMPULSE_MULT   = 1.5
OB_PROXIMITY_ATR  = 0.5
BOS_SWING_WING    = 3
PD_LOOKBACK       = 20

DELTA_OB_STRONG   = 10
DELTA_FVG         = 8
DELTA_CHOCH       = 8
DELTA_BOS         = 6
DELTA_PD          = 5
DELTA_CONFLICT    = -4


# ══════════════════════════════════════════════════════════════════════════════
# FAIR VALUE GAPS
# ══════════════════════════════════════════════════════════════════════════════

def find_fvg(closes: list, highs: list, lows: list,
             lookback: int = FVG_LOOKBACK) -> list:
    n    = len(closes)
    fvgs = []
    if n < 5:
        return fvgs
    trs = [highs[i] - lows[i] for i in range(max(0, n - 21), n)]
    atr_est = mean(trs) if trs else 1.0
    start   = max(0, n - lookback - 2)
    current = closes[-1]
    for i in range(start, n - 2):
        c1_high = highs[i];  c1_low  = lows[i]
        c3_low  = lows[i+2]; c3_high = highs[i+2]
        if c3_low > c1_high:
            size = c3_low - c1_high
            if size < atr_est * 0.05: continue
            fvgs.append({"type": "BULLISH", "top": round(c3_low, 5), "bottom": round(c1_high, 5),
                         "midpoint": round((c3_low + c1_high) / 2, 5),
                         "filled": current < c3_low, "age": n - 3 - i,
                         "size_atr": round(size / atr_est, 3)})
        elif c3_high < c1_low:
            size = c1_low - c3_high
            if size < atr_est * 0.05: continue
            fvgs.append({"type": "BEARISH", "top": round(c1_low, 5), "bottom": round(c3_high, 5),
                         "midpoint": round((c1_low + c3_high) / 2, 5),
                         "filled": current > c1_low, "age": n - 3 - i,
                         "size_atr": round(size / atr_est, 3)})
    fvgs.sort(key=lambda x: x["age"])
    return fvgs


def price_in_fvg(price: float, fvgs: list) -> Optional[dict]:
    for fvg in fvgs:
        if not fvg["filled"] and fvg["bottom"] <= price <= fvg["top"]:
            return fvg
    return None


# ══════════════════════════════════════════════════════════════════════════════
# ORDER BLOCKS
# ══════════════════════════════════════════════════════════════════════════════

def find_order_blocks(closes: list, highs: list, lows: list, opens: list,
                      lookback: int = OB_LOOKBACK,
                      impulse_mult: float = OB_IMPULSE_MULT) -> list:
    n   = len(closes)
    obs = []
    if n < 5:
        return obs
    start   = max(1, n - lookback)
    rets    = [abs(closes[i] - closes[i-1]) for i in range(1, n)]
    avg     = mean(rets[-20:]) if len(rets) >= 20 else (mean(rets) if rets else 1.0)
    current = closes[-1]
    for i in range(start, n - 1):
        move = abs(closes[i+1] - closes[i])
        if move < avg * impulse_mult:
            continue
        if closes[i] < opens[i] and closes[i+1] > opens[i+1]:
            ob_top = opens[i]; ob_bot = closes[i]
            obs.append({"type": "BULLISH_OB", "top": round(ob_top, 5), "bottom": round(ob_bot, 5),
                        "midpoint": round((ob_top + ob_bot) / 2, 5),
                        "age": n - 2 - i, "impulse": round(move / avg, 2),
                        "mitigated": current < ob_top})
        elif closes[i] > opens[i] and closes[i+1] < opens[i+1]:
            ob_top = closes[i]; ob_bot = opens[i]
            obs.append({"type": "BEARISH_OB", "top": round(ob_top, 5), "bottom": round(ob_bot, 5),
                        "midpoint": round((ob_top + ob_bot) / 2, 5),
                        "age": n - 2 - i, "impulse": round(move / avg, 2),
                        "mitigated": current > ob_bot})
    obs.sort(key=lambda x: x["age"])
    return obs


# ══════════════════════════════════════════════════════════════════════════════
# BREAK OF STRUCTURE / CHANGE OF CHARACTER
# ══════════════════════════════════════════════════════════════════════════════

def detect_bos_choch(closes: list, highs: list, lows: list,
                     swing_wing: int = BOS_SWING_WING) -> dict:
    n = len(closes)
    empty = {"bos": None, "choch": None, "structure": "UNKNOWN", "last_sh": None, "last_sl": None}
    if n < swing_wing * 2 + 5:
        return empty
    sh, sl = [], []
    for i in range(swing_wing, n - swing_wing):
        if all(highs[i] >= highs[j] for j in range(i - swing_wing, i + swing_wing + 1) if j != i):
            sh.append((i, highs[i]))
        if all(lows[i]  <= lows[j]  for j in range(i - swing_wing, i + swing_wing + 1) if j != i):
            sl.append((i, lows[i]))
    if len(sh) < 2 or len(sl) < 2:
        return empty
    prev_sh, last_sh = sh[-2][1], sh[-1][1]
    prev_sl, last_sl = sl[-2][1], sl[-1][1]
    current = closes[-1]
    bos = choch = None
    if current > last_sh and last_sh > prev_sh:
        bos = {"direction": "BULLISH", "level": round(last_sh, 5), "type": "BOS"}
    elif current < last_sl and last_sl < prev_sl:
        bos = {"direction": "BEARISH", "level": round(last_sl, 5), "type": "BOS"}
    if current > prev_sh and last_sh < prev_sh:
        choch = {"direction": "BULLISH", "level": round(prev_sh, 5), "type": "CHOCH"}
    elif current < prev_sl and last_sl > prev_sl:
        choch = {"direction": "BEARISH", "level": round(prev_sl, 5), "type": "CHOCH"}
    if   last_sh > prev_sh and last_sl > prev_sl: structure = "BULLISH"
    elif last_sh < prev_sh and last_sl < prev_sl: structure = "BEARISH"
    else:                                          structure = "RANGING"
    return {"bos": bos, "choch": choch, "structure": structure,
            "last_sh": round(last_sh, 5), "last_sl": round(last_sl, 5)}


# ══════════════════════════════════════════════════════════════════════════════
# PREMIUM / DISCOUNT
# ══════════════════════════════════════════════════════════════════════════════

def premium_discount(closes: list, highs: list, lows: list,
                     lookback: int = PD_LOOKBACK) -> dict:
    n  = len(closes)
    lb = min(lookback, n)
    if lb < 2:
        p = closes[-1] if closes else 0
        return {"zone": "EQUILIBRIUM", "ratio": 0.5, "equilibrium": p, "range_high": p, "range_low": p}
    h   = max(highs[-lb:]); l = min(lows[-lb:])
    rng = h - l
    if rng <= 0:
        return {"zone": "EQUILIBRIUM", "ratio": 0.5, "equilibrium": closes[-1], "range_high": h, "range_low": l}
    ratio = (closes[-1] - l) / rng
    zone  = "PREMIUM" if ratio > 0.70 else ("DISCOUNT" if ratio < 0.30 else "EQUILIBRIUM")
    return {"zone": zone, "ratio": round(ratio, 3), "equilibrium": round(l + rng * 0.5, 5),
            "range_high": round(h, 5), "range_low": round(l, 5)}


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def ict_score_delta(closes: list, highs: list, lows: list, opens: list,
                    bias: int, lookback: int = FVG_LOOKBACK) -> tuple:
    """
    Compute ICT-based score adjustment for one timeframe.

    bias: +1 = bullish signal active, -1 = bearish, 0 = neutral

    Returns (delta: float clamped [-20,+20], breakdown: dict)
    """
    n = len(closes)
    empty_ict = {"fvgs": [], "order_blocks": [],
                 "bos": {"bos": None, "choch": None, "structure": "UNKNOWN"},
                 "premium_discount": {"zone": "EQUILIBRIUM", "ratio": 0.5},
                 "tags": [], "delta": 0.0, "structure": "UNKNOWN"}
    if n < 10:
        return 0.0, empty_ict

    delta = 0.0
    tags  = []

    atr_trs = [highs[i] - lows[i] for i in range(max(0, n - 15), n)]
    atr     = mean(atr_trs) if atr_trs else 1.0

    fvgs = find_fvg(closes, highs, lows, lookback)
    obs  = find_order_blocks(closes, highs, lows, opens, lookback)
    bos  = detect_bos_choch(closes, highs, lows)
    pd   = premium_discount(closes, highs, lows)
    c    = closes[-1]

    # 1. Order Blocks
    for ob in obs[:5]:
        if ob["mitigated"]:
            continue
        at_ob = abs(c - ob["midpoint"]) / atr <= OB_PROXIMITY_ATR
        if ob["type"] == "BULLISH_OB":
            if at_ob and bias >= 0:
                delta += DELTA_OB_STRONG
                tags.append(f"At Bullish OB {ob['bottom']}–{ob['top']} (×{ob['impulse']}) 🟢")
                break
            elif at_ob and bias < 0:
                delta += DELTA_CONFLICT
                tags.append(f"At Bullish OB — opposes bearish bias ⚠️")
        elif ob["type"] == "BEARISH_OB":
            if at_ob and bias <= 0:
                delta -= DELTA_OB_STRONG
                tags.append(f"At Bearish OB {ob['bottom']}–{ob['top']} (×{ob['impulse']}) 🔴")
                break
            elif at_ob and bias > 0:
                delta -= DELTA_CONFLICT
                tags.append(f"At Bearish OB — opposes bullish bias ⚠️")

    # 2. Fair Value Gaps
    for fvg in fvgs[:5]:
        if fvg["filled"]:
            continue
        in_gap = fvg["bottom"] <= c <= fvg["top"]
        if not in_gap:
            continue
        if fvg["type"] == "BULLISH" and bias >= 0:
            delta += DELTA_FVG
            tags.append(f"In Bullish FVG {fvg['bottom']}–{fvg['top']} ({fvg['size_atr']:.2f}ATR) 🟢")
        elif fvg["type"] == "BEARISH" and bias <= 0:
            delta -= DELTA_FVG
            tags.append(f"In Bearish FVG {fvg['bottom']}–{fvg['top']} ({fvg['size_atr']:.2f}ATR) 🔴")
        elif fvg["type"] == "BULLISH" and bias < 0:
            delta += DELTA_CONFLICT
            tags.append(f"In Bullish FVG — opposes bearish bias ⚠️")
        elif fvg["type"] == "BEARISH" and bias > 0:
            delta -= DELTA_CONFLICT
            tags.append(f"In Bearish FVG — opposes bullish bias ⚠️")
        break

    # 3. BOS / CHOCH
    if bos["choch"]:
        if bos["choch"]["direction"] == "BULLISH" and bias >= 0:
            delta += DELTA_CHOCH; tags.append(f"CHOCH Bullish @ {bos['choch']['level']} 🔄")
        elif bos["choch"]["direction"] == "BEARISH" and bias <= 0:
            delta -= DELTA_CHOCH; tags.append(f"CHOCH Bearish @ {bos['choch']['level']} 🔄")
    if bos["bos"]:
        if bos["bos"]["direction"] == "BULLISH" and bias >= 0:
            delta += DELTA_BOS; tags.append(f"BOS Bullish @ {bos['bos']['level']} ✅")
        elif bos["bos"]["direction"] == "BEARISH" and bias <= 0:
            delta -= DELTA_BOS; tags.append(f"BOS Bearish @ {bos['bos']['level']} ✅")

    # 4. Premium / Discount
    if pd["zone"] == "DISCOUNT" and bias >= 0:
        delta += DELTA_PD; tags.append(f"Discount zone ({pd['ratio']:.2f}) 🟢")
    elif pd["zone"] == "PREMIUM" and bias <= 0:
        delta -= DELTA_PD; tags.append(f"Premium zone ({pd['ratio']:.2f}) 🔴")
    elif pd["zone"] == "DISCOUNT" and bias < 0:
        delta -= 3; tags.append(f"Discount but bearish — fighting structure ⚠️")
    elif pd["zone"] == "PREMIUM" and bias > 0:
        delta -= 3; tags.append(f"Premium but bullish — overbought risk ⚠️")

    delta = max(-20.0, min(20.0, delta))
    return round(delta, 1), {
        "fvgs": fvgs[:3], "order_blocks": obs[:3], "bos": bos,
        "premium_discount": pd, "tags": tags,
        "delta": round(delta, 1), "structure": bos["structure"], "atr": round(atr, 5),
    }


def enrich_setup_with_ict(setup: dict, closes: list, highs: list,
                           lows: list, opens: list, tf: str = "1day") -> dict:
    """Add ICT context to an existing setup dict (informational — no level changes)."""
    if not setup or not closes:
        return setup
    direction = setup.get("direction", "LONG")
    entry_mid = (setup["entry_low"] + setup["entry_high"]) / 2
    bias      = 1 if direction == "LONG" else -1
    _, ict    = ict_score_delta(closes, highs, lows, opens, bias)

    # Nearest unfilled FVG to entry
    nearest_fvg, best_dist = None, float("inf")
    for fvg in ict["fvgs"]:
        if fvg["filled"]: continue
        d = abs(fvg["midpoint"] - entry_mid)
        if d < best_dist: best_dist = d; nearest_fvg = fvg

    # Nearest unmitigated OB to entry
    nearest_ob, best_dist = None, float("inf")
    for ob in ict["order_blocks"]:
        if ob["mitigated"]: continue
        d = abs(ob["midpoint"] - entry_mid)
        if d < best_dist: best_dist = d; nearest_ob = ob

    setup["ict"] = {
        "structure": ict["structure"], "bos": ict["bos"]["bos"], "choch": ict["bos"]["choch"],
        "premium_discount": ict["premium_discount"], "nearest_fvg": nearest_fvg,
        "nearest_ob": nearest_ob, "tags": ict["tags"], "delta": ict["delta"], "timeframe": tf,
    }
    return setup
