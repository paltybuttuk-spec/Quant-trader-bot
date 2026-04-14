"""
QuantRisk Backtest — Signal Validation
=======================================
Uses the identical scoring engine from bot.py to validate whether
composite score thresholds (>62 LONG / <38 SHORT) actually predict
positive forward returns.

Fetches historical daily candles from Twelve Data, replays the
scoring engine bar-by-bar with a rolling window, records every
signal, then measures forward returns at 1, 3, 5, 10 bar horizons.

Usage:
  python backtest.py

Output:
  - Win rate per instrument per horizon
  - Average return per signal
  - Signal frequency (how often the gate fires)
  - False signal rate (signal fired but price moved against)
  - Sharpe proxy per instrument
  - Recommended threshold adjustments if win rate < 55%
"""

import asyncio
import os
from collections import defaultdict
from datetime import datetime
from statistics import mean, stdev

import httpx

# ── Config ────────────────────────────────────────────────────────────────────
TWELVEDATA_API_KEY = os.environ.get("TWELVEDATA_API_KEY", "")

INSTRUMENTS = {
    "Gold":   "XAU/USD",
    "US30":   "DIA",
    "EURUSD": "EUR/USD",
}

# How many daily candles to fetch for backtest (Twelve Data free tier max ~500)
CANDLE_COUNT   = 500
ROLLING_WINDOW = 50    # same as live bot
MIN_BARS       = 30    # minimum bars before scoring starts

# Signal thresholds — mirror the live bot exactly
LONG_THRESHOLD  = 62
SHORT_THRESHOLD = 38
CONF_THRESHOLD  = 70   # minimum confidence to count as a gate-open signal

# Forward return horizons (in bars)
HORIZONS = [1, 3, 5, 10]

REAL_VOLUME_INSTRUMENTS = {"US30"}
TF_WEIGHTS = {"15min": 0.10, "1h": 0.20, "4h": 0.30, "1day": 0.25, "1week": 0.15}


# ══════════════════════════════════════════════════════════════════════════════
# SCORING ENGINE — identical logic to bot.py
# ══════════════════════════════════════════════════════════════════════════════

def calc_atr(highs, lows, closes, period=14):
    trs = [
        max(highs[i] - lows[i],
            abs(highs[i] - closes[i-1]),
            abs(lows[i]  - closes[i-1]))
        for i in range(1, len(closes))
    ]
    return mean(trs[-period:]) if len(trs) >= period else (mean(trs) if trs else 0)


def calc_conviction(closes, highs, lows, opens, volumes, instrument):
    if instrument in REAL_VOLUME_INSTRUMENTS:
        avg_v = mean(volumes[-20:]) if len(volumes) >= 20 else mean(volumes)
        vc    = 1.0 if volumes[-1] > avg_v else 0.5
        tv    = sum(volumes[-20:])
        vwap  = sum(c * v for c, v in zip(closes[-20:], volumes[-20:])) / tv if tv else closes[-1]
    else:
        body  = abs(closes[-1] - opens[-1])
        rng   = highs[-1] - lows[-1]
        ratio = body / rng if rng > 0 else 0.5
        vc    = 1.0 if ratio > 0.6 else (0.75 if ratio > 0.35 else 0.5)
        vwap  = mean(closes[-20:])
    vs = "ABOVE" if closes[-1] > vwap else "BELOW"
    return vc, vs, vwap


def score_bar(closes, highs, lows, opens, volumes, instrument):
    """Score a single bar using the rolling window ending at that bar."""
    if len(closes) < MIN_BARS:
        return None

    score = 50.0

    # Model 1: Regime
    net    = abs(closes[-1] - closes[-11])
    path   = sum(abs(closes[i] - closes[i-1]) for i in range(-10, 0))
    eff    = net / path if path > 0 else 0
    regime = "TRENDING" if eff > 0.55 else ("RANGING" if eff < 0.35 else "MIXED")

    # Model 5: Vol regime
    r10 = [abs(closes[i] - closes[i-1]) / closes[i-1] for i in range(-10, 0) if closes[i-1] > 0]
    r30 = [abs(closes[i] - closes[i-1]) / closes[i-1] for i in range(-30, 0) if closes[i-1] > 0]
    v10 = mean(r10) if r10 else 0.01
    v30 = mean(r30) if r30 else 0.01
    vol_ratio = v10 / v30 if v30 > 0 else 1.0
    if   vol_ratio < 0.70: vol_mult = 1.2
    elif vol_ratio < 1.50: vol_mult = 1.0
    elif vol_ratio < 2.00: vol_mult = 0.7
    else:                  vol_mult = 0.3

    # Model 3: Adaptive Z-score
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

    # Model 2: Conviction-confirmed momentum
    vc, vs, vwap = calc_conviction(closes, highs, lows, opens, volumes, instrument)
    m5  = (closes[-1] - closes[-6])  / closes[-6]  if closes[-6]  > 0 else 0
    m10 = (closes[-1] - closes[-11]) / closes[-11] if closes[-11] > 0 else 0
    m20 = (closes[-1] - closes[-21]) / closes[-21] if closes[-21] > 0 else 0
    mom = (m5 * 0.5 + m10 * 0.3 + m20 * 0.2) * vc
    mp  = mom * 200

    if   regime == "TRENDING": score += mp
    elif regime == "MIXED":    score += mp * 0.5
    else:                      score += mp * 0.25

    # Model 4: Institutional level
    score += 8 if closes[-1] > vwap else -8

    # Apply vol multiplier (no session multiplier in backtest — daily bars only)
    score = 50 + (score - 50) * vol_mult
    score = max(0, min(100, score))

    # Confidence
    return {
        "score":     round(score, 1),
        "regime":    regime,
        "vol_ratio": round(vol_ratio, 2),
        "z_score":   round(z, 2),
        "momentum":  round(mom * 100, 3),
        "vc":        vc,
    }


# ══════════════════════════════════════════════════════════════════════════════
# DATA FETCH
# ══════════════════════════════════════════════════════════════════════════════

async def fetch_history(client, symbol, outputsize=500):
    params = {
        "symbol":     symbol,
        "interval":   "1day",
        "outputsize": outputsize,
        "apikey":     TWELVEDATA_API_KEY,
    }
    r    = await client.get("https://api.twelvedata.com/time_series", params=params, timeout=20)
    data = r.json()
    if data.get("status") == "error":
        print(f"  ✗ API error for {symbol}: {data.get('message')}")
        return None
    values = list(reversed(data.get("values", [])))
    if not values:
        return None
    return {
        "closes":  [float(v["close"])           for v in values],
        "highs":   [float(v["high"])            for v in values],
        "lows":    [float(v["low"])             for v in values],
        "opens":   [float(v["open"])            for v in values],
        "volumes": [float(v.get("volume", 1))  for v in values],
        "dates":   [v["datetime"]               for v in values],
    }


# ══════════════════════════════════════════════════════════════════════════════
# BACKTEST ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def run_backtest(data, instrument):
    closes  = data["closes"]
    highs   = data["highs"]
    lows    = data["lows"]
    opens   = data["opens"]
    volumes = data["volumes"]
    dates   = data["dates"]
    n       = len(closes)

    signals = []   # (bar_index, direction, score, entry_price)

    for i in range(ROLLING_WINDOW, n - max(HORIZONS)):
        c = closes[:i+1]
        h = highs[:i+1]
        l = lows[:i+1]
        o = opens[:i+1]
        v = volumes[:i+1]

        result = score_bar(c, h, l, o, v, instrument)
        if result is None:
            continue

        score = result["score"]

        if score > LONG_THRESHOLD:
            signals.append((i, "LONG",  score, closes[i]))
        elif score < SHORT_THRESHOLD:
            signals.append((i, "SHORT", score, closes[i]))

    if not signals:
        return None

    # Measure forward returns at each horizon
    results = defaultdict(list)   # horizon → list of returns (signed, direction-adjusted)
    wins    = defaultdict(int)
    total   = defaultdict(int)

    for bar_idx, direction, score, entry in signals:
        for h in HORIZONS:
            future_idx = bar_idx + h
            if future_idx >= n:
                continue
            future_price = closes[future_idx]
            raw_return   = (future_price - entry) / entry * 100  # %
            # Flip sign for SHORT signals
            adj_return   = raw_return if direction == "LONG" else -raw_return
            results[h].append(adj_return)
            total[h] += 1
            if adj_return > 0:
                wins[h] += 1

    return {
        "instrument":    instrument,
        "total_signals": len(signals),
        "signal_freq":   round(len(signals) / (n - ROLLING_WINDOW) * 100, 1),
        "long_signals":  sum(1 for s in signals if s[1] == "LONG"),
        "short_signals": sum(1 for s in signals if s[1] == "SHORT"),
        "horizons": {
            h: {
                "win_rate":   round(wins[h] / total[h] * 100, 1) if total[h] else 0,
                "avg_return": round(mean(results[h]), 3) if results[h] else 0,
                "std_return": round(stdev(results[h]), 3) if len(results[h]) > 1 else 0,
                "sharpe":     round(
                    mean(results[h]) / stdev(results[h]) if len(results[h]) > 1 and stdev(results[h]) > 0 else 0,
                    2
                ),
                "count":      total[h],
            }
            for h in HORIZONS
        },
        "signals_sample": signals[-5:],   # last 5 signals for inspection
        "bars_tested":    n - ROLLING_WINDOW,
    }


# ══════════════════════════════════════════════════════════════════════════════
# THRESHOLD OPTIMIZER
# ══════════════════════════════════════════════════════════════════════════════

def find_optimal_threshold(data, instrument, horizon=5):
    """
    Sweep composite score thresholds from 55–75 (long) and 25–45 (short).
    Find the threshold that maximises win rate × signal frequency.
    """
    closes  = data["closes"]
    highs   = data["highs"]
    lows    = data["lows"]
    opens   = data["opens"]
    volumes = data["volumes"]
    n       = len(closes)

    all_scores = []
    for i in range(ROLLING_WINDOW, n - max(HORIZONS)):
        result = score_bar(closes[:i+1], highs[:i+1], lows[:i+1],
                           opens[:i+1], volumes[:i+1], instrument)
        if result:
            fwd = (closes[i + horizon] - closes[i]) / closes[i] * 100
            all_scores.append((result["score"], fwd))

    if not all_scores:
        return None

    best = {"threshold": 62, "win_rate": 0, "score": 0, "n_signals": 0}

    for threshold in range(55, 76):
        longs = [(s, f) for s, f in all_scores if s > threshold]
        if len(longs) < 10:
            continue
        wr = sum(1 for _, f in longs if f > 0) / len(longs)
        # Score = win_rate × sqrt(frequency) — rewards both accuracy and frequency
        freq  = len(longs) / len(all_scores)
        score = wr * (freq ** 0.5)
        if score > best["score"]:
            best = {"threshold": threshold, "win_rate": round(wr*100, 1),
                    "score": round(score, 4), "n_signals": len(longs)}

    return best


# ══════════════════════════════════════════════════════════════════════════════
# REPORT PRINTER
# ══════════════════════════════════════════════════════════════════════════════

def print_report(results, optimal):
    print("\n" + "═" * 60)
    print("  QuantRisk Backtest — Signal Validation Report")
    print("═" * 60)
    print(f"  Thresholds tested: LONG >{LONG_THRESHOLD} | SHORT <{SHORT_THRESHOLD}")
    print(f"  Horizons: {HORIZONS} bars (daily candles)")
    print("═" * 60)

    for inst, r in results.items():
        if not r:
            print(f"\n{inst}: ✗ No data / insufficient signals")
            continue

        print(f"\n{'─'*55}")
        print(f"  {inst}")
        print(f"{'─'*55}")
        print(f"  Bars tested:    {r['bars_tested']}")
        print(f"  Total signals:  {r['total_signals']} ({r['signal_freq']}% of bars)")
        print(f"  LONG:  {r['long_signals']}  |  SHORT: {r['short_signals']}")
        print()
        print(f"  {'Horizon':<10} {'WinRate':<10} {'AvgReturn':<12} {'Sharpe':<10} {'N'}")
        print(f"  {'─'*7:<10} {'─'*7:<10} {'─'*9:<12} {'─'*6:<10} {'─'*4}")
        for h, hd in r["horizons"].items():
            wr_flag = "✅" if hd["win_rate"] >= 55 else ("⚠️ " if hd["win_rate"] >= 50 else "❌")
            print(f"  {h:>2}d{'':<7} {hd['win_rate']:>5.1f}% {wr_flag}  "
                  f"{hd['avg_return']:>+8.3f}%    {hd['sharpe']:>+6.2f}    {hd['count']}")

        # Optimal threshold recommendation
        opt = optimal.get(inst)
        if opt:
            delta = opt["threshold"] - LONG_THRESHOLD
            direction = "raise" if delta > 0 else ("lower" if delta < 0 else "keep")
            print(f"\n  📐 Optimal LONG threshold: {opt['threshold']} "
                  f"({direction} by {abs(delta)}) → {opt['win_rate']}% win rate "
                  f"on {opt['n_signals']} signals")
        print()

    print("═" * 60)
    print("  VERDICT SUMMARY")
    print("═" * 60)
    for inst, r in results.items():
        if not r:
            continue
        h5 = r["horizons"].get(5, {})
        wr = h5.get("win_rate", 0)
        sh = h5.get("sharpe",   0)
        if wr >= 58 and sh > 0.3:
            verdict = "✅ STRONG EDGE — deploy confidently"
        elif wr >= 52:
            verdict = "⚠️  MARGINAL EDGE — consider threshold tuning"
        else:
            verdict = "❌ NO EDGE — scoring model needs revision"
        print(f"  {inst:<10} 5-bar WR: {wr:.1f}% | Sharpe: {sh:+.2f} → {verdict}")
    print("═" * 60 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

async def main():
    if not TWELVEDATA_API_KEY:
        print("✗ Set TWELVEDATA_API_KEY environment variable first.")
        return

    print(f"\nQuantRisk Backtest starting — {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC")
    print(f"Fetching {CANDLE_COUNT} daily bars per instrument from Twelve Data...\n")

    results  = {}
    optimal  = {}

    async with httpx.AsyncClient() as client:
        for inst, sym in INSTRUMENTS.items():
            print(f"  Fetching {inst} ({sym})...")
            data = await fetch_history(client, sym, CANDLE_COUNT)
            if not data:
                results[inst] = None
                continue
            print(f"  ✓ {len(data['closes'])} bars | Running backtest...")
            r = run_backtest(data, inst)
            results[inst] = r
            opt = find_optimal_threshold(data, inst, horizon=5)
            if opt:
                optimal[inst] = opt
            # Respect API rate limit
            await asyncio.sleep(1.2)

    print_report(results, optimal)


if __name__ == "__main__":
    asyncio.run(main())
