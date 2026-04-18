"""
monte_carlo.py
Monte Carlo price path simulation for QuantRisk Bot v5.

Architecture:
- GARCH(1,1) volatility model — captures vol clustering on trending instruments
- Realized vol fallback if GARCH fails (insufficient data / numerical instability)
- 10,000 simulations per setup, run in asyncio executor (non-blocking)
- Geometric Brownian Motion paths with GARCH-estimated sigma

Output per setup:
  p_t1        — probability of hitting T1 before stop
  p_t2        — probability of hitting T2 before stop
  p_t3        — probability of hitting T3 before stop
  p_stop      — probability of hitting stop first
  ev          — expected value in R-multiples (risk = entry to stop = 1R)
  sharpe      — mean outcome / std of outcomes across all sims
  vol_model   — "GARCH" or "REALIZED" (whichever was used)
  sigma_daily — annualised daily vol estimate used
  win_rate    — % of sims that hit at least T1
"""

import asyncio
import logging
import math
import random
from concurrent.futures import ThreadPoolExecutor
from statistics import mean, stdev

log = logging.getLogger(__name__)

# ── Simulation constants ──────────────────────────────────────────────────────
N_SIMS       = 10_000   # simulations per setup
MAX_STEPS    = 200      # max candles simulated per path (prevents infinite loops)
GARCH_MIN    = 30       # minimum returns needed for GARCH fit
TRADING_YEAR = 252      # trading days per year

# Shared thread pool — reused across all MC calls, max 2 workers (Railway free tier)
_executor = ThreadPoolExecutor(max_workers=2)


# ─── GARCH(1,1) Estimator ─────────────────────────────────────────────────────

def _log_returns(closes: list) -> list:
    """Compute log returns from close prices."""
    return [math.log(closes[i] / closes[i - 1])
            for i in range(1, len(closes))
            if closes[i - 1] > 0 and closes[i] > 0]


def fit_garch(closes: list) -> tuple[float, bool]:
    """
    Fit GARCH(1,1) via moment-matching / variance targeting.
    Returns (sigma_per_step, success_bool).

    Model: sigma²_t = omega + alpha * r²_{t-1} + beta * sigma²_{t-1}
    We estimate omega, alpha, beta using a simplified MLE grid search
    that is numerically stable without scipy/arch dependency.

    Falls back gracefully on any failure.
    """
    try:
        rets = _log_returns(closes)
        if len(rets) < GARCH_MIN:
            return _realized_vol(closes), False

        n       = len(rets)
        var_unc = sum(r ** 2 for r in rets) / n   # unconditional variance

        # Grid search over alpha, beta — omega implied by variance targeting
        best_ll  = -1e18
        best_sig = math.sqrt(var_unc)

        alphas = [0.05, 0.08, 0.10, 0.12, 0.15]
        betas  = [0.80, 0.83, 0.85, 0.87, 0.90]

        for alpha in alphas:
            for beta in betas:
                if alpha + beta >= 0.9999:
                    continue
                omega = var_unc * (1 - alpha - beta)
                if omega <= 0:
                    continue

                # Compute conditional variances and log-likelihood
                h = [var_unc]
                ll = 0.0
                valid = True
                for i, r in enumerate(rets):
                    ht = omega + alpha * rets[i - 1] ** 2 + beta * h[-1] if i > 0 else var_unc
                    if ht <= 0:
                        valid = False
                        break
                    h.append(ht)
                    ll += -0.5 * (math.log(2 * math.pi) + math.log(ht) + r ** 2 / ht)

                if valid and ll > best_ll:
                    best_ll  = ll
                    # Use last conditional sigma as the forward estimate
                    best_sig = math.sqrt(h[-1])

        return best_sig, True

    except Exception as e:
        log.debug(f"GARCH fit failed: {e}")
        return _realized_vol(closes), False


def _realized_vol(closes: list) -> float:
    """
    Fallback: realized volatility from last 20 log returns.
    Returns per-step sigma (same scale as close prices).
    """
    try:
        rets = _log_returns(closes[-21:])
        if len(rets) < 2:
            return closes[-1] * 0.001   # 0.1% per step — minimal fallback
        return stdev(rets)
    except Exception:
        return closes[-1] * 0.001


# ─── Core simulation (runs in thread) ─────────────────────────────────────────

def _simulate(entry: float, stop: float, t1: float, t2: float, t3: float,
              direction: str, sigma: float, n_sims: int) -> dict:
    """
    Run n_sims GBM price paths from entry.
    Each path steps forward until it hits stop, t3, or MAX_STEPS.

    Returns raw outcome counts and R-multiple list for EV/Sharpe.
    """
    risk  = abs(entry - stop)
    if risk <= 0:
        risk = entry * 0.001   # guard against zero risk

    # R-multiples for each target and stop
    r_t1   =  abs(t1 - entry) / risk
    r_t2   =  abs(t2 - entry) / risk
    r_t3   =  abs(t3 - entry) / risk
    r_stop = -1.0              # always -1R

    hit_t1 = hit_t2 = hit_t3 = hit_stop = hit_none = 0
    outcomes = []

    is_long = direction == "LONG"
    drift   = 0.0   # zero drift — conservative, let volatility speak

    for _ in range(n_sims):
        price = entry
        result_r = 0.0
        resolved = False

        for _ in range(MAX_STEPS):
            # GBM step: price * exp((drift - 0.5*sigma²)*dt + sigma*Z)
            z       = random.gauss(0, 1)
            price  *= math.exp(drift + sigma * z)

            if is_long:
                if price <= stop:
                    result_r = r_stop; hit_stop += 1; resolved = True; break
                elif price >= t3:
                    result_r = r_t3;   hit_t3   += 1; resolved = True; break
                elif price >= t2:
                    result_r = r_t2;   hit_t2   += 1; resolved = True; break
                elif price >= t1:
                    result_r = r_t1;   hit_t1   += 1; resolved = True; break
            else:
                if price >= stop:
                    result_r = r_stop; hit_stop += 1; resolved = True; break
                elif price <= t3:
                    result_r = r_t3;   hit_t3   += 1; resolved = True; break
                elif price <= t2:
                    result_r = r_t2;   hit_t2   += 1; resolved = True; break
                elif price <= t1:
                    result_r = r_t1;   hit_t1   += 1; resolved = True; break

        if not resolved:
            hit_none += 1
            result_r  = 0.0   # flat — neither target nor stop hit

        outcomes.append(result_r)

    total = n_sims
    ev    = mean(outcomes)
    try:
        sh = ev / stdev(outcomes) if stdev(outcomes) > 0 else 0.0
    except Exception:
        sh = 0.0

    return {
        "p_t1":       round(hit_t1   / total * 100, 1),
        "p_t2":       round(hit_t2   / total * 100, 1),
        "p_t3":       round(hit_t3   / total * 100, 1),
        "p_stop":     round(hit_stop / total * 100, 1),
        "p_none":     round(hit_none / total * 100, 1),
        "win_rate":   round((hit_t1 + hit_t2 + hit_t3) / total * 100, 1),
        "ev":         round(ev, 3),
        "sharpe":     round(sh, 3),
        "r_t1":       round(r_t1, 2),
        "r_t2":       round(r_t2, 2),
        "r_t3":       round(r_t3, 2),
        "n_sims":     n_sims,
    }


# ─── Public async entry point ─────────────────────────────────────────────────

async def run_monte_carlo(setup: dict, closes: list) -> dict:
    """
    Async wrapper — runs simulation in thread executor so bot stays responsive.

    Args:
        setup:  calc_setup() output dict (must have entry_low/high, stop, t1/t2/t3, direction)
        closes: list of close prices (used for vol estimation)

    Returns:
        mc dict with all simulation stats + vol_model label.
        On any failure returns a safe fallback dict with zeroed stats.
    """
    try:
        entry = (setup["entry_low"] + setup["entry_high"]) / 2

        # ── Vol estimation ────────────────────────────────────────────────────
        sigma, garch_ok = fit_garch(closes)
        vol_model = "GARCH" if garch_ok else "REALIZED"

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            _executor,
            _simulate,
            entry,
            setup["stop"],
            setup["t1"],
            setup["t2"],
            setup["t3"],
            setup["direction"],
            sigma,
            N_SIMS,
        )

        result["vol_model"]   = vol_model
        result["sigma_daily"] = round(sigma * math.sqrt(TRADING_YEAR) * 100, 2)
        log.debug(f"MC done — EV={result['ev']}R | Win={result['win_rate']}% | {vol_model}")
        return result

    except Exception as e:
        log.error(f"Monte Carlo failed: {e}")
        return _mc_fallback()


def _mc_fallback() -> dict:
    return {
        "p_t1": 0.0, "p_t2": 0.0, "p_t3": 0.0,
        "p_stop": 0.0, "p_none": 0.0, "win_rate": 0.0,
        "ev": 0.0, "sharpe": 0.0,
        "r_t1": 0.0, "r_t2": 0.0, "r_t3": 0.0,
        "n_sims": 0, "vol_model": "FAILED", "sigma_daily": 0.0,
    }


# ══════════════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════════════
# PROSPECTIVE SCAN — multi-timeframe, day-trading configured
# ══════════════════════════════════════════════════════════════════════════════
#
# Architecture:
#   Runs on THREE timeframes simultaneously (1H for bias, 15min for entry
#   precision, 4H for structure). Each TF has its own sigma estimate and
#   horizon set. Results are synthesised into a single output.
#
#   TF roles:
#     4H  — macro directional bias (where is price going this session?)
#     1H  — probable destination zones (which levels will price reach today?)
#     15M — entry precision (tight range for the next 1–2 hours)
#
#   Bias threshold: only shown when ≥58% of paths lean one way.
#   WEAK S/R filtered out — only MODERATE/STRONG count as confluence.
#   Decimal places: Gold/indices 2dp, forex 5dp — applied at formatter level.

# ── Per-timeframe horizons (in bars of that TF) ───────────────────────────────
# 4H  : 2 bars = 8H (next session), 4 = full day, 6 = day+half
# 1H  : 4 bars = 4H (intraday), 8 = full session
# 15M : 4 bars = 1H, 8 = 2H  (tight precision window)
TF_HORIZONS = {
    "4h":    [2, 4, 6],
    "1h":    [4, 8],
    "15min": [4, 8],
}

# Bucket width — fraction of one sigma step. Smaller = tighter, more precise.
TF_BUCKET = {
    "4h":    2.5,
    "1h":    2.0,
    "15min": 1.5,
}

# Minimum path density to flag a hot zone
HOT_ZONE_MIN_PCT = 10.0

# Bias only shown when ≥ this % of paths lean one way
BIAS_THRESHOLD = 58.0

# Confluence radius in ATR fractions
CONFLUENCE_ATR = 0.6

# WEAK S/R excluded — only meaningful levels count
SR_MIN_STRENGTH = {"STRONG", "MODERATE"}

# Timeframe weights for synthesising cross-TF bias
TF_BIAS_WEIGHT = {"4h": 0.50, "1h": 0.35, "15min": 0.15}


def _fmt_p(price: float) -> str:
    """Round price appropriately: >10 = 2dp (Gold/indices), ≤10 = 5dp (forex)."""
    return f"{price:.2f}" if price > 10 else f"{price:.5f}"


def _paths_for_tf(start: float, sigma: float, horizons: list, n_sims: int) -> dict:
    """Run GBM paths for one timeframe. Returns {horizon: [end_prices]}."""
    h_set  = set(horizons)
    max_h  = max(horizons)
    result = {h: [] for h in horizons}
    for _ in range(n_sims):
        p = start
        for step in range(1, max_h + 1):
            p *= math.exp(sigma * random.gauss(0, 1))
            if step in h_set:
                result[step].append(p)
    return result


def _summarise_paths(paths: dict, start: float) -> dict:
    """
    Compute stats per horizon. Uses P25/P75 (interquartile range) for display —
    tight enough to be actionable for a day trader, not the misleading P5/P95 fan.
    """
    out = {}
    for h, prices in paths.items():
        if not prices:
            continue
        n        = len(prices)
        sp       = sorted(prices)
        p_up     = sum(1 for p in prices if p > start) / n * 100
        out[h] = {
            "p_up":   round(p_up, 1),
            "p_dn":   round(100 - p_up, 1),
            "median": sp[n // 2],
            "p25":    sp[max(0, int(n * 0.25))],
            "p75":    sp[min(n - 1, int(n * 0.75))],
            "p10":    sp[max(0, int(n * 0.10))],
            "p90":    sp[min(n - 1, int(n * 0.90))],
        }
    return out


def _hot_zones(prices: list, start: float, sigma: float, tf: str) -> list:
    """Build density map for one TF. Returns sorted list of hot zones."""
    if not prices:
        return []
    bw = TF_BUCKET.get(tf, 2.0) * sigma * start
    if bw <= 0:
        return []
    counts = {}
    for p in prices:
        b = round(p / bw) * bw
        counts[b] = counts.get(b, 0) + 1
    total = len(prices)
    zones = []
    for b, cnt in counts.items():
        pct = cnt / total * 100
        if pct >= HOT_ZONE_MIN_PCT:
            zones.append({"price": b, "pct": round(pct, 1),
                          "direction": "UP" if b > start else "DOWN"})
    zones.sort(key=lambda z: z["pct"], reverse=True)
    return zones


def _match_confluence(zones: list, sr: dict, fvgs: list,
                      obs: list, atr: float, start: float) -> list:
    """
    Match hot zones to structural levels (S/R, FVG, OB).
    WEAK S/R filtered. Returns probable setups sorted by confluence count.
    """
    if atr <= 0 or not zones:
        return []
    radius = CONFLUENCE_ATR * atr
    levels = []
    for z in sr.get("support", []):
        if z["strength"] in SR_MIN_STRENGTH:
            levels.append(("SR_S", z["price"], z["strength"], z["touches"]))
    for z in sr.get("resistance", []):
        if z["strength"] in SR_MIN_STRENGTH:
            levels.append(("SR_R", z["price"], z["strength"], z["touches"]))
    for f in fvgs:
        if not f.get("filled"):
            levels.append(("FVG", f["midpoint"], f["type"], 0))
    for ob in obs:
        if not ob.get("mitigated"):
            levels.append(("OB", ob["midpoint"], f"×{ob['impulse']}", 0))

    setups = []
    seen   = set()
    for z in zones:
        zp  = z["price"]
        dir_ = "LONG" if zp > start else "SHORT"
        hit  = [l for l in levels if abs(zp - l[1]) <= radius]
        if not hit:
            continue
        key = (dir_, round(zp / max(atr * 0.3, 0.001)))
        if key in seen:
            continue
        seen.add(key)
        tags = []
        for ltype, lp, lstr, ltch in hit:
            fp = _fmt_p(lp)
            if   ltype == "SR_S": tags.append(f"🟢 Support {lstr} ({ltch}T) @ {fp}")
            elif ltype == "SR_R": tags.append(f"🔴 Resistance {lstr} ({ltch}T) @ {fp}")
            elif ltype == "FVG":  tags.append(f"⬜ FVG {lstr} @ {fp}")
            elif ltype == "OB":   tags.append(f"🟦 OB {lstr} @ {fp}")
        setups.append({
            "direction":   dir_,
            "zone_price":  zp,
            "density_pct": z["pct"],
            "confluence":  tags,
            "n_levels":    len(hit),
        })
    setups.sort(key=lambda s: (-s["n_levels"], -s["density_pct"]))
    return setups


def _run_mtf_prospective(
    start: float,
    sigmas: dict,        # {tf: sigma}
    n_sims: int,
    sr_by_tf: dict,      # {tf: sr_levels_dict}
    fvgs_by_tf: dict,    # {tf: [fvg_list]}
    obs_by_tf: dict,     # {tf: [ob_list]}
    atrs: dict,          # {tf: atr_value}
) -> dict:
    """
    Multi-timeframe prospective core — runs in thread executor.
    Each TF simulated independently, results synthesised.
    """
    tf_results = {}

    for tf in ["4h", "1h", "15min"]:
        sigma = sigmas.get(tf)
        if not sigma:
            continue
        horizons  = TF_HORIZONS[tf]
        paths     = _paths_for_tf(start, sigma, horizons, n_sims)
        summary   = _summarise_paths(paths, start)
        # Use shortest horizon for density map (most concentrated zones)
        short_h   = min(horizons)
        zones     = _hot_zones(paths[short_h], start, sigma, tf)
        setups    = _match_confluence(
            zones,
            sr_by_tf.get(tf, {}),
            fvgs_by_tf.get(tf, []),
            obs_by_tf.get(tf, []),
            atrs.get(tf, 0.0),
            start,
        )
        tf_results[tf] = {
            "summary":  summary,
            "zones":    zones[:5],
            "setups":   setups[:3],
            "sigma":    sigma,
        }

    # ── Synthesise cross-TF directional bias ─────────────────────────────────
    weighted_up = 0.0
    total_w     = 0.0
    for tf, w in TF_BIAS_WEIGHT.items():
        res = tf_results.get(tf, {})
        sum_ = res.get("summary", {})
        # Use shortest horizon p_up for each TF
        if sum_:
            h0   = min(TF_HORIZONS.get(tf, [1]))
            p_up = sum_.get(h0, {}).get("p_up", 50.0)
            weighted_up += p_up * w
            total_w     += w
    composite_up = weighted_up / total_w if total_w > 0 else 50.0

    if   composite_up >= BIAS_THRESHOLD:         bias = "BULLISH"
    elif composite_up <= (100 - BIAS_THRESHOLD): bias = "BEARISH"
    else:                                         bias = "NO CLEAR BIAS"

    # ── Deduplicate probable setups across TFs ────────────────────────────────
    # Same zone appearing in multiple TFs = stronger signal
    all_setups   = []
    zone_counts  = {}   # zone_key → count of TFs agreeing
    for tf in ["4h", "1h", "15min"]:
        atr = atrs.get(tf, 1.0) or 1.0
        for s in tf_results.get(tf, {}).get("setups", []):
            key = (s["direction"], round(s["zone_price"] / (atr * 0.5)))
            zone_counts[key] = zone_counts.get(key, 0) + 1
            s["tf"]          = tf
            s["zone_key"]    = key
            all_setups.append(s)

    # Boost n_levels for setups confirmed by multiple TFs
    seen_keys = set()
    merged    = []
    for s in sorted(all_setups, key=lambda x: (-zone_counts[x["zone_key"]], -x["n_levels"], -x["density_pct"])):
        if s["zone_key"] in seen_keys:
            continue
        seen_keys.add(s["zone_key"])
        s["tf_count"] = zone_counts[s["zone_key"]]
        merged.append(s)

    return {
        "start_price":     start,
        "bias":            bias,
        "p_up":            round(composite_up, 1),
        "p_dn":            round(100 - composite_up, 1),
        "tf_results":      tf_results,    # per-TF detail
        "probable_setups": merged[:4],    # top 4 cross-TF setups
        "n_sims":          n_sims,
    }


async def run_prospective_scan(
    current_price: float,
    candles_by_tf: dict,      # {"1h": candles_dict, "4h": ..., "15min": ...}
    sr_by_tf:      dict = None,
    fvgs_by_tf:    dict = None,
    obs_by_tf:     dict = None,
    atrs:          dict = None,
) -> dict:
    """
    Multi-timeframe prospective async entry point.

    Args:
        current_price  : live WebSocket price
        candles_by_tf  : {"4h": {"closes":[], "highs":[], ...}, "1h": ..., "15min": ...}
        sr_by_tf       : {"4h": sr_dict, "1h": sr_dict, "15min": sr_dict}
        fvgs_by_tf     : {"4h": [...], "1h": [...], "15min": [...]}
        obs_by_tf      : {"4h": [...], "1h": [...], "15min": [...]}
        atrs           : {"4h": float, "1h": float, "15min": float}

    Returns prospective scan dict. Safe fallback on any error.
    """
    try:
        # Fit GARCH per TF independently — each TF has its own vol regime
        sigmas    = {}
        vol_model = "REALIZED"
        for tf, candles in (candles_by_tf or {}).items():
            if candles and candles.get("closes"):
                sig, ok = fit_garch(candles["closes"])
                sigmas[tf] = sig
                if ok:
                    vol_model = "GARCH"

        if not sigmas:
            return _prospect_fallback()

        loop   = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            _executor,
            _run_mtf_prospective,
            current_price,
            sigmas,
            N_SIMS,
            sr_by_tf   or {},
            fvgs_by_tf or {},
            obs_by_tf  or {},
            atrs       or {},
        )

        result["vol_model"] = vol_model
        # Use 1H sigma for annualised display (most representative for day trading)
        sigma_1h = sigmas.get("1h", sigmas.get("4h", list(sigmas.values())[0]))
        result["sigma_daily"] = round(sigma_1h * math.sqrt(TRADING_YEAR) * 100, 2)

        log.debug(
            f"MTF Prospective MC — bias={result['bias']} "
            f"p_up={result['p_up']}% "
            f"setups={len(result['probable_setups'])} ({vol_model})"
        )
        return result

    except Exception as e:
        log.error(f"Prospective scan failed: {e}")
        return _prospect_fallback()


def _prospect_fallback() -> dict:
    return {
        "start_price": 0.0, "bias": "UNKNOWN",
        "p_up": 50.0, "p_dn": 50.0,
        "tf_results": {}, "probable_setups": [],
        "n_sims": 0, "vol_model": "FAILED", "sigma_daily": 0.0,
    }
