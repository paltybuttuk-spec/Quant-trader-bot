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
# PROSPECTIVE SCAN — where will price most likely go from here?
# ══════════════════════════════════════════════════════════════════════════════

# Horizons to simulate forward (in bars — daily candles by default)
PROSPECT_HORIZONS = [5, 10, 20]

# Density bucket width as fraction of sigma per step
# Tighter = more precise zones, but needs more sims to fill
BUCKET_FRACTION   = 3.0

# Minimum path density to flag a zone as a hot zone (% of total paths)
HOT_ZONE_MIN_PCT  = 8.0

# Minimum probability for a probable setup flag (% of paths reaching level)
SETUP_MIN_PCT     = 15.0

# How close a destination zone must be to an S/R level to count as confluence
# Expressed as fraction of ATR
CONFLUENCE_ATR    = 0.4


def _prospect_paths(start: float, sigma: float, n_sims: int,
                    max_horizon: int) -> list:
    """
    Run n_sims GBM paths from start price, recording the price at each horizon.
    Returns list of lists: end_prices[horizon_idx][sim_idx]
    Zero drift — pure volatility fan.
    """
    # Pre-allocate: list of n_sims final prices per horizon
    horizon_set  = set(PROSPECT_HORIZONS)
    max_h        = max(PROSPECT_HORIZONS)
    results      = {h: [] for h in PROSPECT_HORIZONS}

    for _ in range(n_sims):
        price = start
        for step in range(1, max_h + 1):
            price *= math.exp(sigma * random.gauss(0, 1))
            if step in horizon_set:
                results[step].append(price)

    return results


def _build_density_map(prices: list, start: float, sigma: float) -> list:
    """
    Bucket end prices into zones and return sorted hot zones.
    Bucket width = BUCKET_FRACTION × sigma × start (one GBM step worth of move).
    Returns list of dicts: {price_mid, density_pct, direction}
    sorted by density descending.
    """
    if not prices:
        return []

    bucket_w = BUCKET_FRACTION * sigma * start
    if bucket_w <= 0:
        return []

    counts = {}
    for p in prices:
        b = round(p / bucket_w) * bucket_w
        counts[b] = counts.get(b, 0) + 1

    total  = len(prices)
    zones  = []
    for b, cnt in counts.items():
        pct = cnt / total * 100
        if pct >= HOT_ZONE_MIN_PCT:
            zones.append({
                "price_mid":   round(b, 5),
                "density_pct": round(pct, 1),
                "direction":   "UP" if b > start else "DOWN",
            })

    zones.sort(key=lambda z: z["density_pct"], reverse=True)
    return zones


def _match_levels(zones: list, sr_levels: dict, fvgs: list,
                  obs: list, atr: float, start: float) -> list:
    """
    Cross-reference destination hot zones with known S/R, FVG, OB levels.
    Returns probable setups — zones where price is likely to go AND
    a significant market structure level exists there.

    Each probable setup:
      direction      LONG | SHORT (relative to current price)
      level          the structural level price is heading toward
      level_type     SR_SUPPORT | SR_RESISTANCE | FVG | ORDER_BLOCK
      strength       STRONG | MODERATE | WEAK (from S/R) or impulse× (OB)
      density_pct    % of paths landing in this zone at this horizon
      horizon        bars forward
      confluence     list of tags (multiple levels at same zone)
    """
    if atr <= 0:
        return []

    radius    = CONFLUENCE_ATR * atr
    setups    = []
    seen      = set()   # deduplicate by (level_type, rounded_level)

    all_levels = []

    # S/R support
    for z in sr_levels.get("support", []):
        all_levels.append(("SR_SUPPORT",    z["price"], z["strength"], z))
    # S/R resistance
    for z in sr_levels.get("resistance", []):
        all_levels.append(("SR_RESISTANCE", z["price"], z["strength"], z))
    # Unfilled FVGs
    for f in fvgs:
        if not f.get("filled"):
            all_levels.append(("FVG", f["midpoint"], f["type"], f))
    # Unmitigated OBs
    for ob in obs:
        if not ob.get("mitigated"):
            all_levels.append(("ORDER_BLOCK", ob["midpoint"],
                                f"×{ob['impulse']} impulse", ob))

    for zone in zones:
        zone_price = zone["price_mid"]
        direction  = "LONG" if zone_price > start else "SHORT"
        matched    = []

        for ltype, lprice, lstrength, ldata in all_levels:
            if abs(zone_price - lprice) <= radius:
                matched.append({
                    "type":     ltype,
                    "price":    lprice,
                    "strength": lstrength,
                })

        if not matched:
            continue

        # Deduplicate — skip if we already have a setup at this level
        key = (direction, round(zone_price / (atr * 0.5)))
        if key in seen:
            continue
        seen.add(key)

        # Confluence quality score: more matches = higher quality
        conf_tags = []
        for m in matched:
            if m["type"] == "SR_SUPPORT":
                conf_tags.append(f"🟢 S/R Support ({m['strength']}) @ {m['price']:.5f}")
            elif m["type"] == "SR_RESISTANCE":
                conf_tags.append(f"🔴 S/R Resistance ({m['strength']}) @ {m['price']:.5f}")
            elif m["type"] == "FVG":
                conf_tags.append(f"⬜ FVG ({m['strength']}) midpoint @ {m['price']:.5f}")
            elif m["type"] == "ORDER_BLOCK":
                conf_tags.append(f"🟦 OB ({m['strength']}) midpoint @ {m['price']:.5f}")

        setups.append({
            "direction":   direction,
            "zone_price":  zone_price,
            "density_pct": zone["density_pct"],
            "confluence":  conf_tags,
            "n_levels":    len(matched),
            "primary_type": matched[0]["type"],
            "primary_level": matched[0]["price"],
        })

    # Sort: most confluence levels first, then density
    setups.sort(key=lambda s: (-s["n_levels"], -s["density_pct"]))
    return setups


def _prospect_summary(results: dict, start: float) -> dict:
    """
    Compute directional bias and median destinations per horizon.
    results: {horizon: [end_prices]}
    """
    out = {}
    for h, prices in results.items():
        if not prices:
            continue
        n        = len(prices)
        above    = sum(1 for p in prices if p > start)
        p_up     = round(above / n * 100, 1)
        p_dn     = round(100 - p_up, 1)
        sorted_p = sorted(prices)
        median   = sorted_p[n // 2]
        p5       = sorted_p[max(0, int(n * 0.05))]
        p95      = sorted_p[min(n-1, int(n * 0.95))]
        out[h]   = {
            "p_up":   p_up,
            "p_dn":   p_dn,
            "median": round(median, 5),
            "p5":     round(p5, 5),
            "p95":    round(p95, 5),
        }
    return out


def _run_prospective(start: float, sigma: float, n_sims: int,
                     sr_levels: dict, fvgs: list, obs: list, atr: float) -> dict:
    """
    Core prospective simulation — runs in thread executor.
    """
    # 1. Simulate all paths
    results = _prospect_paths(start, sigma, n_sims, max(PROSPECT_HORIZONS))

    # 2. Summary stats per horizon
    summary = _prospect_summary(results, start)

    # 3. Overall directional bias (use longest horizon)
    longest_h = max(PROSPECT_HORIZONS)
    bias_data = summary.get(longest_h, {})
    p_up      = bias_data.get("p_up", 50.0)

    if   p_up >= 60: bias_label = "BULLISH"
    elif p_up <= 40: bias_label = "BEARISH"
    else:            bias_label = "NEUTRAL"

    # 4. Build density map on longest horizon prices
    longest_prices = results.get(longest_h, [])
    hot_zones      = _build_density_map(longest_prices, start, sigma)

    # 5. Match zones to structural levels → probable setups
    probable_setups = _match_levels(hot_zones, sr_levels, fvgs, obs, atr, start)

    return {
        "start_price":     round(start, 5),
        "bias":            bias_label,
        "p_up":            p_up,
        "p_dn":            round(100 - p_up, 1),
        "horizons":        summary,           # {5: {p_up, median, p5, p95}, ...}
        "hot_zones":       hot_zones[:6],     # top 6 density zones
        "probable_setups": probable_setups[:4],  # top 4 by confluence
        "n_sims":          n_sims,
    }


async def run_prospective_scan(current_price: float, closes: list,
                               sr_levels: dict = None,
                               fvgs: list = None,
                               obs: list = None,
                               atr: float = 0.0) -> dict:
    """
    Async entry point for the prospective simulation.

    Runs 10,000 paths from current_price, finds where price is most
    likely to go, and cross-references those destinations with S/R,
    FVG, and OB levels to flag probable future setups.

    Args:
        current_price : live price (WebSocket tick)
        closes        : historical closes for vol estimation
        sr_levels     : output of calc_sr_levels() — can be None
        fvgs          : output of find_fvg() — can be None
        obs           : output of find_order_blocks() — can be None
        atr           : current ATR value — can be 0.0

    Returns:
        Prospective scan dict. On failure returns safe empty fallback.
    """
    try:
        sigma, garch_ok = fit_garch(closes)
        vol_model       = "GARCH" if garch_ok else "REALIZED"

        loop   = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            _executor,
            _run_prospective,
            current_price,
            sigma,
            N_SIMS,
            sr_levels or {},
            fvgs      or [],
            obs       or [],
            atr,
        )

        result["vol_model"]   = vol_model
        result["sigma_daily"] = round(sigma * math.sqrt(TRADING_YEAR) * 100, 2)
        log.debug(
            f"Prospective MC — bias={result['bias']} "
            f"p_up={result['p_up']}% "
            f"probable_setups={len(result['probable_setups'])} "
            f"({vol_model})"
        )
        return result

    except Exception as e:
        log.error(f"Prospective scan failed: {e}")
        return _prospect_fallback()


def _prospect_fallback() -> dict:
    return {
        "start_price": 0.0, "bias": "UNKNOWN",
        "p_up": 50.0, "p_dn": 50.0,
        "horizons": {}, "hot_zones": [],
        "probable_setups": [], "n_sims": 0,
        "vol_model": "FAILED", "sigma_daily": 0.0,
    }
