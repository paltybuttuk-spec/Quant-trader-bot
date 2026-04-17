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
