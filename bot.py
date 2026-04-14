"""
QuantRisk Bot v5 — WebSocket Live Architecture
- WebSocket: live price ticks every few seconds, zero REST budget
- Candle cache: OHLCV history refreshed on staggered schedule
- Scoring engine: re-runs on every tick using cached candles + live price
- Flip detection: near-instant (seconds not 30 minutes)
- Setup alerts: fire the moment conditions are met
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
log = logging.getLogger(__name__)

# ── ENV ───────────────────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID   = int(os.environ["TELEGRAM_CHAT_ID"])
TWELVEDATA_API_KEY = os.environ["TWELVEDATA_API_KEY"]
FINNHUB_API_KEY    = os.environ["FINNHUB_API_KEY"]

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
TIMEFRAMES = ["15min", "1h", "4h", "1day", "1week"]
TF_LABELS  = {"15min": "15M", "1h": "1H", "4h": "4H", "1day": "Daily", "1week": "Weekly"}

INSTRUMENTS = {
    "Gold": {"symbol": "GLD", "corr": "UUP"},
    "US30": {"symbol": "DIA", "corr": "VIXY"},
}
ALL_SYMBOLS     = ["GLD", "DIA", "UUP", "VIXY"]
SYMBOL_TO_INST  = {"GLD": "Gold", "DIA": "US30"}

# ── GLOBAL STATE ──────────────────────────────────────────────────────────────
prev_verdicts   = {}
weekly_history  = defaultdict(list)
score_history   = defaultdict(list)
last_setup_fire = {}
live_scores     = {}
_last_score_ts  = {}   # tick throttle
cache: CandleCache = None
ws_mgr: WebSocketManager = None
app_ref = None


# ═══════════════════════════════════════════════════════════════════════════════
# SESSION
# ═══════════════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════════════
# SCORING ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def calc_atr(highs, lows, closes, period=14):
    trs = [max(highs[i]-lows[i], abs(highs[i]-closes[i-1]), abs(lows[i]-closes[i-1]))
           for i in range(1, len(closes))]
    return mean(trs[-period:]) if len(trs) >= period else (mean(trs) if trs else 0)

def calc_vwap(closes, volumes, period=20):
    c, v = closes[-period:], volumes[-period:]
    tv = sum(v)
    return sum(p*vv for p, vv in zip(c, v)) / tv if tv else closes[-1]

def score_to_verdict(s):
    if s >= 70: return "STRONG BUY"
    if s >= 58: return "BUY"
    if s >= 45: return "NEUTRAL"
    if s >= 32: return "SELL"
    return "STRONG SELL"

def score_timeframe(closes, highs, lows, opens, volumes,
                    uup_momentum=0.0, vixy_pressure=0.0, instrument="Gold"):
    if len(closes) < 30:
        return {"score": 50, "verdict": "NEUTRAL", "regime": "MIXED",
                "z_score": 0, "momentum": 0, "vol_ratio": 1,
                "vol_label": "NORMAL", "vwap_signal": "NEUTRAL",
                "efficiency": 0, "vol_conf": 1.0}

    score = 50.0

    # Regime
    net   = abs(closes[-1] - closes[-11])
    path  = sum(abs(closes[i] - closes[i-1]) for i in range(-10, 0))
    eff   = net / path if path > 0 else 0
    regime = "TRENDING" if eff > 0.55 else ("RANGING" if eff < 0.35 else "MIXED")

    # Volatility
    r10 = [abs(closes[i]-closes[i-1])/closes[i-1] for i in range(-10, 0) if closes[i-1] > 0]
    r30 = [abs(closes[i]-closes[i-1])/closes[i-1] for i in range(-30, 0) if closes[i-1] > 0]
    v10 = mean(r10) if r10 else 0.01
    v30 = mean(r30) if r30 else 0.01
    vol_ratio = v10 / v30 if v30 > 0 else 1.0
    if   vol_ratio < 0.70: vol_mult, vol_label = 1.2, "LOW"
    elif vol_ratio < 1.50: vol_mult, vol_label = 1.0, "NORMAL"
    elif vol_ratio < 2.00: vol_mult, vol_label = 0.7, "HIGH"
    else:                  vol_mult, vol_label = 0.3, "EXTREME"

    # IMPROVEMENT 1: Adaptive Z-score window
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

    # IMPROVEMENT 2: Volume-confirmed momentum
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

    # VWAP
    vwap = calc_vwap(closes, volumes)
    if closes[-1] > vwap: score += 8;  vs = "ABOVE"
    else:                  score -= 8;  vs = "BELOW"

    # IMPROVEMENT 3: Continuous inter-market
    if instrument == "Gold" and uup_momentum:
        score += max(-15, min(15, -uup_momentum * 150))
    elif instrument == "US30" and vixy_pressure:
        score += max(-12, min(12, -vixy_pressure * 100))

    # Multipliers
    sm, _ = get_session_multiplier()
    score  = 50 + (score - 50) * vol_mult
    score  = 50 + (score - 50) * sm
    score  = max(0, min(100, score))

    return {
        "score":      round(score, 1),
        "verdict":    score_to_verdict(score),
        "regime":     regime,
        "z_score":    round(z, 2),
        "momentum":   round(mom * 100, 3),
        "vol_ratio":  round(vol_ratio, 2),
        "vol_label":  vol_label,
        "vwap_signal": vs,
        "efficiency": round(eff, 3),
        "vol_conf":   vc,
    }

def calc_composite(tf_scores: dict):
    if not tf_scores:
        return 50.0, 0.0
    scores    = list(tf_scores.values())
    composite = mean(scores)
    bull = sum(1 for s in scores if s > 58)
    bear = sum(1 for s in scores if s < 42)
    # IMPROVEMENT 6: Alignment bonus
    if bull >= 4:   composite = min(100, composite + 8)
    elif bear >= 4: composite = max(0,   composite - 8)
    aligned    = max(bull, bear)
    confidence = (composite / 100) * (aligned / len(scores)) * 100
    return round(composite, 1), round(confidence, 1)

def update_score_history(inst: str, score: float) -> bool:
    h = score_history[inst]
    h.append(score)
    score_history[inst] = h[-3:]
    if len(h) < 3:
        return False
    return all(s > 58 for s in h) or all(s < 42 for s in h)

def calc_setup(closes, highs, lows, composite, vol_ratio):
    atr   = calc_atr(highs, lows, closes)
    price = closes[-1]
    # IMPROVEMENT 5: Adaptive ATR stop
    sm    = 2.2 if vol_ratio > 1.5 else (1.1 if vol_ratio < 0.7 else 1.5)
    d     = "LONG" if composite > 58 else "SHORT"
    el, eh = round(price - 0.3*atr, 4), round(price + 0.3*atr, 4)
    if d == "LONG":
        stop = round(price - sm*atr, 4)
        t1, t2, t3 = round(price+2*atr,4), round(price+3*atr,4), round(price+4*atr,4)
    else:
        stop = round(price + sm*atr, 4)
        t1, t2, t3 = round(price-2*atr,4), round(price-3*atr,4), round(price-4*atr,4)
    return {
        "direction": d, "entry_low": el, "entry_high": eh,
        "stop": stop, "t1": t1, "t2": t2, "t3": t3,
        "atr": round(atr, 4), "stop_mult": sm,
        "support": round(min(lows[-20:]), 4),
        "resistance": round(max(highs[-20:]), 4),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# LIVE SCORING — fires on every WebSocket tick
# ═══════════════════════════════════════════════════════════════════════════════

async def score_instrument_live(instrument: str, live_price: float):
    cfg      = INSTRUMENTS[instrument]
    corr_sym = cfg["corr"]

    uup_mom = vixy_pres = 0.0
    cd = cache.get(corr_sym, "1day")
    if cd and len(cd["closes"]) >= 6:
        cc = cd["closes"]
        m  = (cc[-1] - cc[-6]) / cc[-6] if cc[-6] > 0 else 0
        if instrument == "Gold":   uup_mom   = m
        elif instrument == "US30": vixy_pres = m

    tf_scores  = {}
    tf_details = {}
    daily_data = None
    vol_ratios = []

    for tf in TIMEFRAMES:
        candles = cache.get_with_live_price(cfg["symbol"], tf, live_price)
        if not candles:
            continue
        d = score_timeframe(
            candles["closes"], candles["highs"], candles["lows"],
            candles["opens"], candles["volumes"],
            uup_momentum=uup_mom, vixy_pressure=vixy_pres, instrument=instrument,
        )
        tf_scores[tf]  = d["score"]
        tf_details[tf] = d
        vol_ratios.append(d["vol_ratio"])
        if tf == "1day":
            daily_data = (candles["closes"], candles["highs"], candles["lows"])

    if not tf_scores:
        return

    composite, conf = calc_composite(tf_scores)
    persistent = update_score_history(instrument, composite)
    avg_vr = mean(vol_ratios) if vol_ratios else 1.0

    setup = None
    if daily_data and (composite > 58 or composite < 42) and conf > 70:
        setup = calc_setup(*daily_data, composite, avg_vr)

    corr_d     = cache.get(corr_sym, "1day")
    corr_last  = corr_d["closes"][-1] if corr_d else None

    result = {
        "instrument":    instrument,
        "symbol":        cfg["symbol"],
        "price":         live_price,
        "composite":     composite,
        "confidence":    conf,
        "verdict":       score_to_verdict(composite),
        "tf_scores":     tf_scores,
        "tf_details":    tf_details,
        "setup":         setup,
        "persistent":    persistent,
        "corr_price":    corr_last,
        "corr_symbol":   corr_sym,
        "uup_momentum":  round(uup_mom * 100, 3),
        "vixy_pressure": round(vixy_pres * 100, 3),
        "live":          True,
        "scored_at":     datetime.utcnow().isoformat(),
    }
    live_scores[instrument] = result
    await check_flips_and_alerts(result)


async def check_flips_and_alerts(data: dict):
    if app_ref is None:
        return
    inst = data["instrument"]

    for tf, detail in data["tf_details"].items():
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

    if is_active_session() and data["setup"] and data["confidence"] > 70 and data["persistent"]:
        last = last_setup_fire.get(inst)
        now  = datetime.utcnow()
        if last is None or (now - last).seconds > 5400:
            try:
                await app_ref.bot.send_message(
                    TELEGRAM_CHAT_ID,
                    fmt_setup_alert(inst, data),
                    parse_mode="Markdown",
                )
                last_setup_fire[inst] = now
            except Exception as e:
                log.error(f"Setup alert failed: {e}")

    weekly_history[inst].append((datetime.utcnow(), data["price"], data["composite"], data["verdict"]))
    weekly_history[inst] = weekly_history[inst][-336:]


async def on_price_tick(symbol: str, price: float):
    inst = SYMBOL_TO_INST.get(symbol)
    if not inst:
        return
    now = _time.time()
    if now - _last_score_ts.get(inst, 0) < 3:
        return
    _last_score_ts[inst] = now
    if not cache.is_populated(symbol):
        return
    try:
        await score_instrument_live(inst, price)
    except Exception as e:
        log.error(f"Live score error {inst}: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# NEWS
# ═══════════════════════════════════════════════════════════════════════════════

async def fetch_news(client, symbol: str) -> list:
    now  = datetime.utcnow()
    from_d = now.replace(day=max(1, now.day - 7)).strftime("%Y-%m-%d")
    url  = "https://finnhub.io/api/v1/company-news"
    try:
        r = await client.get(url, params={
            "symbol": symbol, "from": from_d,
            "to": now.strftime("%Y-%m-%d"), "token": FINNHUB_API_KEY,
        }, timeout=10)
        articles = r.json()
        if not isinstance(articles, list):
            return []
        scored = []
        for a in articles:
            s = a.get("sentiment", {})
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
    emoji = "🥇" if instrument == "Gold" else "🏦"
    lines = [f"{emoji} *{instrument} News*\n", "📈 *Catalysts:*"]
    for _, a in news[:5]:
        lines.append(f"  • {a.get('headline','')[:80]}")
    lines.append("\n📉 *Risks:*")
    for _, a in news[-5:]:
        lines.append(f"  • {a.get('headline','')[:80]}")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# FORMATTERS
# ═══════════════════════════════════════════════════════════════════════════════

VE = {"STRONG BUY":"🟢🟢","BUY":"🟢","NEUTRAL":"🟡","SELL":"🔴","STRONG SELL":"🔴🔴"}
RE = {"TRENDING":"📈","RANGING":"↔️","MIXED":"〰️"}

def fmt_instrument(data: dict, include_setup=True) -> str:
    inst  = data["instrument"]
    sm, sl = get_session_multiplier()
    lines = [
        f"{'🥇' if inst=='Gold' else '🏦'} *{inst}* 🔴 LIVE",
        f"💲 Price: `{data['price']:.4f}`" if data.get("price") else "",
        f"📊 Score: `{data['composite']}/100` — {VE.get(data['verdict'],'⚪')} *{data['verdict']}*",
        f"🎯 Confidence: `{data['confidence']:.1f}%`",
        f"⏰ Session: {sl} (×{sm})",
        "", "*Timeframe Breakdown:*",
    ]
    for tf in TIMEFRAMES:
        d = data["tf_details"].get(tf)
        if not d:
            lines.append(f"  {TF_LABELS[tf]}: —"); continue
        vc = "✅" if d["vol_conf"] == 1.0 else "⚠️"
        lines.append(f"  {TF_LABELS[tf]}: `{d['score']}` {VE.get(d['verdict'],'⚪')} {RE.get(d['regime'],'')} {vc}")
    d15 = data["tf_details"].get("15min") or next(iter(data["tf_details"].values()), {})
    if d15:
        lines += ["", "*Quant Signals:*",
            f"  Z-Score: `{d15.get('z_score',0):+.2f}` (adaptive window)",
            f"  Momentum: `{d15.get('momentum',0):+.3f}%` {'✅' if d15.get('vol_conf',1)==1 else '⚠️ low vol'}",
            f"  Regime: {d15.get('regime','—')} {RE.get(d15.get('regime',''),'')}" ,
            f"  Vol: `{d15.get('vol_ratio',1):.2f}` → {d15.get('vol_label','NORMAL')}",
            f"  VWAP: {'Above ✅' if d15.get('vwap_signal')=='ABOVE' else 'Below ⚠️'}"]
    cp = data.get("corr_price")
    if cp:
        if inst == "Gold":
            m = data["uup_momentum"]
            t = "Headwind 🔴" if m > 0.3 else ("Tailwind 🟢" if m < -0.3 else "Neutral 🟡")
            lines.append(f"\n💱 DXY (UUP): `{cp:.2f}` | Mom: `{m:+.2f}%` → {t}")
        else:
            p = data["vixy_pressure"]
            t = "Risk-Off 🔴" if p > 0.3 else ("Risk-On 🟢" if p < -0.3 else "Calm 🟡")
            lines.append(f"\n😱 VIX (VIXY): `{cp:.2f}` | Mom: `{p:+.2f}%` → {t}")
    if data.get("persistent"):
        lines.append("\n🔒 *Signal Persistent — 3 checks confirmed*")
    if include_setup and data.get("setup"):
        s = data["setup"]
        de = "🟢 LONG" if s["direction"]=="LONG" else "🔴 SHORT"
        lines += ["", f"*🎯 Trade Setup — {de}*",
            f"  Entry: `{s['entry_low']} – {s['entry_high']}`",
            f"  Stop: `{s['stop']}` (×{s['stop_mult']} ATR adaptive)",
            f"  T1: `{s['t1']}` | T2: `{s['t2']}` | T3: `{s['t3']}`",
            f"  ATR: `{s['atr']}` | S: `{s['support']}` | R: `{s['resistance']}`"]
    return "\n".join(l for l in lines if l is not None)

def fmt_flip_alert(inst, tf, old_v, new_v, d):
    return (f"⚡ *SCORE FLIP — {inst}*\n"
            f"Timeframe: *{TF_LABELS.get(tf,tf)}*\n"
            f"{VE.get(old_v,'⚪')} {old_v} → {VE.get(new_v,'⚪')} {new_v}\n"
            f"Score: `{d['score']}` | Z: `{d['z_score']:+.2f}` | Mom: `{d['momentum']:+.3f}%`\n"
            f"Regime: {d['regime']} | Vol: {d['vol_label']}")

def fmt_setup_alert(inst, data):
    s = data["setup"]
    return (f"🚨 *SETUP ALERT — {inst}*\n"
            f"{'🟢 LONG' if s['direction']=='LONG' else '🔴 SHORT'} | "
            f"Score: `{data['composite']}` | Conf: `{data['confidence']:.1f}%`\n"
            f"Entry: `{s['entry_low']} – {s['entry_high']}`\n"
            f"Stop: `{s['stop']}` (adaptive ×{s['stop_mult']})\n"
            f"T1: `{s['t1']}` | T2: `{s['t2']}` | T3: `{s['t3']}`\n"
            f"🔒 Persistent: {data['persistent']}")

def main_keyboard():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🥇 Gold",    callback_data="gold"),
         InlineKeyboardButton("🏦 US30",    callback_data="us30")],
        [InlineKeyboardButton("📊 Summary", callback_data="summary"),
         InlineKeyboardButton("🎯 Setups",  callback_data="setup")],
        [InlineKeyboardButton("📅 Weekly",  callback_data="weekly"),
         InlineKeyboardButton("🔄 Refresh", callback_data="summary")],
    ])


# ═══════════════════════════════════════════════════════════════════════════════
# SCHEDULED JOBS
# ═══════════════════════════════════════════════════════════════════════════════

async def job_daily_report(app):
    async with httpx.AsyncClient() as client:
        gn, dn = await asyncio.gather(fetch_news(client, "GLD"), fetch_news(client, "DIA"))
    hdr = (f"☀️ *QuantRisk Daily Report*\n"
           f"📅 {datetime.utcnow().strftime('%A, %d %B %Y')} | EAT 07:30\n{'─'*30}")
    await app.bot.send_message(TELEGRAM_CHAT_ID, hdr, parse_mode="Markdown")
    for inst, news in [("Gold", gn), ("US30", dn)]:
        if d := live_scores.get(inst):
            await app.bot.send_message(TELEGRAM_CHAT_ID, fmt_instrument(d), parse_mode="Markdown")
        if cats := _fmt_news(news, inst):
            await app.bot.send_message(TELEGRAM_CHAT_ID, cats, parse_mode="Markdown")
    await app.bot.send_message(TELEGRAM_CHAT_ID, "Good trading today 🚀", reply_markup=main_keyboard())

async def job_weekly_recap(app):
    lines = ["📅 *QuantRisk Weekly Recap*\n"]
    for inst, history in weekly_history.items():
        if not history:
            continue
        scores = [h[2] for h in history]
        from collections import Counter
        tv = Counter(h[3] for h in history).most_common(1)[0][0]
        e  = "🥇" if inst == "Gold" else "🏦"
        lines.append(f"{e} *{inst}*\n  Avg: `{round(mean(scores),1)}` | "
                     f"High: `{max(scores)}` | Low: `{min(scores)}`\n"
                     f"  Dominant: {VE.get(tv,'⚪')} {tv} | Points: {len(history)}\n")
    await app.bot.send_message(TELEGRAM_CHAT_ID, "\n".join(lines), parse_mode="Markdown")


# ═══════════════════════════════════════════════════════════════════════════════
# COMMANDS
# ═══════════════════════════════════════════════════════════════════════════════

def _mo(update):
    return update.message or update.callback_query.message

async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    wss = "🟢 WebSocket live" if (ws_mgr and ws_mgr.is_fresh("GLD")) else "🟡 Connecting..."
    cs  = "✅ Cache warm" if (cache and cache.all_populated(ALL_SYMBOLS)) else "⏳ Warming..."
    await _mo(update).reply_text(
        f"👋 *QuantRisk Bot v5 — WebSocket Live*\n\n{wss}\n{cs}\n\n"
        f"Scores update every few seconds via WebSocket.\n"
        f"Flip alerts fire in seconds, not 30 minutes.\n\n"
        f"*6 scoring improvements active ✅*",
        parse_mode="Markdown", reply_markup=main_keyboard(),
    )

async def _send_inst(update, inst):
    d = live_scores.get(inst)
    if not d:
        await _mo(update).reply_text(f"⏳ {inst} — WebSocket warming up, try again in 30s.")
        return
    await _mo(update).reply_text(fmt_instrument(d), parse_mode="Markdown", reply_markup=main_keyboard())

async def cmd_gold(u, c):    await _send_inst(u, "Gold")
async def cmd_us30(u, c):   await _send_inst(u, "US30")

async def cmd_summary(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    sm, sl = get_session_multiplier()
    lines  = [f"📊 *QuantRisk Summary*\n⏰ {sl} (×{sm})\n"]
    for inst in ["Gold", "US30"]:
        d = live_scores.get(inst)
        e = "🥇" if inst == "Gold" else "🏦"
        if not d:
            lines.append(f"{e} *{inst}*: ⏳ Loading..."); continue
        lines.append(f"{e} *{inst}*: `{d['composite']}/100` {VE.get(d['verdict'],'⚪')} {d['verdict']}\n"
                     f"  Conf: `{d['confidence']:.1f}%` | Persistent: {'✅' if d['persistent'] else '⏳'} 🔴 LIVE")
    await _mo(update).reply_text("\n".join(lines), parse_mode="Markdown", reply_markup=main_keyboard())

async def cmd_setup(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    found = False
    for inst in ["Gold", "US30"]:
        d = live_scores.get(inst)
        if d and d.get("setup"):
            await _mo(update).reply_text(fmt_setup_alert(inst, d), parse_mode="Markdown")
            found = True
    if not found:
        await _mo(update).reply_text(
            "🟡 No high-confidence setups right now.\nNeed score >58 or <42 + confidence >70%.",
            reply_markup=main_keyboard(),
        )

async def cmd_report(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    async with httpx.AsyncClient() as client:
        gn, dn = await asyncio.gather(fetch_news(client, "GLD"), fetch_news(client, "DIA"))
    for inst, news in [("Gold", gn), ("US30", dn)]:
        if d := live_scores.get(inst):
            await _mo(update).reply_text(fmt_instrument(d), parse_mode="Markdown")
        if cats := _fmt_news(news, inst):
            await _mo(update).reply_text(cats, parse_mode="Markdown")
    await _mo(update).reply_text("Report complete ✅", reply_markup=main_keyboard())

async def cmd_weekly(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not any(weekly_history.values()):
        await _mo(update).reply_text("📅 History building — check back later.", reply_markup=main_keyboard())
        return
    await job_weekly_recap(ctx.application)

async def button_handler(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.callback_query.answer()
    h = {"gold": cmd_gold, "us30": cmd_us30, "summary": cmd_summary,
         "setup": cmd_setup, "weekly": cmd_weekly}.get(update.callback_query.data)
    if h:
        await h(update, ctx)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    global cache, ws_mgr, app_ref

    app     = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app_ref = app

    app.add_handler(CommandHandler("start",   cmd_start))
    app.add_handler(CommandHandler("gold",    cmd_gold))
    app.add_handler(CommandHandler("us30",    cmd_us30))
    app.add_handler(CommandHandler("report",  cmd_report))
    app.add_handler(CommandHandler("setup",   cmd_setup))
    app.add_handler(CommandHandler("summary", cmd_summary))
    app.add_handler(CommandHandler("weekly",  cmd_weekly))
    app.add_handler(CallbackQueryHandler(button_handler))

    cache  = CandleCache(TWELVEDATA_API_KEY)
    ws_mgr = WebSocketManager(TWELVEDATA_API_KEY, on_tick_callback=on_price_tick)

    scheduler = AsyncIOScheduler(timezone="Africa/Nairobi")
    scheduler.add_job(lambda: asyncio.create_task(job_daily_report(app)),
                      "cron", hour=7, minute=30, id="daily")
    scheduler.add_job(lambda: asyncio.create_task(job_weekly_recap(app)),
                      "cron", day_of_week="mon", hour=7, minute=30, id="weekly")
    scheduler.start()

    async def post_init(application):
        log.info("Warming candle cache...")
        await cache.warm_up(ALL_SYMBOLS, TIMEFRAMES)
        log.info("Cache warm. Starting WebSocket...")
        asyncio.create_task(ws_mgr.start())
        asyncio.create_task(cache.refresh_loop(ALL_SYMBOLS, TIMEFRAMES))
        log.info("QuantRisk Bot v5 fully live. 🚀")

    app.post_init = post_init
    log.info("QuantRisk Bot v5 starting...")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
