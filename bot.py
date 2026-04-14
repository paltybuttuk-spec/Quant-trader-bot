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

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
log = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID   = int(os.environ["TELEGRAM_CHAT_ID"])
TWELVEDATA_API_KEY = os.environ["TWELVEDATA_API_KEY"]
FINNHUB_API_KEY    = os.environ["FINNHUB_API_KEY"]

TIMEFRAMES = ["15min", "1h", "4h", "1day", "1week"]
TF_LABELS  = {"15min": "15M", "1h": "1H", "4h": "4H", "1day": "Daily", "1week": "Weekly"}

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

prev_verdicts   = {}
weekly_history  = defaultdict(list)
score_history   = defaultdict(list)
last_setup_fire = {}
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
    return {"score": round(score,1), "verdict": score_to_verdict(score),
            "regime": regime, "z_score": round(z,2), "momentum": round(mom*100,3),
            "vol_ratio": round(vol_ratio,2), "vol_label": vol_label,
            "vwap_signal": vs, "efficiency": round(eff,3), "vol_conf": vc}

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

def calc_setup(closes, highs, lows, composite, vol_ratio):
    atr   = calc_atr(highs, lows, closes)
    price = closes[-1]
    sm    = 2.2 if vol_ratio > 1.5 else (1.1 if vol_ratio < 0.7 else 1.5)
    d     = "LONG" if composite > 58 else "SHORT"
    el, eh = round(price-0.3*atr,5), round(price+0.3*atr,5)
    if d == "LONG":
        stop = round(price-sm*atr,5)
        t1,t2,t3 = round(price+2*atr,5),round(price+3*atr,5),round(price+4*atr,5)
    else:
        stop = round(price+sm*atr,5)
        t1,t2,t3 = round(price-2*atr,5),round(price-3*atr,5),round(price-4*atr,5)
    return {"direction":d,"entry_low":el,"entry_high":eh,"stop":stop,
            "t1":t1,"t2":t2,"t3":t3,"atr":round(atr,5),"stop_mult":sm,
            "support":round(min(lows[-20:]),5),"resistance":round(max(highs[-20:]),5)}

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
    if not tf_scores: return
    composite, conf = calc_composite(tf_scores)
    persistent  = update_score_history(instrument, composite)
    avg_vr      = mean(vol_ratios) if vol_ratios else 1.0
    setup = None
    if daily_data and (composite > 58 or composite < 42) and conf > 70:
        setup = calc_setup(*daily_data, composite, avg_vr)
    result = {
        "instrument": instrument, "candle_sym": cfg["candle_sym"],
        "price": live_price, "composite": composite, "confidence": conf,
        "verdict": score_to_verdict(composite), "tf_scores": tf_scores,
        "tf_details": tf_details, "setup": setup, "persistent": persistent,
        "corr_price": corr_live, "corr_label": cfg["corr_label"],
        "dxy_momentum": round(dxy_mom*100,3), "vixy_pressure": round(vixy_pres*100,3),
        "live": True, "scored_at": datetime.utcnow().isoformat(),
    }
    live_scores[instrument] = result
    await check_flips_and_alerts(result)

async def check_flips_and_alerts(data):
    if app_ref is None: return
    inst = data["instrument"]
    for tf, detail in data["tf_details"].items():
        new_v = detail["verdict"]
        old_v = prev_verdicts.get(inst,{}).get(tf)
        if old_v is None:
            prev_verdicts.setdefault(inst,{})[tf] = new_v; continue
        if new_v != old_v:
            try:
                await app_ref.bot.send_message(TELEGRAM_CHAT_ID,
                    fmt_flip_alert(inst,tf,old_v,new_v,detail), parse_mode="Markdown")
            except Exception as e:
                log.error(f"Flip alert failed: {e}")
        prev_verdicts.setdefault(inst,{})[tf] = new_v
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
    if include_setup and data.get("setup"):
        s  = data["setup"]
        de = "🟢 LONG" if s["direction"]=="LONG" else "🔴 SHORT"
        lines += ["",f"*🎯 Trade Setup — {de}*",
            f"  Entry: `{s['entry_low']} – {s['entry_high']}`",
            f"  Stop: `{s['stop']}` (×{s['stop_mult']} ATR adaptive)",
            f"  T1: `{s['t1']}` | T2: `{s['t2']}` | T3: `{s['t3']}`",
            f"  ATR: `{s['atr']}` | S: `{s['support']}` | R: `{s['resistance']}`"]
    return "\n".join(l for l in lines if l is not None)

def fmt_flip_alert(inst, tf, old_v, new_v, d):
    e = IE.get(inst,"📊")
    return (f"⚡ *SCORE FLIP — {e} {inst}*\nTimeframe: *{TF_LABELS.get(tf,tf)}*\n"
            f"{VE.get(old_v,'⚪')} {old_v} → {VE.get(new_v,'⚪')} {new_v}\n"
            f"Score: `{d['score']}` | Z: `{d['z_score']:+.2f}` | Mom: `{d['momentum']:+.3f}%`\n"
            f"Regime: {d['regime']} | Vol: {d['vol_label']}")

def fmt_setup_alert(inst, data):
    s = data["setup"]; e = IE.get(inst,"📊")
    return (f"🚨 *SETUP ALERT — {e} {inst}*\n"
            f"{'🟢 LONG' if s['direction']=='LONG' else '🔴 SHORT'} | "
            f"Score: `{data['composite']}` | Conf: `{data['confidence']:.1f}%`\n"
            f"Entry: `{s['entry_low']} – {s['entry_high']}`\n"
            f"Stop: `{s['stop']}` (adaptive ×{s['stop_mult']})\n"
            f"T1: `{s['t1']}` | T2: `{s['t2']}` | T3: `{s['t3']}`\n"
            f"🔒 Persistent: {data['persistent']}")

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
    cache  = CandleCache(TWELVEDATA_API_KEY)
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
