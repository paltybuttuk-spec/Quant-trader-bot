import os
import asyncio
import logging
import httpx
import math
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes
from apscheduler.schedulers.asyncio import AsyncIOScheduler

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
log = logging.getLogger(__name__)

# ── ENV ───────────────────────────────────────────────────────────────────────
TOKEN      = os.environ["TELEGRAM_BOT_TOKEN"]
CHAT_ID    = int(os.environ["TELEGRAM_CHAT_ID"])
TD_KEY     = os.environ["TWELVEDATA_API_KEY"]
FH_KEY     = os.environ["FINNHUB_API_KEY"]
TZ         = "Africa/Nairobi"
HOUR       = 7
MINUTE     = 30

# ── Instruments ───────────────────────────────────────────────────────────────
INSTRUMENTS = {
    "GOLD": {
        "name":       "Gold (GLD)",
        "emoji":      "🥇",
        "class":      "Commodity",
        "symbol":     "GLD",
        "corr_sym":   "UUP",
        "corr_name":  "DXY",
        "spread":     "$0.35/oz",
        "news_sym":   "GLD",
        "cot":        "Non-commercial NET LONG | ETF inflows resuming | CB buying 27t/month avg (CFTC last known)",
    },
    "US30": {
        "name":       "US 30 (DIA)",
        "emoji":      "🏦",
        "class":      "Index",
        "symbol":     "DIA",
        "corr_sym":   "VIXY",
        "corr_name":  "VIX",
        "spread":     "1.5 pts",
        "news_sym":   "DIA",
        "cot":        "HF net long down 42% from 58% | CTA trend followers SHORT | Corp buybacks $150B (CFTC last known)",
    },
}

# ── Session schedule EAT ──────────────────────────────────────────────────────
def current_session() -> tuple:
    now = datetime.now(ZoneInfo(TZ))
    h   = now.hour
    if 8 <= h < 11:
        return "London Open", 1.2
    elif 11 <= h < 16:
        return "London/NY Overlap", 1.4
    elif 16 <= h < 23:
        return "New York", 1.3
    elif 0 <= h < 8:
        return "Asian", 0.7
    else:
        return "After Hours", 0.5

# ── State ─────────────────────────────────────────────────────────────────────
prev_verdicts  = {}
weekly_history = {}
signal_history = {}  # {key+tf: [scores]} for Sharpe calc

# ── Twelve Data ───────────────────────────────────────────────────────────────
FALLBACKS = {
    "GLD":  ["GLD"],
    "DIA":  ["DIA"],
    "UUP":  ["UUP"],
    "VIXY": ["VIXY"],
}

async def td_candles(symbol: str, interval: str, size: int = 50) -> list:
    url = "https://api.twelvedata.com/time_series"
    for sym in FALLBACKS.get(symbol, [symbol]):
        try:
            async with httpx.AsyncClient(timeout=20) as c:
                r = await c.get(url, params={
                    "symbol": sym, "interval": interval,
                    "outputsize": size, "apikey": TD_KEY, "format": "JSON"
                })
                d = r.json()
            if d.get("status") == "error":
                log.warning(f"td {sym} {interval}: {d.get('message')}")
                continue
            vals = d.get("values", [])
            if vals:
                return vals
        except Exception as e:
            log.error(f"td error {sym} {interval}: {e}")
    return []

async def td_price(symbol: str) -> float:
    for sym in FALLBACKS.get(symbol, [symbol]):
        try:
            async with httpx.AsyncClient(timeout=10) as c:
                r = await c.get("https://api.twelvedata.com/price",
                                params={"symbol": sym, "apikey": TD_KEY})
                d = r.json()
            if "price" in d:
                return float(d["price"])
        except Exception:
            pass
    return 0.0

# ── Finnhub news ──────────────────────────────────────────────────────────────
async def get_news(symbol: str) -> tuple:
    today    = datetime.now(ZoneInfo(TZ))
    week_ago = today - timedelta(days=7)
    url      = "https://finnhub.io/api/v1/company-news"
    try:
        async with httpx.AsyncClient(timeout=15) as c:
            r = await c.get(url, params={
                "symbol": symbol,
                "from":   week_ago.strftime("%Y-%m-%d"),
                "to":     today.strftime("%Y-%m-%d"),
                "token":  FH_KEY,
            })
            articles = r.json()
        if not isinstance(articles, list) or len(articles) == 0:
            return [], []
        scored = []
        for a in articles[:20]:
            headline  = a.get("headline", "")
            sentiment = a.get("sentiment", {})
            score     = sentiment.get("compound", 0) if isinstance(sentiment, dict) else 0
            if headline:
                scored.append((score, headline))
        scored.sort(key=lambda x: x[0], reverse=True)
        catalysts = [h for _, h in scored[:5] if _ > 0.05]
        risks     = [h for _, h in scored[-5:] if _ < -0.05]
        if not catalysts:
            catalysts = [h for _, h in scored[:3]]
        if not risks:
            risks = [h for _, h in scored[-3:]]
        return catalysts, risks
    except Exception as e:
        log.error(f"Finnhub news error {symbol}: {e}")
        return [], []

# ── Quant Engine ──────────────────────────────────────────────────────────────

def zscore(closes: list, window: int = 20) -> float:
    if len(closes) < window:
        return 0.0
    subset = closes[:window]
    mean   = sum(subset) / window
    var    = sum((x - mean)**2 for x in subset) / window
    std    = math.sqrt(var) if var > 0 else 0.001
    return (closes[0] - mean) / std

def momentum_factor(closes: list) -> float:
    if len(closes) < 21:
        return 0.0
    r5  = (closes[0] - closes[4])  / closes[4]  * 100 if closes[4]  != 0 else 0
    r10 = (closes[0] - closes[9])  / closes[9]  * 100 if closes[9]  != 0 else 0
    r20 = (closes[0] - closes[19]) / closes[19] * 100 if closes[19] != 0 else 0
    return (r5 * 0.5) + (r10 * 0.3) + (r20 * 0.2)

def trend_efficiency(closes: list, highs: list, lows: list, window: int = 10) -> float:
    if len(closes) < window + 1:
        return 0.5
    net_move  = abs(closes[0] - closes[window])
    path      = sum(abs(closes[i] - closes[i+1]) for i in range(window))
    return net_move / path if path > 0 else 0.0

def vol_regime(closes: list) -> tuple:
    if len(closes) < 31:
        return "NORMAL", 1.0
    returns10 = [abs((closes[i]-closes[i+1])/closes[i+1]) for i in range(10) if closes[i+1] != 0]
    returns30 = [abs((closes[i]-closes[i+1])/closes[i+1]) for i in range(30) if closes[i+1] != 0]
    v10 = sum(returns10) / len(returns10) if returns10 else 0.02
    v30 = sum(returns30) / len(returns30) if returns30 else 0.02
    ratio = v10 / v30 if v30 > 0 else 1.0
    if ratio < 0.7:
        return "LOW", 1.2
    elif ratio > 1.5:
        return "HIGH", 0.7
    elif ratio > 2.0:
        return "EXTREME", 0.3
    return "NORMAL", 1.0

def vwap_position(candles: list) -> float:
    try:
        total_vol = sum(float(c.get("volume", 1)) for c in candles[:20])
        if total_vol == 0:
            return 0.0
        vwap = sum(
            ((float(c["high"]) + float(c["low"]) + float(c["close"])) / 3)
            * float(c.get("volume", 1))
            for c in candles[:20]
        ) / total_vol
        cur = float(candles[0]["close"])
        return (cur - vwap) / vwap * 100
    except Exception:
        return 0.0

def detect_regime(closes: list, highs: list, lows: list) -> str:
    eff = trend_efficiency(closes, highs, lows, 10)
    if eff >= 0.55:
        return "TRENDING"
    elif eff >= 0.35:
        return "MIXED"
    return "RANGING"

def calc_atr(candles: list, period: int = 14) -> float:
    if len(candles) < period + 1:
        return 1.0
    trs = []
    for i in range(period):
        h = float(candles[i]["high"])
        l = float(candles[i]["low"])
        pc = float(candles[i+1]["close"])
        trs.append(max(h-l, abs(h-pc), abs(l-pc)))
    return sum(trs) / period

def support_resistance(candles: list, window: int = 20) -> tuple:
    if len(candles) < window:
        return 0.0, 0.0
    highs = [float(c["high"])  for c in candles[:window]]
    lows  = [float(c["low"])   for c in candles[:window]]
    return min(lows), max(highs)

def sharpe_weight(key: str, tf: str, score: float) -> float:
    hist_key = f"{key}_{tf}"
    if hist_key not in signal_history:
        signal_history[hist_key] = []
    signal_history[hist_key].append(score)
    if len(signal_history[hist_key]) > 20:
        signal_history[hist_key] = signal_history[hist_key][-20:]
    hist = signal_history[hist_key]
    if len(hist) < 3:
        return 1.0
    mean = sum(hist) / len(hist)
    std  = math.sqrt(sum((x-mean)**2 for x in hist) / len(hist))
    return mean / std if std > 0 else 1.0

def quant_score(candles: list, key: str, tf: str, regime: str) -> dict:
    if len(candles) < 21:
        return {"score": 50, "verdict": "NEUTRAL", "z": 0, "mom": 0,
                "eff": 0.5, "vwap_pos": 0, "vol_regime": "NORMAL",
                "vol_mult": 1.0, "atr": 1.0}

    closes = [float(c["close"]) for c in candles]
    highs  = [float(c["high"])  for c in candles]
    lows   = [float(c["low"])   for c in candles]

    z       = zscore(closes, 20)
    mom     = momentum_factor(closes)
    eff     = trend_efficiency(closes, highs, lows, 10)
    vr, vm  = vol_regime(closes)
    vwap_p  = vwap_position(candles)
    atr     = calc_atr(candles, 14)

    score = 50

    if regime == "TRENDING":
        # Momentum leads
        if mom > 3:    score += 25
        elif mom > 1:  score += 15
        elif mom > 0:  score += 8
        elif mom < -3: score -= 25
        elif mom < -1: score -= 15
        else:          score -= 8

        # Z-score as filter only — suppress if too stretched
        if z > 2.5:    score -= 10
        elif z < -2.5: score += 10

        # Trend efficiency boosts confidence
        if eff > 0.65: score += 12
        elif eff > 0.5:score += 6

    elif regime == "RANGING":
        # Mean reversion leads
        if z > 2.0:    score -= 25
        elif z > 1.5:  score -= 15
        elif z < -2.0: score += 25
        elif z < -1.5: score += 15

        # Momentum as filter only
        if mom > 2:    score -= 5
        elif mom < -2: score += 5

    else:  # MIXED
        # Equal weight
        if mom > 1:    score += 10
        elif mom < -1: score -= 10
        if z > 1.8:    score -= 12
        elif z < -1.8: score += 12
        if eff > 0.5:  score += 6

    # VWAP filter — applies always
    if vwap_p > 0.5:   score += 8
    elif vwap_p < -0.5:score -= 8

    # Vol multiplier
    score = 50 + (score - 50) * vm

    # Session weight
    _, sess_mult = current_session()
    score = 50 + (score - 50) * min(sess_mult, 1.3)

    score = max(0, min(100, round(score)))

    # Sharpe weight for aggregation
    sw = sharpe_weight(key, tf, score)

    if score >= 70:   verdict = "STRONG BUY"
    elif score >= 58: verdict = "BUY"
    elif score >= 45: verdict = "NEUTRAL"
    elif score >= 32: verdict = "SELL"
    else:             verdict = "STRONG SELL"

    return {
        "score":      score,
        "verdict":    verdict,
        "z":          round(z, 2),
        "mom":        round(mom, 2),
        "eff":        round(eff, 2),
        "vwap_pos":   round(vwap_p, 2),
        "vol_regime": vr,
        "vol_mult":   round(vm, 2),
        "atr":        round(atr, 4),
        "sharpe_w":   round(sw, 2),
    }

def composite(tf_scores: dict) -> dict:
    if not tf_scores:
        return {"score": 50, "verdict": "NEUTRAL", "emoji": "🟡"}

    total_w = sum(s["sharpe_w"] for s in tf_scores.values())
    if total_w == 0:
        total_w = 1
    score = sum(
        s["score"] * s["sharpe_w"] for s in tf_scores.values()
    ) / total_w
    score = round(score)

    if score >= 70:   verdict, em = "STRONG BUY",  "🟢"
    elif score >= 58: verdict, em = "BUY",          "🟢"
    elif score >= 45: verdict, em = "NEUTRAL",      "🟡"
    elif score >= 32: verdict, em = "SELL",         "🔴"
    else:             verdict, em = "STRONG SELL",  "🔴"

    return {"score": score, "verdict": verdict, "emoji": em}

# ── Trade setup generator ─────────────────────────────────────────────────────
def trade_setup(candles_15m: list, comp: dict, regime: str,
                tf_scores: dict, key: str) -> dict:
    if len(candles_15m) < 15:
        return {}

    closes = [float(c["close"]) for c in candles_15m]
    cur    = closes[0]
    atr    = calc_atr(candles_15m, 14)
    sup, res = support_resistance(candles_15m, 20)
    score  = comp["score"]
    verdict= comp["verdict"]

    direction = None
    if score >= 58:
        direction = "LONG"
    elif score <= 42:
        direction = "SHORT"
    else:
        return {}

    # Entry zone — tight around current price in direction of signal
    if direction == "LONG":
        entry_low  = round(cur - atr * 0.3, 4)
        entry_high = round(cur + atr * 0.1, 4)
        stop       = round(cur - atr * 1.5,  4)
        tp1        = round(cur + atr * 2.0,  4)
        tp2        = round(cur + atr * 3.0,  4)
        tp3        = round(cur + atr * 4.0,  4)
    else:
        entry_low  = round(cur - atr * 0.1, 4)
        entry_high = round(cur + atr * 0.3, 4)
        stop       = round(cur + atr * 1.5,  4)
        tp1        = round(cur - atr * 2.0,  4)
        tp2        = round(cur - atr * 3.0,  4)
        tp3        = round(cur - atr * 4.0,  4)

    aligned = sum(1 for s in tf_scores.values()
                  if (direction == "LONG"  and "BUY"  in s["verdict"]) or
                     (direction == "SHORT" and "SELL" in s["verdict"]))
    confidence = round(score * (aligned / max(len(tf_scores), 1)))
    confidence = min(confidence, 95)

    return {
        "direction":   direction,
        "entry_low":   entry_low,
        "entry_high":  entry_high,
        "stop":        stop,
        "tp1":         tp1,
        "tp2":         tp2,
        "tp3":         tp3,
        "atr":         round(atr, 4),
        "confidence":  confidence,
        "aligned":     aligned,
        "total_tf":    len(tf_scores),
        "regime":      regime,
        "support":     round(sup, 4),
        "resistance":  round(res, 4),
    }

# ── Helpers ───────────────────────────────────────────────────────────────────
def bar(score: int) -> str:
    f = round(score / 10)
    return "[" + "█"*f + "░"*(10-f) + f"] {score}/100"

def ve(score: int) -> str:
    if score >= 58: return "🟢"
    if score >= 45: return "🟡"
    return "🔴"

def sep() -> str:
    return "─" * 30

def fmt_price(p: float, sym: str) -> str:
    if sym in ("GLD", "UUP", "VIXY"):
        return f"${p:.2f}"
    return f"${p:,.2f}"

# ── Correlation signal ────────────────────────────────────────────────────────
async def corr_signal(key: str) -> str:
    d = INSTRUMENTS[key]
    try:
        price = await td_price(d["corr_sym"])
        if price == 0:
            return f"{d['corr_name']} data unavailable"
        if key == "GOLD":
            if price > 29:
                return f"UUP (DXY proxy) strong at ${price:.2f} — headwind for Gold"
            elif price < 27:
                return f"UUP (DXY proxy) weak at ${price:.2f} — tailwind for Gold"
            return f"UUP (DXY proxy) neutral at ${price:.2f} — no strong USD impact"
        else:
            if price > 22:
                return f"VIXY (VIX proxy) elevated at ${price:.2f} — risk-off pressure on US30"
            elif price < 16:
                return f"VIXY (VIX proxy) low at ${price:.2f} — risk-on supportive for US30"
            return f"VIXY (VIX proxy) neutral at ${price:.2f} — no extreme fear or greed"
    except Exception as e:
        return f"Correlation check error: {e}"

# ── Full report ───────────────────────────────────────────────────────────────
async def full_report(key: str) -> str:
    d      = INSTRUMENTS[key]
    now    = datetime.now(ZoneInfo(TZ)).strftime("%d %b %Y  %H:%M EAT")
    sess, smult = current_session()

    tf_scores   = {}
    regime      = "MIXED"
    price_str   = "N/A"
    chg_str     = "N/A"
    chg_icon    = ""
    candles_15m = []

    timeframes = ["15min", "1h", "4h", "1day", "1week"]
    for tf in timeframes:
        candles = await td_candles(d["symbol"], tf, 50)
        if not candles:
            await asyncio.sleep(1)
            continue
        if tf == "15min":
            candles_15m = candles
        closes = [float(c["close"]) for c in candles]
        highs  = [float(c["high"])  for c in candles]
        lows   = [float(c["low"])   for c in candles]
        if tf == "1day":
            regime = detect_regime(closes, highs, lows)
            if len(candles) >= 2:
                cur  = float(candles[0]["close"])
                prev = float(candles[1]["close"])
                chg  = (cur - prev) / prev * 100
                price_str = fmt_price(cur, d["symbol"])
                chg_str   = f"{chg:+.2f}%"
                chg_icon  = "📈" if chg >= 0 else "📉"
        tf_scores[tf] = quant_score(candles, key, tf, regime)
        await asyncio.sleep(0.5)

    comp = composite(tf_scores)

    # Store history
    if key not in weekly_history:
        weekly_history[key] = []
    weekly_history[key].append((
        datetime.now(ZoneInfo(TZ)), price_str, comp["score"], comp["verdict"]
    ))
    if len(weekly_history[key]) > 500:
        weekly_history[key] = weekly_history[key][-500:]

    # Support resistance from daily
    sup_val, res_val = 0.0, 0.0
    if "1day" in tf_scores:
        dc = await td_candles(d["symbol"], "1day", 20)
        if dc:
            sup_val, res_val = support_resistance(dc, 20)

    # Trade setup from 15min
    setup = {}
    if candles_15m:
        setup = trade_setup(candles_15m, comp, regime, tf_scores, key)

    # News
    cats, risks = await get_news(d["news_sym"])

    # Correlation
    corr = await corr_signal(key)

    # TF table
    tf_labels = {"15min":"15M","1h":"1H","4h":"4H","1day":"Daily","1week":"Weekly"}
    lines = []
    for tf, label in tf_labels.items():
        if tf in tf_scores:
            s  = tf_scores[tf]
            b  = "█"*round(s["score"]/10) + "░"*(10-round(s["score"]/10))
            lines.append(
                f"{label:<6} [{b}] {s['score']:>3}  {ve(s['score'])} {s['verdict']}"
            )
        else:
            lines.append(f"{label:<6} [░░░░░░░░░░]  --  No data")
    tf_block = "\n".join(lines)

    # Quant signal summary from daily
    qs = tf_scores.get("1day", {})
    quant_block = (
        f"Z-Score:    {qs.get('z', 0):+.2f}  "
        f"({'stretched' if abs(qs.get('z',0))>2 else 'normal'})\n"
        f"Momentum:   {qs.get('mom', 0):+.2f}%  "
        f"({'strong' if abs(qs.get('mom',0))>2 else 'moderate'})\n"
        f"Trend Eff:  {qs.get('eff', 0):.2f}  "
        f"({'trending' if qs.get('eff',0)>0.55 else 'ranging'})\n"
        f"VWAP Pos:   {qs.get('vwap_pos', 0):+.2f}%  "
        f"({'above' if qs.get('vwap_pos',0)>0 else 'below'})\n"
        f"Vol Regime: {qs.get('vol_regime','NORMAL')}"
    )

    # Alignment
    vs    = [tf_scores[tf]["verdict"] for tf in tf_scores]
    buys  = sum(1 for v in vs if "BUY" in v and "SELL" not in v)
    sells = sum(1 for v in vs if "SELL" in v)
    if buys >= 4:    align = "Strong bullish alignment across timeframes"
    elif buys >= 3:  align = "Moderate bullish alignment"
    elif sells >= 4: align = "Strong bearish alignment across timeframes"
    elif sells >= 3: align = "Moderate bearish alignment"
    else:            align = "Mixed signals — trade with caution"

    # Build setup block
    setup_block = ""
    if setup:
        rr = "1:2 / 1:3 / 1:4"
        setup_block = (
            f"\n{sep()}\n"
            f"TRADE SETUP  {sess}\n\n"
            f"Direction:  {setup['direction']}\n"
            f"Entry zone: {setup['entry_low']} — {setup['entry_high']}\n"
            f"Stop loss:  {setup['stop']}  (1.5x ATR)\n"
            f"Target 1:   {setup['tp1']}  RR 1:2\n"
            f"Target 2:   {setup['tp2']}  RR 1:3\n"
            f"Target 3:   {setup['tp3']}  RR 1:4\n\n"
            f"Confidence: {setup['confidence']}%\n"
            f"TF aligned: {setup['aligned']} of {setup['total_tf']}\n"
            f"Support:    {setup['support']}\n"
            f"Resistance: {setup['resistance']}"
        )
    else:
        setup_block = f"\n{sep()}\nTRADE SETUP\nNo high-confidence setup at this time"

    # News blocks
    cat_block = "\n".join(f"  + {c}" for c in cats) if cats else "  Fetching latest news..."
    risk_block= "\n".join(f"  - {r}" for r in risks) if risks else "  Fetching latest news..."

    return (
        f"{d['emoji']} {d['name']}  {d['class']}\n"
        f"{now}  {sess}\n"
        f"{sep()}\n\n"
        f"Price:   {price_str}  {chg_icon} {chg_str}\n"
        f"Spread:  {d['spread']}\n"
        f"Regime:  {regime}\n\n"
        f"{sep()}\n"
        f"QUANT SCORES  (Sharpe-weighted)\n\n"
        f"{tf_block}\n\n"
        f"Composite: {bar(comp['score'])}\n"
        f"Verdict:   {comp['emoji']} {comp['verdict']}\n"
        f"{align}\n\n"
        f"{sep()}\n"
        f"QUANT SIGNALS  (Daily)\n\n"
        f"{quant_block}\n\n"
        f"{sep()}\n"
        f"CORRELATION\n"
        f"{corr}\n"
        f"{setup_block}\n\n"
        f"{sep()}\n"
        f"CATALYSTS  (live news)\n"
        f"{cat_block}\n\n"
        f"RISKS  (live news)\n"
        f"{risk_block}\n\n"
        f"{sep()}\n"
        f"POSITIONING\n"
        f"{d['cot']}\n\n"
        f"{sep()}\n"
        f"{comp['emoji']} VERDICT: {comp['verdict']}\n"
        f"Regime: {regime}  Vol: {qs.get('vol_regime','N/A')}  "
        f"Session: {sess}"
    )

# ── Summary ───────────────────────────────────────────────────────────────────
async def summary_msg() -> str:
    now  = datetime.now(ZoneInfo(TZ)).strftime("%d %b %Y  %H:%M EAT")
    sess, _ = current_session()
    rows = []
    for key, d in INSTRUMENTS.items():
        candles = await td_candles(d["symbol"], "1day", 50)
        price_str, chg_str, arrow = "N/A", "N/A", ""
        regime = "MIXED"
        if candles and len(candles) >= 2:
            closes = [float(c["close"]) for c in candles]
            highs  = [float(c["high"])  for c in candles]
            lows   = [float(c["low"])   for c in candles]
            cur    = closes[0]
            prev   = closes[1]
            chg    = (cur - prev) / prev * 100
            price_str = fmt_price(cur, d["symbol"])
            chg_str   = f"{chg:+.2f}%"
            arrow     = "▲" if chg >= 0 else "▼"
            regime    = detect_regime(closes, highs, lows)
            tf_scores = {"1day": quant_score(candles, key, "1day", regime)}
            comp      = composite(tf_scores)
        else:
            comp = {"score": 50, "verdict": "N/A", "emoji": "⚪"}

        b = "█"*round(comp["score"]/10) + "░"*(10-round(comp["score"]/10))
        rows.append(
            f"{d['emoji']} {d['name']}\n"
            f"   [{b}] {comp['score']}/100\n"
            f"   Price:   {price_str}  {arrow} {chg_str}\n"
            f"   Regime:  {regime}\n"
            f"   Verdict: {comp['emoji']} {comp['verdict']}"
        )
        await asyncio.sleep(1)

    return (
        f"RISK SCORE SNAPSHOT\n"
        f"{now}  {sess}\n"
        f"{sep()}\n\n"
        f"{chr(10).join(rows)}\n\n"
        f"{sep()}\n"
        f"Use /gold or /us30 for full quant report\n"
        f"Use /setup for trade setups"
    )

# ── Setup command ─────────────────────────────────────────────────────────────
async def setup_msg() -> str:
    now  = datetime.now(ZoneInfo(TZ)).strftime("%d %b %Y  %H:%M EAT")
    sess, _ = current_session()
    blocks = []
    for key, d in INSTRUMENTS.items():
        candles_15m = await td_candles(d["symbol"], "15min", 50)
        candles_1d  = await td_candles(d["symbol"], "1day",  50)
        if not candles_15m or not candles_1d:
            blocks.append(f"{d['emoji']} {d['name']}\n   No data available")
            continue
        closes = [float(c["close"]) for c in candles_1d]
        highs  = [float(c["high"])  for c in candles_1d]
        lows   = [float(c["low"])   for c in candles_1d]
        regime = detect_regime(closes, highs, lows)
        tf_scores = {}
        for tf in ["15min","1h","4h","1day"]:
            c = await td_candles(d["symbol"], tf, 50)
            if c:
                tf_scores[tf] = quant_score(c, key, tf, regime)
            await asyncio.sleep(0.5)
        comp  = composite(tf_scores)
        setup = trade_setup(candles_15m, comp, regime, tf_scores, key)
        if setup:
            blocks.append(
                f"{d['emoji']} {d['name']}  {sess}\n\n"
                f"Direction:  {setup['direction']}\n"
                f"Entry:      {setup['entry_low']} — {setup['entry_high']}\n"
                f"Stop:       {setup['stop']}  (1.5x ATR)\n"
                f"Target 1:   {setup['tp1']}  RR 1:2\n"
                f"Target 2:   {setup['tp2']}  RR 1:3\n"
                f"Target 3:   {setup['tp3']}  RR 1:4\n\n"
                f"Score:      {comp['score']}/100  {comp['emoji']} {comp['verdict']}\n"
                f"Confidence: {setup['confidence']}%\n"
                f"Regime:     {regime}\n"
                f"TF aligned: {setup['aligned']} of {setup['total_tf']}\n"
                f"Support:    {setup['support']}\n"
                f"Resistance: {setup['resistance']}"
            )
        else:
            blocks.append(
                f"{d['emoji']} {d['name']}\n"
                f"Score: {comp['score']}/100  {comp['emoji']} {comp['verdict']}\n"
                f"No high-confidence setup right now\n"
                f"Regime: {regime}"
            )
        await asyncio.sleep(1)

    return (
        f"TRADE SETUPS\n"
        f"{now}  {sess}\n"
        f"{sep()}\n\n"
        f"{(chr(10)*2 + sep() + chr(10)*2).join(blocks)}\n\n"
        f"{sep()}\n"
        f"Entry zones based on 15M candles + ATR\n"
        f"Always apply your own risk management"
    )

# ── Weekly summary ────────────────────────────────────────────────────────────
async def weekly_msg() -> str:
    now      = datetime.now(ZoneInfo(TZ)).strftime("%d %b %Y  %H:%M EAT")
    week_ago = datetime.now(ZoneInfo(TZ)) - timedelta(days=7)
    rows = []
    for key, d in INSTRUMENTS.items():
        hist = [x for x in weekly_history.get(key, []) if x[0] >= week_ago]
        if len(hist) >= 2:
            scores   = [x[2] for x in hist]
            verdicts = [x[3] for x in hist]
            flips    = sum(1 for i in range(1, len(verdicts))
                          if verdicts[i] != verdicts[i-1])
            avg      = round(sum(scores) / len(scores))
            b        = "█"*round(avg/10) + "░"*(10-round(avg/10))
            rows.append(
                f"{d['emoji']} {d['name']}\n"
                f"   Avg Score:    [{b}] {avg}/100\n"
                f"   Score range:  {min(scores)} to {max(scores)}\n"
                f"   Score shift:  {scores[0]} to {scores[-1]}\n"
                f"   Verdict flips:{flips}\n"
                f"   Current:      {ve(scores[-1])} {verdicts[-1]}"
            )
        else:
            rows.append(
                f"{d['emoji']} {d['name']}\n"
                f"   Not enough data yet"
            )
    return (
        f"WEEKLY RECAP\n"
        f"{now}\n"
        f"{sep()}\n\n"
        f"{chr(10).join(rows)}\n\n"
        f"{sep()}\n"
        f"Full reports via /gold or /us30"
    )

# ── Score flip checker ────────────────────────────────────────────────────────
async def check_flips(app: Application):
    log.info("Checking score flips...")
    for key, d in INSTRUMENTS.items():
        if key not in prev_verdicts:
            prev_verdicts[key] = {}
        candles_1d = await td_candles(d["symbol"], "1day", 50)
        if not candles_1d:
            continue
        closes = [float(c["close"]) for c in candles_1d]
        highs  = [float(c["high"])  for c in candles_1d]
        lows   = [float(c["low"])   for c in candles_1d]
        regime = detect_regime(closes, highs, lows)

        for tf in ["15min", "1h", "4h", "1day"]:
            try:
                candles = await td_candles(d["symbol"], tf, 50)
                if not candles:
                    continue
                result = quant_score(candles, key, tf, regime)
                new_vd = result["verdict"]
                old_vd = prev_verdicts[key].get(tf)
                if old_vd and old_vd != new_vd:
                    label = {"15min":"15M","1h":"1H","4h":"4H","1day":"Daily"}[tf]
                    em    = "🟢" if "BUY" in new_vd else ("🔴" if "SELL" in new_vd else "🟡")
                    cmd   = "/gold" if key == "GOLD" else "/us30"
                    msg   = (
                        f"SCORE FLIP ALERT\n\n"
                        f"{d['emoji']} {d['name']}  {label}\n\n"
                        f"FROM: {old_vd}\n"
                        f"TO:   {em} {new_vd}\n\n"
                        f"Score:    {result['score']}/100\n"
                        f"Z-Score:  {result['z']:+.2f}\n"
                        f"Momentum: {result['mom']:+.2f}%\n"
                        f"Regime:   {regime}\n"
                        f"Vol:      {result['vol_regime']}\n\n"
                        f"See {cmd} for full analysis"
                    )
                    await app.bot.send_message(CHAT_ID, msg)
                    log.info(f"Flip: {key} {tf} {old_vd} to {new_vd}")
                prev_verdicts[key][tf] = new_vd
                await asyncio.sleep(1)
            except Exception as e:
                log.error(f"Flip check error {key} {tf}: {e}")

# ── Setup checker (every 15min during active sessions) ────────────────────────
async def check_setups(app: Application):
    sess, smult = current_session()
    if smult < 0.8:
        return
    log.info(f"Checking setups — {sess}")
    for key, d in INSTRUMENTS.items():
        try:
            candles_15m = await td_candles(d["symbol"], "15min", 50)
            candles_1d  = await td_candles(d["symbol"], "1day",  50)
            if not candles_15m or not candles_1d:
                continue
            closes = [float(c["close"]) for c in candles_1d]
            highs  = [float(c["high"])  for c in candles_1d]
            lows   = [float(c["low"])   for c in candles_1d]
            regime = detect_regime(closes, highs, lows)
            tf_scores = {}
            for tf in ["15min","1h","4h","1day"]:
                c = await td_candles(d["symbol"], tf, 50)
                if c:
                    tf_scores[tf] = quant_score(c, key, tf, regime)
                await asyncio.sleep(0.5)
            comp  = composite(tf_scores)
            setup = trade_setup(candles_15m, comp, regime, tf_scores, key)
            if setup and setup.get("confidence", 0) >= 70:
                msg = (
                    f"TRADE SETUP ALERT\n\n"
                    f"{d['emoji']} {d['name']}  {sess}\n\n"
                    f"Direction:  {setup['direction']}\n"
                    f"Entry:      {setup['entry_low']} — {setup['entry_high']}\n"
                    f"Stop:       {setup['stop']}  (1.5x ATR)\n"
                    f"Target 1:   {setup['tp1']}  RR 1:2\n"
                    f"Target 2:   {setup['tp2']}  RR 1:3\n"
                    f"Target 3:   {setup['tp3']}  RR 1:4\n\n"
                    f"Score:      {comp['score']}/100  {comp['emoji']} {comp['verdict']}\n"
                    f"Confidence: {setup['confidence']}%\n"
                    f"Regime:     {regime}\n"
                    f"TF aligned: {setup['aligned']} of {setup['total_tf']}"
                )
                await app.bot.send_message(CHAT_ID, msg)
                log.info(f"Setup alert sent: {key} {setup['direction']} {setup['confidence']}%")
            await asyncio.sleep(2)
        except Exception as e:
            log.error(f"Setup check error {key}: {e}")

# ── Scheduled jobs ────────────────────────────────────────────────────────────
async def daily_job(app: Application):
    log.info("Sending daily report...")
    try:
        msg = await summary_msg()
        await app.bot.send_message(CHAT_ID, msg)
        for key in INSTRUMENTS:
            await asyncio.sleep(3)
            report = await full_report(key)
            await app.bot.send_message(CHAT_ID, report, reply_markup=kb())
        log.info("Daily report done.")
    except Exception as e:
        log.error(f"Daily job error: {e}")
        await app.bot.send_message(CHAT_ID, f"Daily report error: {e}")

async def weekly_job(app: Application):
    log.info("Sending weekly summary...")
    try:
        msg = await weekly_msg()
        await app.bot.send_message(CHAT_ID, msg, reply_markup=kb())
    except Exception as e:
        log.error(f"Weekly job error: {e}")

# ── Keyboard ──────────────────────────────────────────────────────────────────
def kb():
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("🥇 Gold",      callback_data="full_GOLD"),
            InlineKeyboardButton("🏦 US30",      callback_data="full_US30"),
        ],
        [
            InlineKeyboardButton("📊 Summary",   callback_data="summary"),
            InlineKeyboardButton("🎯 Setups",    callback_data="setups"),
        ],
        [
            InlineKeyboardButton("📅 Weekly",    callback_data="weekly"),
            InlineKeyboardButton("🔄 Refresh",   callback_data="refresh"),
        ],
    ])

# ── Handlers ──────────────────────────────────────────────────────────────────
async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    text = (
        "Risk Score Bot v4  Quant Edition\n\n"
        "Commands\n"
        "/report   full quant report both instruments\n"
        "/gold     Gold full quant report\n"
        "/us30     US30 full quant report\n"
        "/setup    live trade setups with entry stop target\n"
        "/summary  quick score snapshot\n"
        "/weekly   weekly performance recap\n\n"
        "Automated\n"
        "Daily report    7:30 AM EAT\n"
        "Weekly recap    Monday 7:30 AM EAT\n"
        "Score flips     every 30 minutes\n"
        "Setup alerts    every 15 minutes during sessions\n\n"
        "Quant models\n"
        "Momentum factor across 5 10 20 bars\n"
        "Z-score mean reversion\n"
        "Trend efficiency regime filter\n"
        "VWAP institutional positioning\n"
        "Volatility regime classifier\n"
        "Sharpe-weighted aggregation\n\n"
        "Powered by Twelve Data and Finnhub"
    )
    await update.message.reply_text(text, reply_markup=kb())

async def cmd_report(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Fetching live quant data...")
    msg = await summary_msg()
    await update.message.reply_text(msg)
    for key in INSTRUMENTS:
        await asyncio.sleep(2)
        report = await full_report(key)
        await update.message.reply_text(report, reply_markup=kb())

async def cmd_gold(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Fetching live Gold quant data...")
    report = await full_report("GOLD")
    await update.message.reply_text(report, reply_markup=kb())

async def cmd_us30(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Fetching live US30 quant data...")
    report = await full_report("US30")
    await update.message.reply_text(report, reply_markup=kb())

async def cmd_setup(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Calculating trade setups...")
    msg = await setup_msg()
    await update.message.reply_text(msg, reply_markup=kb())

async def cmd_summary(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Fetching live data...")
    msg = await summary_msg()
    await update.message.reply_text(msg, reply_markup=kb())

async def cmd_weekly(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    msg = await weekly_msg()
    await update.message.reply_text(msg, reply_markup=kb())

async def btn_handler(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    if q.data in ("summary", "refresh"):
        await q.message.reply_text("Fetching live data...")
        msg = await summary_msg()
        await q.message.reply_text(msg, reply_markup=kb())
    elif q.data == "setups":
        await q.message.reply_text("Calculating trade setups...")
        msg = await setup_msg()
        await q.message.reply_text(msg, reply_markup=kb())
    elif q.data == "weekly":
        msg = await weekly_msg()
        await q.message.reply_text(msg, reply_markup=kb())
    elif q.data.startswith("full_"):
        key = q.data.split("_")[1]
        await q.message.reply_text(f"Fetching live {key} quant data...")
        report = await full_report(key)
        await q.message.reply_text(report, reply_markup=kb())

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    app = Application.builder().token(TOKEN).build()

    app.add_handler(CommandHandler("start",   cmd_start))
    app.add_handler(CommandHandler("report",  cmd_report))
    app.add_handler(CommandHandler("gold",    cmd_gold))
    app.add_handler(CommandHandler("us30",    cmd_us30))
    app.add_handler(CommandHandler("setup",   cmd_setup))
    app.add_handler(CommandHandler("summary", cmd_summary))
    app.add_handler(CommandHandler("weekly",  cmd_weekly))
    app.add_handler(CallbackQueryHandler(btn_handler))

    scheduler = AsyncIOScheduler(timezone=TZ)
    scheduler.add_job(daily_job,    "cron",     hour=HOUR, minute=MINUTE, args=[app])
    scheduler.add_job(weekly_job,   "cron",     day_of_week="mon", hour=HOUR, minute=MINUTE, args=[app])
    scheduler.add_job(check_flips,  "interval", minutes=30, args=[app])
    scheduler.add_job(check_setups, "interval", minutes=15, args=[app])
    scheduler.start()

    log.info(f"Bot v4 Quant Edition live — {HOUR:02d}:{MINUTE:02d} EAT daily")
    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()
