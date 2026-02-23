import os
import asyncio
import logging
import time
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv

import ccxt.async_support as ccxt

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

# -----------------------
# LOGGING
# -----------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger("bingx_pro")

# -----------------------
# ENV / CONFIG
# -----------------------
load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN is missing in environment variables")

EXCHANGE_NAME = os.getenv("EXCHANGE", "bingx").strip().lower()

SCAN_EVERY_SEC = int(os.getenv("SCAN_EVERY_SEC", "300"))          # tick frequency
ROTATION_BATCH = int(os.getenv("ROTATION_BATCH", "120"))          # how many symbols per tick to prefilter
CANDIDATES_TOP = int(os.getenv("CANDIDATES_TOP", "20"))           # deep analyze top N
MAX_SIGNALS_PER_TICK = int(os.getenv("MAX_SIGNALS_PER_TICK", "1"))

COOLDOWN_MIN = int(os.getenv("COOLDOWN_MIN", "180"))
COOLDOWN_SEC = COOLDOWN_MIN * 60

MIN_ATR_PCT_15M = float(os.getenv("MIN_ATR_PCT_15M", "0.55"))     # volatility filter on 15m
MIN_ADX_15M = float(os.getenv("MIN_ADX_15M", "16"))
MIN_ADX_1H = float(os.getenv("MIN_ADX_1H", "18"))

# Strategy parameters
EMA_FAST = 50
EMA_SLOW = 200
ATR_LEN = 14
ADX_LEN = 14
RSI_LEN = 14

LEVEL_LOOKBACK_15M = 48               # 12h on 15m
BREAK_ATR_K = 0.10                    # breakout confirm offset (5m) in ATR
RETEST_ATR_K = 0.25                   # retest zone (5m) in ATR
RETEST_MAX_BARS_5M = 6                # 30 minutes

SL_ATR_K = 1.25
TP1_ATR_K = 1.0
TP2_ATR_K = 2.2

MAX_CONCURRENCY = 6
SEM = asyncio.Semaphore(MAX_CONCURRENCY)

# -----------------------
# STATE
# -----------------------
SUBSCRIBERS: set[int] = set()
UNIVERSE: List[str] = []              # all USDT linear swaps
ROT_IDX: int = 0

LAST_SENT: Dict[Tuple[str, str], float] = {}   # (symbol, direction) -> ts
PENDING_BREAK: Dict[str, Dict] = {}            # symbol -> pending breakout state

# -----------------------
# INDICATORS
# -----------------------
def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()

def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            (df["high"] - df["low"]).abs(),
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr

def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    tr = true_range(df)
    return tr.rolling(length).mean()

def rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    gain = (delta.where(delta > 0, 0.0)).rolling(length).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(length).mean()
    rs = gain / (loss.replace(0, np.nan))
    return 100 - (100 / (1 + rs))

def adx(df: pd.DataFrame, length: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = true_range(df)
    atr_val = tr.rolling(length).mean()

    plus_di = 100 * (pd.Series(plus_dm).rolling(length).mean() / atr_val)
    minus_di = 100 * (pd.Series(minus_dm).rolling(length).mean() / atr_val)

    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    return dx.rolling(length).mean()

def fmt_price(x: float) -> str:
    if x >= 1000:
        return f"{x:.2f}"
    if x >= 1:
        return f"{x:.4f}"
    return f"{x:.6f}"

# -----------------------
# DATA
# -----------------------
def parse_ohlcv(ohlcv: List[List[float]]) -> pd.DataFrame:
    df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna()
    return df

async def fetch_df(ex, symbol: str, tf: str, limit: int = 300) -> Optional[pd.DataFrame]:
    async with SEM:
        try:
            ohlcv = await ex.fetch_ohlcv(symbol, timeframe=tf, limit=limit)
            if not ohlcv or len(ohlcv) < max(120, ATR_LEN + 50):
                return None
            return parse_ohlcv(ohlcv)
        except Exception:
            return None

async def get_exchange():
    if not hasattr(ccxt, EXCHANGE_NAME):
        raise RuntimeError(f"Exchange '{EXCHANGE_NAME}' not found in ccxt")
    ex_class = getattr(ccxt, EXCHANGE_NAME)
    ex = ex_class({
        "enableRateLimit": True,
        "options": {"defaultType": "swap"},
    })
    return ex

async def build_universe(ex) -> List[str]:
    # load markets and filter: swap + linear + quote USDT
    markets = await ex.load_markets()
    out = []
    for sym, m in markets.items():
        try:
            if not m.get("swap"):
                continue
            # linear swaps are USDT-margined; in ccxt often m["linear"] True
            if m.get("linear") is not True:
                continue
            if (m.get("quote") or "") != "USDT":
                continue
            # avoid expiring futures
            if m.get("future"):
                continue
            out.append(sym)
        except Exception:
            continue
    out = sorted(set(out))
    return out

# -----------------------
# STRATEGY CORE (MTF)
# -----------------------
def trend_bias(df_4h: pd.DataFrame, df_1h: pd.DataFrame) -> Optional[str]:
    # 4h regime: price vs EMA200
    c4 = df_4h["close"]
    e200_4 = ema(c4, EMA_SLOW).iloc[-1]
    close4 = c4.iloc[-1]
    atr4 = atr(df_4h, ATR_LEN).iloc[-1]
    if pd.isna(e200_4) or pd.isna(atr4):
        return None
    # avoid "stuck near EMA200"
    if abs(close4 - e200_4) < 0.30 * atr4:
        return None
    regime = "LONG" if close4 > e200_4 else "SHORT" if close4 < e200_4 else None
    if regime is None:
        return None

    # 1h confirm: EMA50 vs EMA200 + ADX
    c1 = df_1h["close"]
    e50_1 = ema(c1, EMA_FAST).iloc[-1]
    e200_1 = ema(c1, EMA_SLOW).iloc[-1]
    adx1 = adx(df_1h, ADX_LEN).iloc[-1]
    if pd.isna(e50_1) or pd.isna(e200_1) or pd.isna(adx1):
        return None
    if adx1 < MIN_ADX_1H:
        return None

    if regime == "LONG" and (e50_1 > e200_1):
        return "LONG"
    if regime == "SHORT" and (e50_1 < e200_1):
        return "SHORT"
    return None

def level_from_15m(df_15m: pd.DataFrame, bias: str) -> Optional[float]:
    if len(df_15m) < LEVEL_LOOKBACK_15M + 5:
        return None

    df = df_15m.copy()
    df["atr"] = atr(df, ATR_LEN)
    df["adx"] = adx(df, ADX_LEN)
    df["rsi"] = rsi(df["close"], RSI_LEN)

    last = df.iloc[-1]
    atr15 = last["atr"]
    if pd.isna(atr15) or atr15 <= 0:
        return None

    atr_pct = float(atr15 / last["close"] * 100.0)
    if atr_pct < MIN_ATR_PCT_15M:
        return None

    if pd.isna(last["adx"]) or float(last["adx"]) < MIN_ADX_15M:
        return None

    # "impulse candle" filter: body >= 0.35 ATR
    body = abs(float(last["close"] - last["open"]))
    if body < 0.35 * float(atr15):
        return None

    # choose breakout level
    window = df.iloc[-(LEVEL_LOOKBACK_15M + 1):-1]
    if bias == "LONG":
        level = float(window["high"].max())
        # avoid "overheated" RSI on 15m
        if float(last["rsi"]) > 78:
            return None
        return level
    else:
        level = float(window["low"].min())
        if float(last["rsi"]) < 22:
            return None
        return level

def check_break_and_retest_5m(df_5m: pd.DataFrame, bias: str, level: float) -> Optional[Tuple[float, float, float, float]]:
    """
    Returns (entry, sl, tp1, tp2) when:
    - breakout confirm happened (close beyond level +/- 0.10*ATR)
    - then retest in zone within RETEST_MAX_BARS_5M and rebound candle
    """
    if len(df_5m) < 120:
        return None

    df = df_5m.copy()
    df["atr"] = atr(df, ATR_LEN)
    df["ema20"] = ema(df["close"], 20)

    # use last ATR(5m)
    atr5 = float(df["atr"].iloc[-1]) if pd.notna(df["atr"].iloc[-1]) else 0.0
    if atr5 <= 0:
        return None

    closes = df["close"].values
    opens = df["open"].values
    highs = df["high"].values
    lows = df["low"].values
    ema20_last = float(df["ema20"].iloc[-1])

    # scan last N bars for breakout then retest
    n = RETEST_MAX_BARS_5M
    bars = df.iloc[-(n + 4):].reset_index(drop=True)  # small slice
    if len(bars) < n + 2:
        return None

    # identify first breakout bar in slice
    level_off = BREAK_ATR_K * atr5
    retest_off = RETEST_ATR_K * atr5

    # We want: breakout happened earlier, retest later.
    breakout_idx = None
    for i in range(0, len(bars)):
        c = float(bars["close"].iloc[i])
        if bias == "LONG":
            if c > level + level_off:
                breakout_idx = i
                break
        else:
            if c < level - level_off:
                breakout_idx = i
                break

    if breakout_idx is None:
        return None

    # After breakout: look for retest within next n bars
    # Retest candle: touches zone and closes back in correct side + rebound candle quality
    for j in range(breakout_idx + 1, min(breakout_idx + 1 + n, len(bars))):
        o = float(bars["open"].iloc[j])
        c = float(bars["close"].iloc[j])
        h = float(bars["high"].iloc[j])
        l = float(bars["low"].iloc[j])

        body = abs(c - o)

        if bias == "LONG":
            touched = (l <= level + retest_off)
            closed_ok = (c > level)
            rebound = (c > o) and (body >= 0.25 * atr5) and (c > ema20_last)
            if touched and closed_ok and rebound:
                entry = c
                sl = entry - SL_ATR_K * atr5
                tp1 = entry + TP1_ATR_K * atr5
                tp2 = entry + TP2_ATR_K * atr5
                return (entry, sl, tp1, tp2)
        else:
            touched = (h >= level - retest_off)
            closed_ok = (c < level)
            rebound = (c < o) and (body >= 0.25 * atr5) and (c < ema20_last)
            if touched and closed_ok and rebound:
                entry = c
                sl = entry + SL_ATR_K * atr5
                tp1 = entry - TP1_ATR_K * atr5
                tp2 = entry - TP2_ATR_K * atr5
                return (entry, sl, tp1, tp2)

    return None

@dataclass
class Signal:
    symbol: str
    direction: str
    entry: float
    sl: float
    tp1: float
    tp2: float
    level: float

def cooldown_ok(symbol: str, direction: str) -> bool:
    t = LAST_SENT.get((symbol, direction), 0.0)
    return (time.time() - t) >= COOLDOWN_SEC

# -----------------------
# SCAN PIPELINE
# -----------------------
async def prefilter_symbol(ex, symbol: str) -> Optional[Tuple[str, float]]:
    """
    Cheap prefilter: 15m ATR% + 15m volume sum
    returns (symbol, score) where score ranks candidates
    """
    df15 = await fetch_df(ex, symbol, "15m", 220)
    if df15 is None:
        return None
    df15["atr"] = atr(df15, ATR_LEN)
    atr15 = df15["atr"].iloc[-1]
    if pd.isna(atr15) or float(atr15) <= 0:
        return None
    close = float(df15["close"].iloc[-1])
    atr_pct = float(atr15 / close * 100.0)
    if atr_pct < MIN_ATR_PCT_15M:
        return None
    vol = float(df15["volume"].iloc[-96:].sum())  # ~24h on 15m
    # rank by volatility + liquidity proxy
    score = atr_pct * 2.0 + np.log10(max(1.0, vol)) * 0.8
    return (symbol, float(score))

async def deep_analyze(ex, symbol: str) -> Optional[Signal]:
    # Fetch higher TFs
    df4 = await fetch_df(ex, symbol, "4h", 260)
    df1 = await fetch_df(ex, symbol, "1h", 260)
    df15 = await fetch_df(ex, symbol, "15m", 260)
    df5 = await fetch_df(ex, symbol, "5m", 260)
    if any(x is None for x in (df4, df1, df15, df5)):
        return None

    bias = trend_bias(df4, df1)
    if bias is None:
        return None

    level = level_from_15m(df15, bias)
    if level is None:
        return None

    res = check_break_and_retest_5m(df5, bias, level)
    if res is None:
        return None

    entry, sl, tp1, tp2 = res

    # final cooldown gate
    if not cooldown_ok(symbol, bias):
        return None

    return Signal(
        symbol=symbol,
        direction=bias,
        entry=float(entry),
        sl=float(sl),
        tp1=float(tp1),
        tp2=float(tp2),
        level=float(level),
    )

async def scan_tick(app: Application):
    global ROT_IDX

    ex = app.bot_data.get("exchange")
    if ex is None:
        return
    if not UNIVERSE:
        return

    # rotation slice
    n = len(UNIVERSE)
    start = ROT_IDX % n
    end = min(n, start + ROTATION_BATCH)
    batch = UNIVERSE[start:end]
    ROT_IDX = end if end < n else 0

    # cheap prefilter
    pre: List[Tuple[str, float]] = []
    async def pf(sym: str):
        r = await prefilter_symbol(ex, sym)
        if r:
            pre.append(r)

    await asyncio.gather(*[pf(s) for s in batch])

    if not pre:
        return

    pre.sort(key=lambda x: x[1], reverse=True)
    candidates = [s for s, _ in pre[:CANDIDATES_TOP]]

    # deep analyze candidates
    found: List[Signal] = []
    async def da(sym: str):
        sig = await deep_analyze(ex, sym)
        if sig:
            found.append(sig)

    await asyncio.gather(*[da(s) for s in candidates])

    if not found:
        return

    # send best first (we can add scoring later)
    found = found[:MAX_SIGNALS_PER_TICK]
    for sig in found:
        LAST_SENT[(sig.symbol, sig.direction)] = time.time()

        text = (
            f"🚨 <b>{sig.symbol}</b> — <b>{sig.direction}</b>\n\n"
            f"TF: <b>4h/1h фильтр</b> | <b>15m уровень</b> | <b>5m вход</b>\n"
            f"Уровень (15m): <code>{fmt_price(sig.level)}</code>\n\n"
            f"Entry: <code>{fmt_price(sig.entry)}</code>\n"
            f"SL: <code>{fmt_price(sig.sl)}</code>\n"
            f"TP1: <code>{fmt_price(sig.tp1)}</code>\n"
            f"TP2: <code>{fmt_price(sig.tp2)}</code>\n\n"
            f"Логика: подтверждение пробоя + ретест (5m).\n"
            f"⚠️ Не финсовет."
        )

        for chat_id in list(SUBSCRIBERS):
            try:
                await app.bot.send_message(chat_id=chat_id, text=text, parse_mode="HTML")
            except Exception:
                pass

async def scanner_loop(app: Application):
    await asyncio.sleep(3)
    while True:
        try:
            await scan_tick(app)
        except Exception as e:
            log.warning("scan_tick error: %s", e)
        await asyncio.sleep(max(30, SCAN_EVERY_SEC))

# -----------------------
# TELEGRAM COMMANDS
# -----------------------
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    SUBSCRIBERS.add(chat_id)

    n = len(UNIVERSE)
    await update.message.reply_text(
        "✅ Готово. Ты подписан на сигналы.\n\n"
        "Команды:\n"
        "/status — статус\n"
        "/now — принудительный тик скана\n"
        "/universe — сколько USDT perpetual найдено\n"
        "/off — отключить для этого чата\n\n"
        "Логика: 4h/1h фильтр → 15m уровень → 5m подтверждение+ретест.\n"
        "⚠️ Не финсовет."
    )
    await cmd_status(update, context)

async def cmd_off(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    SUBSCRIBERS.discard(chat_id)
    await update.message.reply_text("⛔ Ок, отключил для этого чата. Чтобы включить снова — /start")

async def cmd_universe(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(f"USDT perpetual (linear swap) найдено: {len(UNIVERSE)}")

async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ex = context.application.bot_data.get("exchange_name", EXCHANGE_NAME)
    await update.message.reply_text(
        f"Статус ✅\n"
        f"Exchange: {ex}\n"
        f"Universe: {len(UNIVERSE)}\n"
        f"Tick: каждые {SCAN_EVERY_SEC} сек\n"
        f"Rotation batch: {ROTATION_BATCH}\n"
        f"Candidates top: {CANDIDATES_TOP}\n"
        f"Cooldown: {COOLDOWN_MIN} мин\n"
        f"ATR%15m >= {MIN_ATR_PCT_15M}\n"
        f"ADX15m >= {MIN_ADX_15M}, ADX1h >= {MIN_ADX_1H}\n"
        f"Подписчиков: {len(SUBSCRIBERS)}"
    )

async def cmd_now(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("⏳ Запускаю тик скана…")
    await scan_tick(context.application)
    await update.message.reply_text("✅ Готово. Если сетап был — я отправил сигнал.")

# -----------------------
# STARTUP / SHUTDOWN
# -----------------------
async def on_startup(app: Application):
    ex = await get_exchange()
    app.bot_data["exchange"] = ex
    app.bot_data["exchange_name"] = EXCHANGE_NAME

    uni = await build_universe(ex)
    global UNIVERSE
    UNIVERSE = uni

    log.info("Universe built: %s symbols", len(UNIVERSE))
    asyncio.create_task(scanner_loop(app))

async def on_shutdown(app: Application):
    ex = app.bot_data.get("exchange")
    try:
        if ex:
            await ex.close()
    except Exception:
        pass

# -----------------------
# MAIN
# -----------------------
def main():
    app = Application.builder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("universe", cmd_universe))
    app.add_handler(CommandHandler("now", cmd_now))
    app.add_handler(CommandHandler("off", cmd_off))

    app.post_init = on_startup
    app.post_shutdown = on_shutdown

    log.info("BOT STARTING...")
    app.run_polling(allowed_updates=Update.ALL_TYPES, close_loop=False)

if __name__ == "__main__":
    main()
