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

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    ContextTypes,
)

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

# scan mechanics
SCAN_EVERY_SEC_DEFAULT = int(os.getenv("SCAN_EVERY_SEC", "300"))      # tick frequency
ROTATION_BATCH_DEFAULT = int(os.getenv("ROTATION_BATCH", "120"))      # symbols per tick (prefilter)
CANDIDATES_TOP_DEFAULT = int(os.getenv("CANDIDATES_TOP", "20"))       # deep analyze top N
MAX_SIGNALS_PER_TICK_DEFAULT = int(os.getenv("MAX_SIGNALS_PER_TICK", "1"))

COOLDOWN_MIN_DEFAULT = int(os.getenv("COOLDOWN_MIN", "180"))
COOLDOWN_SEC_DEFAULT = COOLDOWN_MIN_DEFAULT * 60

# quality thresholds (balanced by default)
MIN_ATR_PCT_15M_DEFAULT = float(os.getenv("MIN_ATR_PCT_15M", "0.55"))
MIN_ADX_15M_DEFAULT = float(os.getenv("MIN_ADX_15M", "16"))
MIN_ADX_1H_DEFAULT = float(os.getenv("MIN_ADX_1H", "18"))

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
UNIVERSE: List[str] = []
ROT_IDX: int = 0

# anti-spam
LAST_SENT_TS: Dict[Tuple[str, str], float] = {}   # (symbol, direction) -> ts
LAST_SENT_ENTRY: Dict[Tuple[str, str], float] = {}  # (symbol, direction) -> entry

# per-chat settings (in-memory)
CHAT_CFG: Dict[int, dict] = {}

def cfg(chat_id: int) -> dict:
    if chat_id not in CHAT_CFG:
        CHAT_CFG[chat_id] = {
            "autoscan": True,
            "scan_every_sec": SCAN_EVERY_SEC_DEFAULT,
            "rotation_batch": ROTATION_BATCH_DEFAULT,
            "candidates_top": CANDIDATES_TOP_DEFAULT,
            "max_signals_per_tick": MAX_SIGNALS_PER_TICK_DEFAULT,
            "cooldown_sec": COOLDOWN_SEC_DEFAULT,
            "min_atr_pct_15m": MIN_ATR_PCT_15M_DEFAULT,
            "min_adx_15m": MIN_ADX_15M_DEFAULT,
            "min_adx_1h": MIN_ADX_1H_DEFAULT,
            "mode": "BALANCED",  # STRICT / BALANCED / MORE
        }
    return CHAT_CFG[chat_id]

# -----------------------
# UI
# -----------------------
def kb_main(chat_id: int) -> InlineKeyboardMarkup:
    c = cfg(chat_id)
    autos = "ON ✅" if c["autoscan"] else "OFF ⛔"
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("⚡ Скан сейчас", callback_data="scan_now")],
        [InlineKeyboardButton("📌 Статус", callback_data="status"),
         InlineKeyboardButton("🧺 Universe", callback_data="universe")],
        [InlineKeyboardButton(f"⏯ Автоскан: {autos}", callback_data="toggle_autoscan")],
        [InlineKeyboardButton("⚙️ Настройки", callback_data="settings")],
        [InlineKeyboardButton("❌ Отключить в этом чате", callback_data="off")],
    ])

def kb_settings(chat_id: int) -> InlineKeyboardMarkup:
    c = cfg(chat_id)
    return InlineKeyboardMarkup([
        [InlineKeyboardButton(f"🎯 Режим: {c['mode']}", callback_data="mode")],
        [InlineKeyboardButton(f"⏱ Интервал: {int(c['scan_every_sec']/60)}м", callback_data="interval")],
        [InlineKeyboardButton(f"🧪 Порог ATR%15m: {c['min_atr_pct_15m']}", callback_data="atr")],
        [InlineKeyboardButton("⬅️ Назад", callback_data="back")],
    ])

def kb_pick_mode() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🔥 STRICT (чище, реже)", callback_data="set_mode:STRICT")],
        [InlineKeyboardButton("✅ BALANCED (рекоменд)", callback_data="set_mode:BALANCED")],
        [InlineKeyboardButton("⚡ MORE (чаще)", callback_data="set_mode:MORE")],
        [InlineKeyboardButton("⬅️ Назад", callback_data="settings")],
    ])

def kb_pick_interval(chat_id: int) -> InlineKeyboardMarkup:
    # seconds
    options = [180, 300, 600, 900]  # 3m, 5m, 10m, 15m
    rows = [[InlineKeyboardButton(f"{int(x/60)} мин", callback_data=f"set_interval:{x}") for x in options[:2]],
            [InlineKeyboardButton(f"{int(x/60)} мин", callback_data=f"set_interval:{x}") for x in options[2:]]]
    rows.append([InlineKeyboardButton("⬅️ Назад", callback_data="settings")])
    return InlineKeyboardMarkup(rows)

def kb_pick_atr() -> InlineKeyboardMarkup:
    # min atr% 15m
    options = [0.40, 0.55, 0.70, 0.90]
    rows = [
        [InlineKeyboardButton(f"{options[0]}", callback_data=f"set_atr:{options[0]}"),
         InlineKeyboardButton(f"{options[1]}", callback_data=f"set_atr:{options[1]}")],
        [InlineKeyboardButton(f"{options[2]}", callback_data=f"set_atr:{options[2]}"),
         InlineKeyboardButton(f"{options[3]}", callback_data=f"set_atr:{options[3]}")],
        [InlineKeyboardButton("⬅️ Назад", callback_data="settings")],
    ]
    return InlineKeyboardMarkup(rows)

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
    markets = await ex.load_markets()
    out = []
    for sym, m in markets.items():
        try:
            if not m.get("swap"):
                continue
            if m.get("linear") is not True:
                continue
            if (m.get("quote") or "") != "USDT":
                continue
            if m.get("future"):
                continue
            out.append(sym)
        except Exception:
            continue
    return sorted(set(out))

# -----------------------
# STRATEGY
# -----------------------
def trend_bias(df_4h: pd.DataFrame, df_1h: pd.DataFrame, min_adx_1h: float) -> Optional[str]:
    c4 = df_4h["close"]
    e200_4 = ema(c4, EMA_SLOW).iloc[-1]
    close4 = c4.iloc[-1]
    atr4 = atr(df_4h, ATR_LEN).iloc[-1]
    if pd.isna(e200_4) or pd.isna(atr4):
        return None
    if abs(close4 - e200_4) < 0.30 * atr4:
        return None
    regime = "LONG" if close4 > e200_4 else "SHORT" if close4 < e200_4 else None
    if regime is None:
        return None

    c1 = df_1h["close"]
    e50_1 = ema(c1, EMA_FAST).iloc[-1]
    e200_1 = ema(c1, EMA_SLOW).iloc[-1]
    adx1 = adx(df_1h, ADX_LEN).iloc[-1]
    if pd.isna(e50_1) or pd.isna(e200_1) or pd.isna(adx1):
        return None
    if float(adx1) < float(min_adx_1h):
        return None

    if regime == "LONG" and (e50_1 > e200_1):
        return "LONG"
    if regime == "SHORT" and (e50_1 < e200_1):
        return "SHORT"
    return None

def level_from_15m(df_15m: pd.DataFrame, bias: str, min_atr_pct_15m: float, min_adx_15m: float) -> Optional[float]:
    if len(df_15m) < LEVEL_LOOKBACK_15M + 5:
        return None

    df = df_15m.copy()
    df["atr"] = atr(df, ATR_LEN)
    df["adx"] = adx(df, ADX_LEN)
    df["rsi"] = rsi(df["close"], RSI_LEN)

    last = df.iloc[-1]
    atr15 = last["atr"]
    if pd.isna(atr15) or float(atr15) <= 0:
        return None

    atr_pct = float(atr15 / last["close"] * 100.0)
    if atr_pct < float(min_atr_pct_15m):
        return None

    if pd.isna(last["adx"]) or float(last["adx"]) < float(min_adx_15m):
        return None

    body = abs(float(last["close"] - last["open"]))
    if body < 0.35 * float(atr15):
        return None

    window = df.iloc[-(LEVEL_LOOKBACK_15M + 1):-1]
    if bias == "LONG":
        level = float(window["high"].max())
        if float(last["rsi"]) > 78:
            return None
        return level
    else:
        level = float(window["low"].min())
        if float(last["rsi"]) < 22:
            return None
        return level

def check_break_and_retest_5m(df_5m: pd.DataFrame, bias: str, level: float) -> Optional[Tuple[float, float, float, float]]:
    if len(df_5m) < 120:
        return None

    df = df_5m.copy()
    df["atr"] = atr(df, ATR_LEN)
    df["ema20"] = ema(df["close"], 20)

    atr5 = float(df["atr"].iloc[-1]) if pd.notna(df["atr"].iloc[-1]) else 0.0
    if atr5 <= 0:
        return None
    ema20_last = float(df["ema20"].iloc[-1])

    bars = df.iloc[-(RETEST_MAX_BARS_5M + 8):].reset_index(drop=True)
    if len(bars) < RETEST_MAX_BARS_5M + 3:
        return None

    level_off = BREAK_ATR_K * atr5
    retest_off = RETEST_ATR_K * atr5

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

    # retest + rebound
    for j in range(breakout_idx + 1, min(breakout_idx + 1 + RETEST_MAX_BARS_5M, len(bars))):
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

def cooldown_ok(chat_id: int, symbol: str, direction: str) -> bool:
    c = cfg(chat_id)
    t = LAST_SENT_TS.get((symbol, direction), 0.0)
    return (time.time() - t) >= float(c["cooldown_sec"])

def anti_duplicate(symbol: str, direction: str, entry: float) -> bool:
    prev = LAST_SENT_ENTRY.get((symbol, direction))
    if prev is None:
        return True
    # if entry is within 0.2% => treat as duplicate
    return abs(prev - entry) / max(entry, 1e-9) > 0.002

# -----------------------
# SCAN PIPELINE
# -----------------------
async def prefilter_symbol(ex, symbol: str, min_atr_pct_15m: float) -> Optional[Tuple[str, float]]:
    df15 = await fetch_df(ex, symbol, "15m", 220)
    if df15 is None:
        return None
    df15["atr"] = atr(df15, ATR_LEN)
    atr15 = df15["atr"].iloc[-1]
    if pd.isna(atr15) or float(atr15) <= 0:
        return None
    close = float(df15["close"].iloc[-1])
    atr_pct = float(atr15 / close * 100.0)
    if atr_pct < float(min_atr_pct_15m):
        return None
    vol = float(df15["volume"].iloc[-96:].sum())
    score = atr_pct * 2.0 + np.log10(max(1.0, vol)) * 0.8
    return (symbol, float(score))

async def deep_analyze(ex, symbol: str, c: dict) -> Optional[Signal]:
    df4 = await fetch_df(ex, symbol, "4h", 260)
    df1 = await fetch_df(ex, symbol, "1h", 260)
    df15 = await fetch_df(ex, symbol, "15m", 260)
    df5 = await fetch_df(ex, symbol, "5m", 260)
    if any(x is None for x in (df4, df1, df15, df5)):
        return None

    bias = trend_bias(df4, df1, c["min_adx_1h"])
    if bias is None:
        return None

    level = level_from_15m(df15, bias, c["min_atr_pct_15m"], c["min_adx_15m"])
    if level is None:
        return None

    res = check_break_and_retest_5m(df5, bias, level)
    if res is None:
        return None

    entry, sl, tp1, tp2 = res

    return Signal(symbol=symbol, direction=bias, entry=float(entry), sl=float(sl), tp1=float(tp1), tp2=float(tp2), level=float(level))

async def scan_tick(app: Application, forced_chat_id: Optional[int] = None):
    """
    If forced_chat_id is provided -> uses that chat settings and sends results only there.
    Otherwise -> sends to all subscribers using their own settings, but scanning happens once
    using a "global" balanced config to save API calls.
    """
    global ROT_IDX
    ex = app.bot_data.get("exchange")
    if ex is None or not UNIVERSE:
        return

    # choose a global scan config (use default-ish) to avoid doing different scans per chat
    # (per-chat thresholds still apply at send stage)
    global_cfg = {
        "rotation_batch": ROTATION_BATCH_DEFAULT,
        "candidates_top": CANDIDATES_TOP_DEFAULT,
        "min_atr_pct_15m": MIN_ATR_PCT_15M_DEFAULT,
        "min_adx_15m": MIN_ADX_15M_DEFAULT,
        "min_adx_1h": MIN_ADX_1H_DEFAULT,
    }

    n = len(UNIVERSE)
    start = ROT_IDX % n
    end = min(n, start + int(global_cfg["rotation_batch"]))
    batch = UNIVERSE[start:end]
    ROT_IDX = end if end < n else 0

    pre: List[Tuple[str, float]] = []

    async def pf(sym: str):
        r = await prefilter_symbol(ex, sym, global_cfg["min_atr_pct_15m"])
        if r:
            pre.append(r)

    await asyncio.gather(*[pf(s) for s in batch])
    if not pre:
        return

    pre.sort(key=lambda x: x[1], reverse=True)
    candidates = [s for s, _ in pre[: int(global_cfg["candidates_top"])]]

    found: List[Signal] = []

    async def da(sym: str):
        sig = await deep_analyze(ex, sym, global_cfg)
        if sig:
            found.append(sig)

    await asyncio.gather(*[da(s) for s in candidates])

    if not found:
        return

    # send top signals (global limit)
    found = found[:MAX_SIGNALS_PER_TICK_DEFAULT]

    targets = [forced_chat_id] if forced_chat_id else list(SUBSCRIBERS)

    for sig in found:
        for chat_id in targets:
            if chat_id is None:
                continue
            c = cfg(chat_id)
            if not cooldown_ok(chat_id, sig.symbol, sig.direction):
                continue
            if not anti_duplicate(sig.symbol, sig.direction, sig.entry):
                continue

            LAST_SENT_TS[(sig.symbol, sig.direction)] = time.time()
            LAST_SENT_ENTRY[(sig.symbol, sig.direction)] = sig.entry

            text = (
                f"🚨 <b>{sig.symbol}</b> — <b>{sig.direction}</b>\n\n"
                f"TF: <b>4h/1h фильтр</b> | <b>15m уровень</b> | <b>5m вход</b>\n"
                f"Уровень (15m): <code>{fmt_price(sig.level)}</code>\n\n"
                f"Entry: <code>{fmt_price(sig.entry)}</code>\n"
                f"SL: <code>{fmt_price(sig.sl)}</code>\n"
                f"TP1: <code>{fmt_price(sig.tp1)}</code>\n"
                f"TP2: <code>{fmt_price(sig.tp2)}</code>\n\n"
                f"Логика: подтверждение пробоя + ретест (5m)\n"
                f"⚠️ Не финсовет."
            )
            try:
                await app.bot.send_message(chat_id=chat_id, text=text, parse_mode="HTML")
            except Exception:
                pass

# -----------------------
# LOOP
# -----------------------
async def scanner_loop(app: Application):
    await asyncio.sleep(3)
    while True:
        try:
            # if no subscribers - just sleep
            if not SUBSCRIBERS:
                await asyncio.sleep(30)
                continue

            # autoscan is per chat; if at least one chat has autoscan ON, run tick
            any_on = any(cfg(cid).get("autoscan", True) for cid in SUBSCRIBERS)
            if any_on:
                await scan_tick(app)
        except Exception as e:
            log.warning("scanner_loop error: %s", e)

        # choose smallest interval among active chats
        if SUBSCRIBERS:
            intervals = [int(cfg(cid)["scan_every_sec"]) for cid in SUBSCRIBERS if cfg(cid).get("autoscan", True)]
            sleep_s = min(intervals) if intervals else 60
        else:
            sleep_s = 60

        await asyncio.sleep(max(30, sleep_s))

# -----------------------
# TELEGRAM HANDLERS
# -----------------------
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    SUBSCRIBERS.add(chat_id)
    cfg(chat_id)  # init
    await update.message.reply_text(
        "✅ Готово. Я включён.\n\n"
        "Управление — кнопками ниже.",
        reply_markup=kb_main(chat_id)
    )

async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    c = cfg(chat_id)
    await update.message.reply_text(
        f"Статус ✅\n"
        f"Exchange: {EXCHANGE_NAME}\n"
        f"Universe: {len(UNIVERSE)}\n"
        f"Автоскан: {'ON' if c['autoscan'] else 'OFF'}\n"
        f"Интервал: {int(c['scan_every_sec']/60)} мин\n"
        f"Mode: {c['mode']}\n"
        f"ATR%15m >= {c['min_atr_pct_15m']}\n"
        f"ADX15m >= {c['min_adx_15m']}, ADX1h >= {c['min_adx_1h']}\n"
        f"Cooldown: {int(c['cooldown_sec']/60)} мин",
        reply_markup=kb_main(chat_id)
    )

async def cmd_now(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    SUBSCRIBERS.add(chat_id)
    cfg(chat_id)
    await update.message.reply_text("⏳ Сканирую сейчас…")
    await scan_tick(context.application, forced_chat_id=chat_id)
    await update.message.reply_text("✅ Готово. Если сетап был — я отправил сигнал.", reply_markup=kb_main(chat_id))

async def cmd_off(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    SUBSCRIBERS.discard(chat_id)
    await update.message.reply_text("⛔ Отключил для этого чата. Чтобы включить снова — /start")

async def cmd_universe(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    await update.message.reply_text(f"USDT perpetual (linear swap): {len(UNIVERSE)}", reply_markup=kb_main(chat_id))

async def cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    chat_id = q.message.chat_id
    c = cfg(chat_id)

    data = q.data

    if data == "back":
        await q.message.edit_text("Меню:", reply_markup=kb_main(chat_id))
        return

    if data == "status":
        await q.message.edit_text(
            f"📌 <b>Status</b>\n\n"
            f"Exchange: <code>{EXCHANGE_NAME}</code>\n"
            f"Universe: <code>{len(UNIVERSE)}</code>\n"
            f"Автоскан: <b>{'ON' if c['autoscan'] else 'OFF'}</b>\n"
            f"Интервал: <b>{int(c['scan_every_sec']/60)} мин</b>\n"
            f"Mode: <b>{c['mode']}</b>\n"
            f"ATR%15m >= <b>{c['min_atr_pct_15m']}</b>\n"
            f"ADX15m >= <b>{c['min_adx_15m']}</b>, ADX1h >= <b>{c['min_adx_1h']}</b>\n"
            f"Cooldown: <b>{int(c['cooldown_sec']/60)} мин</b>",
            parse_mode="HTML",
            reply_markup=kb_main(chat_id)
        )
        return

    if data == "universe":
        await q.message.edit_text(f"🧺 Universe: <b>{len(UNIVERSE)}</b> USDT perpetual (linear).", parse_mode="HTML", reply_markup=kb_main(chat_id))
        return

    if data == "scan_now":
        await q.message.reply_text("⏳ Сканирую сейчас…")
        await scan_tick(context.application, forced_chat_id=chat_id)
        await q.message.reply_text("✅ Готово.", reply_markup=kb_main(chat_id))
        return

    if data == "toggle_autoscan":
        c["autoscan"] = not bool(c["autoscan"])
        await q.message.edit_text("Меню:", reply_markup=kb_main(chat_id))
        return

    if data == "settings":
        await q.message.edit_text("⚙️ <b>Настройки</b>", parse_mode="HTML", reply_markup=kb_settings(chat_id))
        return

    if data == "mode":
        await q.message.edit_text("🎯 Выбери режим:", reply_markup=kb_pick_mode())
        return

    if data.startswith("set_mode:"):
        mode = data.split(":", 1)[1]
        c["mode"] = mode

        # Apply presets
        if mode == "STRICT":
            c["min_atr_pct_15m"] = 0.70
            c["min_adx_1h"] = 20
            c["min_adx_15m"] = 18
            c["cooldown_sec"] = 240 * 60
            c["scan_every_sec"] = max(c["scan_every_sec"], 600)
        elif mode == "MORE":
            c["min_atr_pct_15m"] = 0.40
            c["min_adx_1h"] = 16
            c["min_adx_15m"] = 14
            c["cooldown_sec"] = 120 * 60
            c["scan_every_sec"] = min(c["scan_every_sec"], 300)
        else:  # BALANCED
            c["min_atr_pct_15m"] = 0.55
            c["min_adx_1h"] = 18
            c["min_adx_15m"] = 16
            c["cooldown_sec"] = 180 * 60

        await q.message.edit_text("✅ Режим применён.", reply_markup=kb_settings(chat_id))
        return

    if data == "interval":
        await q.message.edit_text("⏱ Выбери интервал:", reply_markup=kb_pick_interval(chat_id))
        return

    if data.startswith("set_interval:"):
        sec = int(data.split(":", 1)[1])
        c["scan_every_sec"] = sec
        await q.message.edit_text("✅ Интервал обновлён.", reply_markup=kb_settings(chat_id))
        return

    if data == "atr":
        await q.message.edit_text("🧪 Порог ATR%15m:", reply_markup=kb_pick_atr())
        return

    if data.startswith("set_atr:"):
        val = float(data.split(":", 1)[1])
        c["min_atr_pct_15m"] = val
        await q.message.edit_text("✅ ATR% порог обновлён.", reply_markup=kb_settings(chat_id))
        return

    if data == "off":
        SUBSCRIBERS.discard(chat_id)
        await q.message.edit_text("⛔ Отключил для этого чата. Чтобы включить снова — /start")
        return

# -----------------------
# STARTUP / SHUTDOWN
# -----------------------
async def on_startup(app: Application):
    ex = await get_exchange()
    app.bot_data["exchange"] = ex

    global UNIVERSE
    UNIVERSE = await build_universe(ex)

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
    app.add_handler(CommandHandler("now", cmd_now))
    app.add_handler(CommandHandler("universe", cmd_universe))
    app.add_handler(CommandHandler("off", cmd_off))
    app.add_handler(CallbackQueryHandler(cb))

    app.post_init = on_startup
    app.post_shutdown = on_shutdown

    log.info("BOT STARTING...")
    app.run_polling(allowed_updates=Update.ALL_TYPES, close_loop=False)

if __name__ == "__main__":
    main()
