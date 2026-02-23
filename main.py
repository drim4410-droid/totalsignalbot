import os
import asyncio
import logging
import time
import math
import sqlite3
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv

import ccxt.async_support as ccxt

from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
)
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    ContextTypes,
)

# =========================
# VERSION
# =========================
BOT_VERSION = "BingX-TOP200-PRO-v1-2026-02-23"

# =========================
# LOGGING
# =========================
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("bingx_top200_pro")

# =========================
# ENV
# =========================
load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN env var not set")

OWNER_ID = os.getenv("OWNER_ID", "").strip()
if not OWNER_ID.isdigit():
    raise RuntimeError("OWNER_ID env var not set or not a number")
OWNER_ID = int(OWNER_ID)

EXCHANGE_NAME = os.getenv("EXCHANGE", "bingx").strip().lower()

SCAN_EVERY_SEC = int(os.getenv("SCAN_EVERY_SEC", "300"))              # autoscan tick
ROTATION_BATCH = int(os.getenv("ROTATION_BATCH", "120"))              # per tick symbols prefilter
TOP200_REFRESH_SEC = int(os.getenv("TOP200_REFRESH_SEC", "21600"))    # 6h
SIGNAL_COOLDOWN_MIN = int(os.getenv("SIGNAL_COOLDOWN_MIN", "180"))    # per symbol+direction cooldown
SIGNAL_COOLDOWN_SEC = SIGNAL_COOLDOWN_MIN * 60

# =========================
# Strategy params (balanced by default)
# =========================
EMA_FAST = 50
EMA_SLOW = 200
ATR_LEN = 14
ADX_LEN = 14
RSI_LEN = 14

LEVEL_LOOKBACK_15M = 48        # 12h on 15m
BREAK_ATR_K = 0.10
RETEST_ATR_K = 0.25
RETEST_MAX_BARS_5M = 6

SL_ATR_K = 1.25
TP1_ATR_K = 1.0
TP2_ATR_K = 2.2

# Thresholds per mode
MODES = {
    "BALANCED": {
        "min_atr_pct_15m": 0.55,
        "min_adx_15m": 16.0,
        "min_adx_1h": 18.0,
        "max_signals_per_tick": 1,
    },
    "STRICT": {
        "min_atr_pct_15m": 0.75,
        "min_adx_15m": 18.0,
        "min_adx_1h": 20.0,
        "max_signals_per_tick": 1,
    },
    "MORE": {
        "min_atr_pct_15m": 0.40,
        "min_adx_15m": 14.0,
        "min_adx_1h": 16.0,
        "max_signals_per_tick": 2,
    },
}

# =========================
# Concurrency
# =========================
MAX_CONCURRENCY = 6
SEM = asyncio.Semaphore(MAX_CONCURRENCY)

# =========================
# SQLite storage
# =========================
DB_PATH = os.getenv("DB_PATH", "bot.db")

def db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn

def init_db():
    conn = db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users(
            chat_id INTEGER PRIMARY KEY,
            requested_at INTEGER,
            approved_until INTEGER,
            approved_by INTEGER,
            status TEXT
        );
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS settings(
            key TEXT PRIMARY KEY,
            value TEXT
        );
    """)
    conn.commit()
    conn.close()

def user_get(chat_id: int) -> Optional[dict]:
    conn = db()
    cur = conn.execute("SELECT chat_id, requested_at, approved_until, approved_by, status FROM users WHERE chat_id=?", (chat_id,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    return {
        "chat_id": row[0],
        "requested_at": row[1],
        "approved_until": row[2],
        "approved_by": row[3],
        "status": row[4],
    }

def user_upsert_request(chat_id: int):
    now = int(time.time())
    conn = db()
    conn.execute("""
        INSERT INTO users(chat_id, requested_at, approved_until, approved_by, status)
        VALUES(?, ?, NULL, NULL, 'PENDING')
        ON CONFLICT(chat_id) DO UPDATE SET requested_at=excluded.requested_at, status='PENDING';
    """, (chat_id, now))
    conn.commit()
    conn.close()

def user_set_approved(chat_id: int, days: int, approved_by: int):
    until = int(time.time()) + days * 24 * 3600
    conn = db()
    conn.execute("""
        INSERT INTO users(chat_id, requested_at, approved_until, approved_by, status)
        VALUES(?, NULL, ?, ?, 'APPROVED')
        ON CONFLICT(chat_id) DO UPDATE SET approved_until=excluded.approved_until, approved_by=excluded.approved_by, status='APPROVED';
    """, (chat_id, until, approved_by))
    conn.commit()
    conn.close()

def user_set_denied(chat_id: int, denied_by: int):
    conn = db()
    conn.execute("""
        INSERT INTO users(chat_id, requested_at, approved_until, approved_by, status)
        VALUES(?, NULL, NULL, ?, 'DENIED')
        ON CONFLICT(chat_id) DO UPDATE SET approved_until=NULL, approved_by=excluded.approved_by, status='DENIED';
    """, (chat_id, denied_by))
    conn.commit()
    conn.close()

def approved_chat_ids() -> List[int]:
    now = int(time.time())
    conn = db()
    cur = conn.execute("""
        SELECT chat_id FROM users
        WHERE status='APPROVED' AND approved_until IS NOT NULL AND approved_until > ?;
    """, (now,))
    rows = [r[0] for r in cur.fetchall()]
    conn.close()
    return rows

def cleanup_expired():
    now = int(time.time())
    conn = db()
    conn.execute("""
        UPDATE users SET status='EXPIRED'
        WHERE status='APPROVED' AND approved_until IS NOT NULL AND approved_until <= ?;
    """, (now,))
    conn.commit()
    conn.close()

def get_global_setting(key: str, default: str) -> str:
    conn = db()
    cur = conn.execute("SELECT value FROM settings WHERE key=?", (key,))
    row = cur.fetchone()
    conn.close()
    return row[0] if row else default

def set_global_setting(key: str, value: str):
    conn = db()
    conn.execute("""
        INSERT INTO settings(key, value) VALUES(?, ?)
        ON CONFLICT(key) DO UPDATE SET value=excluded.value;
    """, (key, value))
    conn.commit()
    conn.close()

# =========================
# Global runtime state
# =========================
UNIVERSE_ALL: List[str] = []     # all usdt linear perpetual
TOP200: List[str] = []           # by liquidity
TOP20: List[str] = []            # top 20 slice
ROT_IDX = 0

LAST_SENT_TS: Dict[Tuple[str, str], float] = {}   # (symbol, direction) -> ts
LAST_SENT_ENTRY: Dict[Tuple[str, str], float] = {}

def now_ts() -> float:
    return time.time()

# =========================
# UI
# =========================
def kb_start(chat_id: int) -> InlineKeyboardMarkup:
    u = user_get(chat_id)
    if chat_id == OWNER_ID:
        return kb_main(chat_id)
    if u and u["status"] == "APPROVED" and (u["approved_until"] or 0) > int(time.time()):
        return kb_main(chat_id)
    if u and u["status"] == "PENDING":
        return InlineKeyboardMarkup([
            [InlineKeyboardButton("⏳ Запрос отправлен (ожидай)", callback_data="noop")],
            [InlineKeyboardButton("🔁 Отправить запрос снова", callback_data="request_access")],
        ])
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("✅ Запросить доступ", callback_data="request_access")],
    ])

def kb_main(chat_id: int) -> InlineKeyboardMarkup:
    mode = get_global_setting("MODE", "BALANCED")
    autoscan = get_global_setting("AUTOSCAN", "ON")
    autos_txt = "ON ✅" if autoscan == "ON" else "OFF ⛔"
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🧠 Новый сигнал (Top20)", callback_data="new_signal")],
        [InlineKeyboardButton("⏯ Автоскан: " + autos_txt, callback_data="toggle_autoscan")],
        [InlineKeyboardButton("📌 Статус", callback_data="status"),
         InlineKeyboardButton("⚙️ Настройки", callback_data="settings")],
        [InlineKeyboardButton("🧺 Топ-200 список", callback_data="show_top200")],
        [InlineKeyboardButton("🧩 Версия", callback_data="version")],
    ])

def kb_settings(chat_id: int) -> InlineKeyboardMarkup:
    mode = get_global_setting("MODE", "BALANCED")
    return InlineKeyboardMarkup([
        [InlineKeyboardButton(f"🎯 Режим: {mode}", callback_data="pick_mode")],
        [InlineKeyboardButton("⏱ Скан сейчас (принуд.)", callback_data="scan_now")],
        [InlineKeyboardButton("⬅️ Назад", callback_data="back_main")],
    ])

def kb_pick_mode() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🔥 STRICT (чище, реже)", callback_data="set_mode:STRICT")],
        [InlineKeyboardButton("✅ BALANCED (рекоменд)", callback_data="set_mode:BALANCED")],
        [InlineKeyboardButton("⚡ MORE (чаще)", callback_data="set_mode:MORE")],
        [InlineKeyboardButton("⬅️ Назад", callback_data="settings")],
    ])

def kb_owner_approve(chat_id: int) -> InlineKeyboardMarkup:
    # OWNER only
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("✅ 7 дней", callback_data=f"approve:{chat_id}:7"),
            InlineKeyboardButton("✅ 15 дней", callback_data=f"approve:{chat_id}:15"),
            InlineKeyboardButton("✅ 30 дней", callback_data=f"approve:{chat_id}:30"),
        ],
        [InlineKeyboardButton("⛔ Deny", callback_data=f"deny:{chat_id}")],
    ])

# =========================
# Permissions
# =========================
def is_allowed(chat_id: int) -> bool:
    if chat_id == OWNER_ID:
        return True
    u = user_get(chat_id)
    if not u:
        return False
    if u["status"] != "APPROVED":
        return False
    return (u["approved_until"] or 0) > int(time.time())

# =========================
# Indicators
# =========================
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
    return true_range(df).rolling(length).mean()

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

# =========================
# Exchange data
# =========================
def parse_ohlcv(ohlcv: List[List[float]]) -> pd.DataFrame:
    df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna()

async def fetch_df(ex, symbol: str, tf: str, limit: int = 260) -> Optional[pd.DataFrame]:
    async with SEM:
        try:
            ohlcv = await ex.fetch_ohlcv(symbol, timeframe=tf, limit=limit)
            if not ohlcv or len(ohlcv) < 120:
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

async def build_universe_usdt_linear_swaps(ex) -> List[str]:
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

async def rebuild_top200(ex):
    global UNIVERSE_ALL, TOP200, TOP20

    if not UNIVERSE_ALL:
        UNIVERSE_ALL = await build_universe_usdt_linear_swaps(ex)

    # fetch tickers once; if exchange doesn't support fetch_tickers well, fallback to smaller
    ranked: List[Tuple[str, float]] = []
    try:
        tickers = await ex.fetch_tickers(UNIVERSE_ALL)
        for sym in UNIVERSE_ALL:
            t = tickers.get(sym) or {}
            # use quoteVolume first, then baseVolume
            qv = t.get("quoteVolume")
            bv = t.get("baseVolume")
            last = t.get("last") or t.get("close")
            if last is None:
                continue
            vol = None
            if isinstance(qv, (int, float)) and qv > 0:
                vol = float(qv)
            elif isinstance(bv, (int, float)) and bv > 0 and isinstance(last, (int, float)) and last > 0:
                vol = float(bv) * float(last)
            if vol is None:
                continue
            ranked.append((sym, vol))
    except Exception as e:
        log.warning("fetch_tickers failed, fallback ranking: %s", e)
        # fallback: use last 24h volume proxy from 15m candles (slower) on a rotation subset
        ranked = []
        sample = UNIVERSE_ALL[:500]
        async def one(sym: str):
            df = await fetch_df(ex, sym, "15m", 120)
            if df is None:
                return
            vol = float(df["volume"].iloc[-96:].sum())
            ranked.append((sym, vol))
        await asyncio.gather(*[one(s) for s in sample])

    ranked.sort(key=lambda x: x[1], reverse=True)
    TOP200 = [s for s, _ in ranked[:200]]
    TOP20 = TOP200[:20]
    log.info("Top200 rebuilt: %s | Top20: %s", len(TOP200), ", ".join(TOP20[:5]))

# =========================
# Strategy (MTF with breakout+retest)
# =========================
@dataclass
class Signal:
    symbol: str
    direction: str
    entry: float
    sl: float
    tp1: float
    tp2: float
    level: float
    score: float
    reason: str

def trend_bias(df_4h: pd.DataFrame, df_1h: pd.DataFrame, min_adx_1h: float) -> Optional[str]:
    c4 = df_4h["close"]
    e200_4 = ema(c4, EMA_SLOW).iloc[-1]
    close4 = float(c4.iloc[-1])
    atr4 = float(atr(df_4h, ATR_LEN).iloc[-1]) if pd.notna(atr(df_4h, ATR_LEN).iloc[-1]) else math.nan
    if math.isnan(float(e200_4)) or math.isnan(atr4):
        return None

    # avoid chop near EMA200 on 4h
    if abs(close4 - float(e200_4)) < 0.30 * atr4:
        return None

    regime = "LONG" if close4 > float(e200_4) else "SHORT" if close4 < float(e200_4) else None
    if not regime:
        return None

    c1 = df_1h["close"]
    e50_1 = ema(c1, EMA_FAST).iloc[-1]
    e200_1 = ema(c1, EMA_SLOW).iloc[-1]
    adx1 = adx(df_1h, ADX_LEN).iloc[-1]
    if pd.isna(e50_1) or pd.isna(e200_1) or pd.isna(adx1):
        return None
    if float(adx1) < float(min_adx_1h):
        return None

    if regime == "LONG" and float(e50_1) > float(e200_1):
        return "LONG"
    if regime == "SHORT" and float(e50_1) < float(e200_1):
        return "SHORT"
    return None

def level_from_15m(df_15m: pd.DataFrame, bias: str, min_atr_pct_15m: float, min_adx_15m: float) -> Optional[Tuple[float, float, float, float]]:
    if len(df_15m) < LEVEL_LOOKBACK_15M + 10:
        return None

    df = df_15m.copy()
    df["atr"] = atr(df, ATR_LEN)
    df["adx"] = adx(df, ADX_LEN)
    df["rsi"] = rsi(df["close"], RSI_LEN)

    last = df.iloc[-1]
    atr15 = float(last["atr"]) if pd.notna(last["atr"]) else 0.0
    if atr15 <= 0:
        return None

    close = float(last["close"])
    atr_pct = (atr15 / close) * 100.0
    if atr_pct < float(min_atr_pct_15m):
        return None
    if pd.isna(last["adx"]) or float(last["adx"]) < float(min_adx_15m):
        return None

    body = abs(float(last["close"] - last["open"]))
    if body < 0.35 * atr15:
        return None

    window = df.iloc[-(LEVEL_LOOKBACK_15M + 1):-1]
    if bias == "LONG":
        if float(last["rsi"]) > 78:
            return None
        level = float(window["high"].max())
        return (level, atr15, atr_pct, float(last["adx"]))
    else:
        if float(last["rsi"]) < 22:
            return None
        level = float(window["low"].min())
        return (level, atr15, atr_pct, float(last["adx"]))

def check_break_and_retest_5m(df_5m: pd.DataFrame, bias: str, level: float) -> Optional[Tuple[float, float, float, float, float]]:
    if len(df_5m) < 140:
        return None

    df = df_5m.copy()
    df["atr"] = atr(df, ATR_LEN)
    df["ema20"] = ema(df["close"], 20)

    atr5 = float(df["atr"].iloc[-1]) if pd.notna(df["atr"].iloc[-1]) else 0.0
    if atr5 <= 0:
        return None
    ema20_last = float(df["ema20"].iloc[-1])

    bars = df.iloc[-(RETEST_MAX_BARS_5M + 10):].reset_index(drop=True)
    if len(bars) < RETEST_MAX_BARS_5M + 4:
        return None

    level_off = BREAK_ATR_K * atr5
    retest_off = RETEST_ATR_K * atr5

    breakout_idx = None
    for i in range(len(bars)):
        c = float(bars["close"].iloc[i])
        if bias == "LONG" and c > level + level_off:
            breakout_idx = i
            break
        if bias == "SHORT" and c < level - level_off:
            breakout_idx = i
            break
    if breakout_idx is None:
        return None

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
                return (entry, sl, tp1, tp2, atr5)
        else:
            touched = (h >= level - retest_off)
            closed_ok = (c < level)
            rebound = (c < o) and (body >= 0.25 * atr5) and (c < ema20_last)
            if touched and closed_ok and rebound:
                entry = c
                sl = entry + SL_ATR_K * atr5
                tp1 = entry - TP1_ATR_K * atr5
                tp2 = entry - TP2_ATR_K * atr5
                return (entry, sl, tp1, tp2, atr5)

    return None

def cooldown_ok(symbol: str, direction: str, entry: float) -> bool:
    t = LAST_SENT_TS.get((symbol, direction), 0.0)
    if (now_ts() - t) < SIGNAL_COOLDOWN_SEC:
        return False
    prev = LAST_SENT_ENTRY.get((symbol, direction))
    if prev is not None and abs(prev - entry) / max(entry, 1e-9) <= 0.002:
        return False
    return True

async def analyze_symbol_strong(ex, symbol: str, mode: str) -> Optional[Signal]:
    p = MODES.get(mode, MODES["BALANCED"])
    df4 = await fetch_df(ex, symbol, "4h", 260)
    df1 = await fetch_df(ex, symbol, "1h", 260)
    df15 = await fetch_df(ex, symbol, "15m", 260)
    df5 = await fetch_df(ex, symbol, "5m", 260)
    if any(x is None for x in (df4, df1, df15, df5)):
        return None

    bias = trend_bias(df4, df1, p["min_adx_1h"])
    if not bias:
        return None

    lvl_pack = level_from_15m(df15, bias, p["min_atr_pct_15m"], p["min_adx_15m"])
    if not lvl_pack:
        return None
    level, atr15, atr_pct_15, adx15 = lvl_pack

    br = check_break_and_retest_5m(df5, bias, level)
    if not br:
        return None
    entry, sl, tp1, tp2, atr5 = br

    if not cooldown_ok(symbol, bias, entry):
        return None

    # Score (to pick best for Top20 “New Signal”)
    # Higher: more volatility, stronger adx, cleaner RR distance, and not too extended from level
    ext = abs(entry - level) / max(atr5, 1e-9)  # in ATRs
    score = 0.0
    score += min(35.0, atr_pct_15 * 20.0)              # volatility weight
    score += min(25.0, max(0.0, (adx15 - 14.0) * 1.5)) # trendiness
    score += 20.0                                      # MTF alignment baseline
    score += max(0.0, 12.0 - ext * 6.0)                # penalize too extended entries
    score = max(0.0, min(100.0, score))

    reason = (
        f"MTF: 4h/1h согласованы ({bias}) | 15m ATR%={atr_pct_15:.2f} | ADX15={adx15:.1f} | "
        f"Вход: подтверждение+ретест (5m)"
    )

    return Signal(symbol, bias, entry, sl, tp1, tp2, level, score, reason)

def format_signal(sig: Signal) -> str:
    return (
        f"🚨 <b>{sig.symbol}</b> — <b>{sig.direction}</b>\n"
        f"🧠 Score: <b>{sig.score:.0f}%</b>\n"
        f"🧩 <code>{BOT_VERSION}</code>\n\n"
        f"Уровень (15m): <code>{fmt_price(sig.level)}</code>\n"
        f"Entry: <code>{fmt_price(sig.entry)}</code>\n"
        f"SL: <code>{fmt_price(sig.sl)}</code>\n"
        f"TP1: <code>{fmt_price(sig.tp1)}</code>\n"
        f"TP2: <code>{fmt_price(sig.tp2)}</code>\n\n"
        f"Причина: {sig.reason}\n"
        f"⚠️ Не финсовет."
    )

# =========================
# Scanning logic
# =========================
async def prefilter_symbol(ex, symbol: str) -> Optional[Tuple[str, float]]:
    # Cheap filter via 15m ATR% + 24h volume proxy (from candles)
    df15 = await fetch_df(ex, symbol, "15m", 140)
    if df15 is None:
        return None
    df15["atr"] = atr(df15, ATR_LEN)
    atr15 = df15["atr"].iloc[-1]
    if pd.isna(atr15) or float(atr15) <= 0:
        return None
    close = float(df15["close"].iloc[-1])
    atr_pct = float(atr15) / close * 100.0
    vol24 = float(df15["volume"].iloc[-96:].sum()) if len(df15) >= 96 else float(df15["volume"].sum())
    score = atr_pct * 2.0 + math.log10(max(1.0, vol24)) * 0.8
    return (symbol, score)

async def scan_rotation_find_signal(app: Application) -> Optional[Signal]:
    global ROT_IDX
    ex = app.bot_data.get("exchange")
    if ex is None or not TOP200:
        return None

    mode = get_global_setting("MODE", "BALANCED")

    n = len(TOP200)
    start = ROT_IDX % n
    end = min(n, start + ROTATION_BATCH)
    batch = TOP200[start:end]
    ROT_IDX = end if end < n else 0

    # prefilter
    pre: List[Tuple[str, float]] = []
    async def pf(sym: str):
        r = await prefilter_symbol(ex, sym)
        if r:
            pre.append(r)

    await asyncio.gather(*[pf(s) for s in batch])
    if not pre:
        return None

    pre.sort(key=lambda x: x[1], reverse=True)
    candidates = [s for s, _ in pre[: min(25, len(pre))]]

    # deep analyze candidates, pick best
    found: List[Signal] = []

    async def da(sym: str):
        sig = await analyze_symbol_strong(ex, sym, mode)
        if sig:
            found.append(sig)

    await asyncio.gather(*[da(s) for s in candidates])
    if not found:
        return None

    found.sort(key=lambda s: s.score, reverse=True)
    return found[0]

async def new_signal_top20(app: Application) -> Optional[Signal]:
    ex = app.bot_data.get("exchange")
    if ex is None or not TOP20:
        return None
    mode = get_global_setting("MODE", "BALANCED")

    found: List[Signal] = []
    async def one(sym: str):
        sig = await analyze_symbol_strong(ex, sym, mode)
        if sig:
            found.append(sig)

    await asyncio.gather(*[one(s) for s in TOP20])
    if not found:
        return None

    found.sort(key=lambda s: s.score, reverse=True)
    return found[0]

# =========================
# Background tasks
# =========================
async def autoscan_loop(app: Application):
    await asyncio.sleep(5)
    while True:
        try:
            cleanup_expired()
            autoscan = get_global_setting("AUTOSCAN", "ON")
            if autoscan != "ON":
                await asyncio.sleep(10)
                continue

            chats = approved_chat_ids()
            if not chats:
                await asyncio.sleep(10)
                continue

            sig = await scan_rotation_find_signal(app)
            if sig:
                # mark cooldown
                LAST_SENT_TS[(sig.symbol, sig.direction)] = now_ts()
                LAST_SENT_ENTRY[(sig.symbol, sig.direction)] = sig.entry

                msg = format_signal(sig)
                for chat_id in chats:
                    try:
                        await app.bot.send_message(chat_id=chat_id, text=msg, parse_mode="HTML")
                    except Exception:
                        pass
        except Exception as e:
            log.warning("autoscan_loop error: %s", e)

        await asyncio.sleep(max(30, SCAN_EVERY_SEC))

async def top200_refresh_loop(app: Application):
    await asyncio.sleep(2)
    while True:
        try:
            ex = app.bot_data.get("exchange")
            if ex:
                await rebuild_top200(ex)
        except Exception as e:
            log.warning("top200_refresh_loop error: %s", e)

        await asyncio.sleep(max(3600, TOP200_REFRESH_SEC))

# =========================
# Telegram handlers
# =========================
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    cleanup_expired()

    u = user_get(chat_id)
    if chat_id == OWNER_ID:
        text = (
            f"👑 <b>Owner panel</b>\n"
            f"🧩 <code>{BOT_VERSION}</code>\n\n"
            f"Сканим: <b>Top200 USDT Perpetual BingX</b>\n"
            f"Нажми кнопки ниже."
        )
        await update.message.reply_text(text, parse_mode="HTML", reply_markup=kb_main(chat_id))
        return

    if is_allowed(chat_id):
        until = user_get(chat_id)["approved_until"]
        text = (
            f"✅ Доступ активен\n"
            f"До: <b>{time.strftime('%Y-%m-%d %H:%M', time.localtime(until))}</b>\n\n"
            f"🧩 <code>{BOT_VERSION}</code>\n"
            f"Управление — кнопками."
        )
        await update.message.reply_text(text, parse_mode="HTML", reply_markup=kb_main(chat_id))
        return

    # not allowed
    if u and u["status"] == "PENDING":
        text = "⏳ Запрос на доступ уже отправлен. Жди одобрения."
    elif u and u["status"] in ("DENIED", "EXPIRED"):
        text = f"⛔ Доступ: {u['status']}. Можно запросить снова."
    else:
        text = "🔒 Доступ только после одобрения владельцем. Нажми кнопку ниже:"
    await update.message.reply_text(text, reply_markup=kb_start(chat_id))

async def cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    data = q.data or ""
    chat_id = q.message.chat_id
    await q.answer()

    cleanup_expired()

    # NOOP
    if data == "noop":
        return

    # request access
    if data == "request_access":
        user_upsert_request(chat_id)
        await q.message.edit_text("✅ Запрос отправлен. Ожидай одобрения владельца.")
        # notify owner
        try:
            await context.application.bot.send_message(
                chat_id=OWNER_ID,
                text=(
                    f"🆕 <b>Запрос доступа</b>\n"
                    f"chat_id: <code>{chat_id}</code>\n\n"
                    f"Выбери срок доступа:"
                ),
                parse_mode="HTML",
                reply_markup=kb_owner_approve(chat_id),
            )
        except Exception:
            pass
        return

    # owner approvals
    if data.startswith("approve:") or data.startswith("deny:"):
        if chat_id != OWNER_ID:
            await q.message.reply_text("⛔ Только владелец может это делать.")
            return

        if data.startswith("approve:"):
            _, uid, days = data.split(":")
            if not uid.isdigit() or not days.isdigit():
                return
            uid = int(uid); days = int(days)
            user_set_approved(uid, days, OWNER_ID)
            await q.message.reply_text(f"✅ Одобрено для {uid} на {days} дней.")
            try:
                until = user_get(uid)["approved_until"]
                await context.application.bot.send_message(
                    chat_id=uid,
                    text=(
                        f"✅ Доступ одобрен на <b>{days} дней</b>\n"
                        f"До: <b>{time.strftime('%Y-%m-%d %H:%M', time.localtime(until))}</b>\n\n"
                        f"Нажми /start"
                    ),
                    parse_mode="HTML",
                )
            except Exception:
                pass
            return

        if data.startswith("deny:"):
            _, uid = data.split(":")
            if not uid.isdigit():
                return
            uid = int(uid)
            user_set_denied(uid, OWNER_ID)
            await q.message.reply_text(f"⛔ Отказано для {uid}.")
            try:
                await context.application.bot.send_message(chat_id=uid, text="⛔ Доступ не одобрен владельцем.")
            except Exception:
                pass
            return

    # if not allowed -> block access to main features
    if not is_allowed(chat_id):
        await q.message.reply_text("🔒 Нет доступа. Нажми /start и запроси доступ.")
        return

    # MAIN MENU actions
    if data == "back_main":
        await q.message.edit_text("Меню:", reply_markup=kb_main(chat_id))
        return

    if data == "version":
        await q.message.reply_text(f"🧩 <code>{BOT_VERSION}</code>", parse_mode="HTML")
        return

    if data == "status":
        mode = get_global_setting("MODE", "BALANCED")
        autoscan = get_global_setting("AUTOSCAN", "ON")
        txt = (
            f"📌 <b>Status</b>\n\n"
            f"🧩 <code>{BOT_VERSION}</code>\n"
            f"Биржа: <code>{EXCHANGE_NAME}</code>\n"
            f"Автоскан: <b>{autoscan}</b>\n"
            f"Режим: <b>{mode}</b>\n"
            f"Top200: <b>{len(TOP200)}</b>\n"
            f"Top20: <b>{', '.join(TOP20[:8])}{'…' if len(TOP20)>8 else ''}</b>\n"
            f"Tick: каждые <b>{SCAN_EVERY_SEC}</b> сек\n"
            f"Rotation batch: <b>{ROTATION_BATCH}</b>\n"
            f"Cooldown: <b>{SIGNAL_COOLDOWN_MIN}</b> мин"
        )
        await q.message.reply_text(txt, parse_mode="HTML", reply_markup=kb_main(chat_id))
        return

    if data == "settings":
        await q.message.reply_text("⚙️ Настройки:", reply_markup=kb_settings(chat_id))
        return

    if data == "pick_mode":
        await q.message.edit_text("🎯 Выбери режим:", reply_markup=kb_pick_mode())
        return

    if data.startswith("set_mode:"):
        mode = data.split(":", 1)[1]
        if mode not in MODES:
            return
        set_global_setting("MODE", mode)
        await q.message.reply_text(f"✅ Режим установлен: {mode}", reply_markup=kb_main(chat_id))
        return

    if data == "toggle_autoscan":
        current = get_global_setting("AUTOSCAN", "ON")
        newv = "OFF" if current == "ON" else "ON"
        set_global_setting("AUTOSCAN", newv)
        await q.message.reply_text(f"⏯ Автоскан: {newv}", reply_markup=kb_main(chat_id))
        return

    if data == "scan_now":
        await q.message.reply_text("⏳ Сканирую топ-200 (принудительно)…")
        sig = await scan_rotation_find_signal(context.application)
        if not sig:
            await q.message.reply_text("Пока нет сильного сетапа в текущем батче. Попробуй позже.")
            return
        LAST_SENT_TS[(sig.symbol, sig.direction)] = now_ts()
        LAST_SENT_ENTRY[(sig.symbol, sig.direction)] = sig.entry
        await q.message.reply_text(format_signal(sig), parse_mode="HTML", reply_markup=kb_main(chat_id))
        return

    if data == "new_signal":
        await q.message.reply_text("🔎 Ищу лучший сигнал среди <b>Top20</b>…", parse_mode="HTML")
        sig = await new_signal_top20(context.application)
        if not sig:
            await q.message.reply_text("В Top20 сейчас нет сильного сетапа. Попробуй через 5–10 минут.")
            return
        # mark cooldown for this symbol/direction
        LAST_SENT_TS[(sig.symbol, sig.direction)] = now_ts()
        LAST_SENT_ENTRY[(sig.symbol, sig.direction)] = sig.entry
        await q.message.reply_text(format_signal(sig), parse_mode="HTML", reply_markup=kb_main(chat_id))
        return

    if data == "show_top200":
        # show first 40 to avoid huge spam
        top = TOP200[:40]
        txt = "🧺 <b>Top-200 (первые 40)</b>\n\n" + "\n".join([f"• <code>{s}</code>" for s in top])
        await q.message.reply_text(txt, parse_mode="HTML", reply_markup=kb_main(chat_id))
        return

async def cmd_owner(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_chat.id != OWNER_ID:
        return
    pending = []
    conn = db()
    cur = conn.execute("SELECT chat_id, requested_at FROM users WHERE status='PENDING' ORDER BY requested_at DESC LIMIT 20;")
    for chat_id, ts in cur.fetchall():
        pending.append((chat_id, ts))
    conn.close()

    if not pending:
        await update.message.reply_text("Нет ожидающих запросов.")
        return
    lines = ["🧾 <b>Ожидают одобрения</b>:\n"]
    for cid, ts in pending:
        lines.append(f"• <code>{cid}</code>  ({time.strftime('%Y-%m-%d %H:%M', time.localtime(ts))})")
    await update.message.reply_text("\n".join(lines), parse_mode="HTML")

# =========================
# Startup / shutdown
# =========================
async def on_startup(app: Application):
    init_db()

    # defaults
    if get_global_setting("MODE", "") == "":
        set_global_setting("MODE", "BALANCED")
    if get_global_setting("AUTOSCAN", "") == "":
        set_global_setting("AUTOSCAN", "ON")

    ex = await get_exchange()
    app.bot_data["exchange"] = ex

    # build initial universe & top200
    await rebuild_top200(ex)

    # background tasks
    asyncio.create_task(autoscan_loop(app))
    asyncio.create_task(top200_refresh_loop(app))

    log.info("BOT STARTED | %s | Top200=%s", BOT_VERSION, len(TOP200))

async def on_shutdown(app: Application):
    ex = app.bot_data.get("exchange")
    try:
        if ex:
            await ex.close()
    except Exception:
        pass

# =========================
# MAIN
# =========================
def main():
    app = Application.builder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("owner", cmd_owner))  # owner tool: list pending
    app.add_handler(CallbackQueryHandler(cb))

    app.post_init = on_startup
    app.post_shutdown = on_shutdown

    log.info("BOOT %s", BOT_VERSION)
    app.run_polling(allowed_updates=Update.ALL_TYPES, close_loop=False)

if __name__ == "__main__":
    main()
