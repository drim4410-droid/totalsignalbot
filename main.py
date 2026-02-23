import os
import asyncio
import logging
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from dotenv import load_dotenv

import ccxt.async_support as ccxt
import pandas as pd
import numpy as np

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

# ----------------------------
# LOGGING
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger("signalbot")

# ----------------------------
# CONFIG
# ----------------------------
load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN is missing")

EXCHANGE_NAME = os.getenv("EXCHANGE", "bingx").strip().lower()

SYMBOLS = [s.strip() for s in os.getenv("SYMBOLS", "BTC/USDT,ETH/USDT").split(",") if s.strip()]
TIMEFRAMES = [t.strip() for t in os.getenv("TIMEFRAMES", "5m,15m,1h,4h").split(",") if t.strip()]

SCAN_EVERY_MIN = int(os.getenv("SCAN_EVERY_MIN", "15"))
MIN_ATR_PCT = float(os.getenv("MIN_ATR_PCT", "0.6"))  # ATR% threshold

# Внутренние настройки стратегии
EMA_FAST = 50
EMA_SLOW = 200
RSI_LEN = 14
ATR_LEN = 14
ADX_LEN = 14

# ----------------------------
# SIMPLE INDICATORS (no external ta-lib)
# ----------------------------
def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()

def rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    gain = (delta.where(delta > 0, 0.0)).rolling(length).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(length).mean()
    rs = gain / (loss.replace(0, np.nan))
    return 100 - (100 / (1 + rs))

def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        (df["high"] - df["low"]).abs(),
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr

def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    tr = true_range(df)
    return tr.rolling(length).mean()

def adx(df: pd.DataFrame, length: int = 14) -> pd.Series:
    # Classic ADX (simplified, sufficient for filtering)
    high = df["high"]
    low = df["low"]
    close = df["close"]

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

# ----------------------------
# STRATEGY
# ----------------------------
@dataclass
class Signal:
    symbol: str
    direction: str  # LONG / SHORT
    timeframe: str
    entry: float
    sl: float
    tp1: float
    tp2: float
    reasons: List[str]

def compute_signal(df: pd.DataFrame, symbol: str, tf: str, trend_bias: Optional[str]) -> Optional[Signal]:
    """
    trend_bias: "LONG" / "SHORT" from higher timeframe filters (1h/4h).
    """
    if len(df) < 250:
        return None

    df = df.copy()
    df["ema_fast"] = ema(df["close"], EMA_FAST)
    df["ema_slow"] = ema(df["close"], EMA_SLOW)
    df["rsi"] = rsi(df["close"], RSI_LEN)
    df["atr"] = atr(df, ATR_LEN)
    df["adx"] = adx(df, ADX_LEN)

    last = df.iloc[-1]
    prev = df.iloc[-2]

    atr_pct = (last["atr"] / last["close"]) * 100 if pd.notna(last["atr"]) else 0
    if atr_pct < MIN_ATR_PCT:
        return None

    # Trend on same TF (extra safety)
    local_trend_long = last["ema_fast"] > last["ema_slow"]
    local_trend_short = last["ema_fast"] < last["ema_slow"]

    # ADX filter (avoid flat market)
    if pd.isna(last["adx"]) or last["adx"] < 18:
        return None

    # Trigger: close breaks previous swing area (simple but robust)
    # LONG trigger: close > max(high of last 20) and RSI not overbought too hard
    # SHORT trigger: close < min(low of last 20) and RSI not oversold too hard
    lookback = 20
    hh = df["high"].iloc[-lookback-1:-1].max()
    ll = df["low"].iloc[-lookback-1:-1].min()

    entry = float(last["close"])
    atr_val = float(last["atr"])

    reasons = [
        f"ATR%={atr_pct:.2f} (волатильность ок)",
        f"ADX={float(last['adx']):.1f} (есть трендовость)",
    ]

    # Decide direction with higher timeframe bias if provided
    # If bias exists, only trade with it.
    if trend_bias == "LONG":
        if not local_trend_long:
            return None
    if trend_bias == "SHORT":
        if not local_trend_short:
            return None

    # LONG setup
    if (entry > hh) and local_trend_long and (last["rsi"] < 72):
        sl = entry - 1.5 * atr_val
        tp1 = entry + 1.5 * (entry - sl)  # RR ~ 1:1.5
        tp2 = entry + 2.5 * (entry - sl)  # RR ~ 1:2.5
        reasons += [
            "EMA50 > EMA200 (тренд вверх)",
            f"Пробой high {lookback} свечей: close({entry:.4f}) > {hh:.4f}",
            f"RSI={float(last['rsi']):.1f} (не перегрет)"
        ]
        return Signal(symbol, "LONG", tf, entry, sl, tp1, tp2, reasons)

    # SHORT setup
    if (entry < ll) and local_trend_short and (last["rsi"] > 28):
        sl = entry + 1.5 * atr_val
        tp1 = entry - 1.5 * (sl - entry)
        tp2 = entry - 2.5 * (sl - entry)
        reasons += [
            "EMA50 < EMA200 (тренд вниз)",
            f"Пробой low {lookback} свечей: close({entry:.4f}) < {ll:.4f}",
            f"RSI={float(last['rsi']):.1f} (не перепродан)"
        ]
        return Signal(symbol, "SHORT", tf, entry, sl, tp1, tp2, reasons)

    return None

# ----------------------------
# EXCHANGE / DATA
# ----------------------------
def parse_ohlcv(ohlcv: List[List[float]]) -> pd.DataFrame:
    # [timestamp, open, high, low, close, volume]
    df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna()

async def fetch_df(ex, symbol: str, tf: str, limit: int = 500) -> Optional[pd.DataFrame]:
    try:
        ohlcv = await ex.fetch_ohlcv(symbol, timeframe=tf, limit=limit)
        if not ohlcv or len(ohlcv) < 200:
            return None
        return parse_ohlcv(ohlcv)
    except Exception as e:
        log.warning(f"fetch_ohlcv failed: {symbol} {tf} -> {e}")
        return None

async def get_exchange():
    if not hasattr(ccxt, EXCHANGE_NAME):
        raise RuntimeError(f"Exchange '{EXCHANGE_NAME}' not found in ccxt")
    ex_class = getattr(ccxt, EXCHANGE_NAME)
    ex = ex_class({
        "enableRateLimit": True,
        # публичных данных достаточно, ключи не нужны
        "options": {"defaultType": "swap"}  # для деривативов (если биржа поддерживает)
    })
    return ex

# ----------------------------
# TELEGRAM BOT STATE
# ----------------------------
SUBSCRIBERS: set[int] = set()
LAST_SENT: Dict[Tuple[str, str, str], float] = {}  # (symbol, tf, dir) -> last_entry to avoid spam

def fmt_price(x: float) -> str:
    # Pretty formatting
    if x >= 100:
        return f"{x:.2f}"
    if x >= 1:
        return f"{x:.4f}"
    return f"{x:.6f}"

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    SUBSCRIBERS.add(chat_id)
    await update.message.reply_text(
        "✅ Подписал.\n\n"
        "Команды:\n"
        "/status — статус\n"
        "/on — включить сигналы\n"
        "/off — выключить сигналы\n"
        "/pairs — показать пары\n"
        "/tfs — показать таймфреймы\n\n"
        "Сигналы: тренд + волатильность + пробой. Меньше шума, больше качества."
    )
    context.application.bot_data["signals_enabled"] = True

async def cmd_on(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.application.bot_data["signals_enabled"] = True
    await update.message.reply_text("✅ Сигналы включены.")

async def cmd_off(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.application.bot_data["signals_enabled"] = False
    await update.message.reply_text("⛔ Сигналы выключены.")

async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    enabled = context.application.bot_data.get("signals_enabled", True)
    await update.message.reply_text(
        f"Статус: {'✅ ON' if enabled else '⛔ OFF'}\n"
        f"Биржа: {EXCHANGE_NAME}\n"
        f"Пары: {', '.join(SYMBOLS)}\n"
        f"TF: {', '.join(TIMEFRAMES)}\n"
        f"Скан каждые: {SCAN_EVERY_MIN} мин\n"
        f"MIN_ATR_PCT: {MIN_ATR_PCT}"
    )

async def cmd_pairs(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Пары: " + ", ".join(SYMBOLS))

async def cmd_tfs(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Таймфреймы: " + ", ".join(TIMEFRAMES))

# ----------------------------
# SCAN LOOP
# ----------------------------
def higher_tf_bias(df_1h: Optional[pd.DataFrame], df_4h: Optional[pd.DataFrame]) -> Optional[str]:
    """
    Determine global bias: LONG/SHORT/None based on EMA50/EMA200 on higher TFs.
    """
    def bias(df: pd.DataFrame) -> Optional[str]:
        if df is None or len(df) < 250:
            return None
        c = df["close"]
        e50 = ema(c, EMA_FAST).iloc[-1]
        e200 = ema(c, EMA_SLOW).iloc[-1]
        if pd.isna(e50) or pd.isna(e200):
            return None
        return "LONG" if e50 > e200 else "SHORT" if e50 < e200 else None

    b1 = bias(df_1h) if df_1h is not None else None
    b4 = bias(df_4h) if df_4h is not None else None
    # если оба совпали — отлично, иначе не торгуем (повышаем качество)
    if b1 and b4 and b1 == b4:
        return b1
    return None

async def scan_and_send(app: Application):
    enabled = app.bot_data.get("signals_enabled", True)
    if not enabled:
        return

    ex = app.bot_data.get("exchange")
    if ex is None:
        return

    for symbol in SYMBOLS:
        # higher TF bias from 1h + 4h
        df_1h = await fetch_df(ex, symbol, "1h", 500) if "1h" in TIMEFRAMES or "4h" in TIMEFRAMES else None
        df_4h = await fetch_df(ex, symbol, "4h", 500) if "4h" in TIMEFRAMES else None
        bias = higher_tf_bias(df_1h, df_4h)

        # работаем на 15m/5m как “триггеры”
        for tf in TIMEFRAMES:
            if tf in ("1h", "4h"):
                continue  # эти уже как фильтр
            df = await fetch_df(ex, symbol, tf, 500)
            if df is None:
                continue

            sig = compute_signal(df, symbol, tf, bias)
            if not sig:
                continue

            key = (sig.symbol, sig.timeframe, sig.direction)
            last_entry = LAST_SENT.get(key)
            # антиспам: если entry почти тот же — не шлём снова
            if last_entry and abs(last_entry - sig.entry) / sig.entry < 0.002:
                continue
            LAST_SENT[key] = sig.entry

            text = (
                f"📣 *{sig.symbol}*  `{sig.timeframe}`\n"
                f"*{sig.direction}*\n\n"
                f"Entry: `{fmt_price(sig.entry)}`\n"
                f"SL: `{fmt_price(sig.sl)}`\n"
                f"TP1: `{fmt_price(sig.tp1)}`\n"
                f"TP2: `{fmt_price(sig.tp2)}`\n\n"
                f"Причины:\n- " + "\n- ".join(sig.reasons)
            )

            for chat_id in list(SUBSCRIBERS):
                try:
                    await app.bot.send_message(chat_id=chat_id, text=text, parse_mode="Markdown")
                except Exception as e:
                    log.warning(f"send_message failed to {chat_id}: {e}")

async def scanner_loop(app: Application):
    while True:
        try:
            await scan_and_send(app)
        except Exception as e:
            log.exception(f"scan loop error: {e}")
        await asyncio.sleep(SCAN_EVERY_MIN * 60)

# ----------------------------
# MAIN
# ----------------------------
async def on_startup(app: Application):
    ex = await get_exchange()
    app.bot_data["exchange"] = ex
    app.bot_data["signals_enabled"] = True
    asyncio.create_task(scanner_loop(app))
    log.info("BOT STARTED")

async def on_shutdown(app: Application):
    ex = app.bot_data.get("exchange")
    try:
        if ex:
            await ex.close()
    except Exception:
        pass
    log.info("BOT STOPPED")

def build_app() -> Application:
    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("on", cmd_on))
    app.add_handler(CommandHandler("off", cmd_off))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("pairs", cmd_pairs))
    app.add_handler(CommandHandler("tfs", cmd_tfs))
    return app

def main():
    app = build_app()
    app.post_init = on_startup
    app.post_shutdown = on_shutdown
    app.run_polling(close_loop=False)

if __name__ == "__main__":
    main()
