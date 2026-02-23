import os
import logging
import random
from datetime import datetime, timezone

import requests
from dotenv import load_dotenv
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

# -----------------------
# ENV + LOGGING
# -----------------------
load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN is missing. Set it in Railway Variables.")

OWNER_ID = os.getenv("OWNER_ID", "").strip()  # можно пустым
AUTO_EVERY_MIN = int(os.getenv("AUTO_EVERY_MIN", "15"))
SYMBOLS = os.getenv("SYMBOLS", "BTCUSDT,ETHUSDT").strip()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger("bingxbot")

# Простая память подписчиков (в Railway при перезапуске сбросится)
SUBSCRIBERS: set[int] = set()

# -----------------------
# UI
# -----------------------
def main_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("📌 Получить сигнал", callback_data="get_signal")],
        [InlineKeyboardButton("🏓 Ping", callback_data="ping")],
        [InlineKeyboardButton("✅ Подписаться на авто-сигналы", callback_data="sub")],
        [InlineKeyboardButton("❌ Отписаться", callback_data="unsub")],
    ])

# -----------------------
# HELPERS
# -----------------------
def now_utc_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

def pick_symbol() -> str:
    items = [s.strip() for s in SYMBOLS.split(",") if s.strip()]
    return random.choice(items) if items else "BTCUSDT"

def build_fake_signal() -> str:
    symbol = pick_symbol()
    direction = random.choice(["LONG 🟢", "SHORT 🔴"])
    entry = round(random.uniform(100, 50000), 2)
    tp = round(entry * (1.01 if "LONG" in direction else 0.99), 2)
    sl = round(entry * (0.99 if "LONG" in direction else 1.01), 2)
    return (
        f"📣 <b>{symbol}</b>\n"
        f"🕒 {now_utc_str()}\n\n"
        f"Direction: <b>{direction}</b>\n"
        f"Entry: <b>{entry}</b>\n"
        f"TP: <b>{tp}</b>\n"
        f"SL: <b>{sl}</b>\n"
    )

async def safe_send(context: ContextTypes.DEFAULT_TYPE, chat_id: int, text: str):
    try:
        await context.bot.send_message(chat_id=chat_id, text=text, parse_mode="HTML")
    except Exception as e:
        log.warning("send_message failed chat_id=%s err=%s", chat_id, e)

# -----------------------
# HANDLERS
# -----------------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    text = (
        "👋 Привет! Я бот.\n\n"
        "Кнопки ниже:\n"
        "• Получить сигнал\n"
        "• Ping\n"
        "• Подписка на авто-сигналы\n"
    )
    await update.message.reply_text(text, reply_markup=main_kb())

async def cmd_subscribe(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    SUBSCRIBERS.add(chat_id)
    await update.message.reply_text("✅ Подписал. Авто-сигналы будут приходить периодически.", reply_markup=main_kb())

async def cmd_unsubscribe(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    SUBSCRIBERS.discard(chat_id)
    await update.message.reply_text("❌ Отписал.", reply_markup=main_kb())

async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    text = (
        f"🧾 Status\n"
        f"Time: {now_utc_str()}\n"
        f"AUTO_EVERY_MIN: {AUTO_EVERY_MIN}\n"
        f"SYMBOLS: {SYMBOLS}\n"
        f"Subscribers: {len(SUBSCRIBERS)}\n"
        f"Chat: {chat_id}\n"
    )
    await update.message.reply_text(text, reply_markup=main_kb())

async def on_button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()

    chat_id = q.message.chat_id
    data = q.data

    if data == "ping":
        await q.message.reply_text("pong ✅", reply_markup=main_kb())
        return

    if data == "get_signal":
        await q.message.reply_text(build_fake_signal(), parse_mode="HTML", reply_markup=main_kb())
        return

    if data == "sub":
        SUBSCRIBERS.add(chat_id)
        await q.message.reply_text("✅ Подписал на авто-сигналы.", reply_markup=main_kb())
        return

    if data == "unsub":
        SUBSCRIBERS.discard(chat_id)
        await q.message.reply_text("❌ Отписал.", reply_markup=main_kb())
        return

# -----------------------
# AUTOSCAN JOB
# -----------------------
async def autoscan_job(context: ContextTypes.DEFAULT_TYPE):
    if not SUBSCRIBERS:
        return

    text = "🤖 Авто-сигнал\n\n" + build_fake_signal()
    for chat_id in list(SUBSCRIBERS):
        await safe_send(context, chat_id, text)

# -----------------------
# MAIN
# -----------------------
def main():
    app = Application.builder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("sub", cmd_subscribe))
    app.add_handler(CommandHandler("unsub", cmd_unsubscribe))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CallbackQueryHandler(on_button))

    # автозадача
    app.job_queue.run_repeating(autoscan_job, interval=AUTO_EVERY_MIN * 60, first=10)

    log.info("BOT STARTING...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
