"""
telegram_bot.py
Telegram-бот для управления несколькими scalper-ботами
"""

import asyncio
import logging
import shutil
import os
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes

from src.core.config import load_config
from src.utils.logger import setup_logger

logger = setup_logger("telegram_bot", logging.INFO)
config = load_config()

# ====================== КОМАНДЫ ======================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 Scalper Bot Manager\n\n"
        "Команды:\n"
        "/newbot ИМЯ — создать нового бота\n"
        "/startbot ИМЯ — запустить бота\n"
        "/stopbot ИМЯ — остановить бота\n"
        "/deletebot ИМЯ — полностью удалить бота\n"
        "/listbots — список всех ботов"
    )

async def newbot(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Использование: /newbot BTC")
        return
    bot_name = context.args[0].upper()
    await update.message.reply_text(f"Создаю бота {bot_name}...")
    import subprocess
    subprocess.run(["./manage_bot.sh", "create", bot_name])
    await update.message.reply_text(f"✅ Бот {bot_name} создан!\nОтредактируй config и выполни /startbot {bot_name}")

async def startbot(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Использование: /startbot BTC")
        return
    bot_name = context.args[0].upper()
    await update.message.reply_text(f"Запускаю бота {bot_name}...")
    import subprocess
    subprocess.run(["./manage_bot.sh", "start", bot_name])
    await update.message.reply_text(f"✅ Бот {bot_name} запущен!")

async def stopbot(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Использование: /stopbot BTC")
        return
    bot_name = context.args[0].upper()
    await update.message.reply_text(f"Останавливаю бота {bot_name}...")
    import subprocess
    subprocess.run(["./manage_bot.sh", "stop", bot_name])
    await update.message.reply_text(f"✅ Бот {bot_name} остановлен. Позиции остались на бирже.")

async def deletebot(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Использование: /deletebot BTC")
        return
    
    bot_name = context.args[0].upper()
    await update.message.reply_text(f"🗑 Удаляю бота {bot_name}...")

    import subprocess
    # Сначала останавливаем
    subprocess.run(["./manage_bot.sh", "stop", bot_name], check=False)

    # Полностью удаляем папку
    bot_dir = f"bots/bot-{bot_name.lower()}"
    if os.path.exists(bot_dir):
        shutil.rmtree(bot_dir)
        await update.message.reply_text(f"✅ Бот {bot_name} полностью удалён (папка + контейнер + данные).")
    else:
        await update.message.reply_text(f"⚠️ Папка бота {bot_name} не найдена.")

async def listbots(update: Update, context: ContextTypes.DEFAULT_TYPE):
    import subprocess
    result = subprocess.run(["./manage_bot.sh", "list"], capture_output=True, text=True)
    await update.message.reply_text(result.stdout or "Нет запущенных ботов")

# ====================== КНОПКА ГРАФИКА ======================

async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    if query.data.startswith("graph_"):
        symbol = query.data.split("_")[1]
        await query.edit_message_text(f"📊 Генерирую график для {symbol}...")
        await query.message.reply_text(f"График для {symbol} (в разработке)")

# ====================== ЗАПУСК ======================

async def main():
    app = Application.builder().token(config['monitoring']['telegram_token']).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("newbot", newbot))
    app.add_handler(CommandHandler("startbot", startbot))
    app.add_handler(CommandHandler("stopbot", stopbot))
    app.add_handler(CommandHandler("deletebot", deletebot))
    app.add_handler(CommandHandler("listbots", listbots))
    app.add_handler(CallbackQueryHandler(button_handler))

    logger.info("Telegram-бот запущен")
    await app.run_polling()

if __name__ == '__main__':
    asyncio.run(main())