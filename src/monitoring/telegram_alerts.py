"""
src/monitoring/telegram_alerts.py

=== Основной принцип работы файла ===

Модуль Telegram-уведомлений + генерация графика.
- send_position_alert — отправляет подробный лог + кнопку "Показать график"
- button_handler — по callback "graph_{symbol}" строит график
- **Все** данные берутся только из готовых функций проекта:
  • anomalous_surge_channel_feature — price channel
  • calculate_volume_profile_va — VAH / VAL
  • AnomalyDetector — аномальные свечи
  • compare_halves — Avg Left Half / Avg Right Half (средние цены в половинах)
"""

import asyncio
import logging
import os
from datetime import datetime

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application

from src.core.config import load_config
from src.data.storage import Storage
from src.features.channels import anomalous_surge_channel_feature, calculate_volume_profile_va
from src.features.anomaly_detector import AnomalyDetector
from src.features.half_comparator import compare_halves   # ← для средних в половинах
from src.utils.logger import setup_logger

logger = setup_logger("telegram_alerts", logging.INFO)
config = load_config()
storage = Storage()

# Папка для графиков
os.makedirs("graphs", exist_ok=True)


async def send_position_alert(log_text: str, symbol: str):
    """Отправляет подробный лог + кнопку 'Показать график'"""
    if not config.get('monitoring', {}).get('enable_telegram', False):
        return

    token = config['monitoring'].get('telegram_token')
    chat_id = config['monitoring'].get('telegram_chat_id')
    if not token or not chat_id:
        return

    keyboard = [[InlineKeyboardButton("📊 Показать график", callback_data=f"graph_{symbol}")]]
    reply_markup = InlineKeyboardMarkup(keyboard)

    try:
        app = Application.builder().token(token).build()
        await app.bot.send_message(
            chat_id=chat_id,
            text=log_text,
            reply_markup=reply_markup,
            parse_mode='HTML'
        )
    except Exception as e:
        logger.error(f"Ошибка отправки Telegram: {e}")


async def button_handler(update, context):
    """Обработка нажатия кнопки 'Показать график'"""
    query = update.callback_query
    await query.answer()

    if not query.data.startswith("graph_"):
        return

    symbol = query.data.split("_")[1]
    await query.edit_message_text(f"📊 Генерирую график для {symbol}...")

    # === ВСЁ БЕРЁТСЯ ИЗ ГОТОВЫХ ФУНКЦИЙ ПРОЕКТА ===
    df = storage.get_candles(symbol, "1m", limit=300)
    if df.empty:
        await query.message.reply_text("❌ Нет данных для графика")
        return

    df = df.sort_index()

    # 1. Price Channel — напрямую из проекта
    channel_df = anomalous_surge_channel_feature(df, period=config["features"].get("price_channel_period", 100))

    # 2. VAH / VAL — напрямую из проекта
    va_dict = calculate_volume_profile_va(df)
    vah = va_dict.get('vah')
    val = va_dict.get('val')

    # 3. Средние цены в половинах — напрямую из compare_halves (готовая функция проекта)
    window_size = config["features"].get("half_comparison_period", 100)
    window_df = df.iloc[-window_size:]
    half_changes = compare_halves(window_df, window_size=window_size, va_std={}, va_delta={})
    avg_left = half_changes.get('left_mean_price', df['close'].iloc[:window_size//2].mean())
    avg_right = half_changes.get('right_mean_price', df['close'].iloc[window_size//2:].mean())

    # 4. Аномальные свечи — напрямую из AnomalyDetector
    anomaly_detector = AnomalyDetector(config)
    anomaly_flags = anomaly_detector.detect(df)
    anomalies = df[(anomaly_flags.get('C', pd.Series(False))) | (anomaly_flags.get('CV', pd.Series(False)))]

    # График
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        row_heights=[0.55, 0.25, 0.20],
                        vertical_spacing=0.03,
                        subplot_titles=("Price + VA + Channel", "Volume + 95th", "Anomalies"))

    fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'],
                                 low=df['low'], close=df['close'], name="Price"), row=1, col=1)

    if vah is not None:
        fig.add_hline(y=vah, line_dash="dash", line_color="red", annotation_text="VAH", row=1, col=1)
    if val is not None:
        fig.add_hline(y=val, line_dash="dash", line_color="green", annotation_text="VAL", row=1, col=1)

    fig.add_trace(go.Scatter(x=channel_df.index, y=channel_df['asc_upper'],
                             name="Upper Channel", line=dict(color='red', dash='dash')), row=1, col=1)
    fig.add_trace(go.Scatter(x=channel_df.index, y=channel_df['asc_lower'],
                             name="Lower Channel", line=dict(color='green', dash='dash')), row=1, col=1)

    fig.add_hline(y=avg_left, line_color="orange", annotation_text="Avg Left Half", row=1, col=1)
    fig.add_hline(y=avg_right, line_color="blue", annotation_text="Avg Right Half", row=1, col=1)

    # Volume + 95th percentile
    fig.add_trace(go.Bar(x=df.index, y=df['volume'], name="Volume", marker_color="lightblue"), row=2, col=1)
    percentile_95 = df['volume'].quantile(0.95)
    fig.add_hline(y=percentile_95, line_dash="dot", line_color="purple", annotation_text="95th Volume", row=2, col=1)

    # Аномалии
    fig.add_trace(go.Scatter(x=anomalies.index, y=anomalies['close'],
                             mode="markers", marker=dict(color="red", size=9, symbol="x"),
                             name="Anomaly"), row=3, col=1)

    fig.update_layout(
        title=f"{symbol} — Scalping Chart ({datetime.now().strftime('%H:%M:%S')})",
        height=950,
        template="plotly_dark",
        showlegend=False
    )

    graph_path = f"graphs/{symbol}_live.png"
    fig.write_image(graph_path, width=1200, height=950)

    await context.bot.send_photo(
        chat_id=query.message.chat_id,
        photo=open(graph_path, "rb"),
        caption=f"📈 {symbol} — Полный scalping-график (VA + Channel + Half Averages + Anomalies)"
    )

    logger.info(f"График отправлен для {symbol}")