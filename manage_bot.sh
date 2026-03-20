#!/bin/bash
set -e

COMMAND=$1
BOT_NAME=$2

if [ -z "$COMMAND" ]; then
    echo "Использование:"
    echo "  ./manage_bot.sh create ИМЯ     # создать бота"
    echo "  ./manage_bot.sh start ИМЯ     # запустить бота"
    echo "  ./manage_bot.sh stop ИМЯ      # остановить бота"
    echo "  ./manage_bot.sh list         # список всех ботов"
    exit 1
fi

BOT_DIR="bots/bot-${BOT_NAME,,}"
CONFIG_DIR="$BOT_DIR/config"

case "$COMMAND" in
    create)
        if [ -z "$BOT_NAME" ]; then
            echo "Укажите имя бота. Пример: ./manage_bot.sh create BTC"
            exit 1
        fi
        mkdir -p "$CONFIG_DIR"
        cp config/bot_config.yaml "$CONFIG_DIR/bot_config.yaml"
        echo "✅ Бот $BOT_NAME создан."
        echo "   Папка: $BOT_DIR"
        echo "   Теперь отредактируйте конфиг: $CONFIG_DIR/bot_config.yaml"
        echo "   После редактирования запустите: ./manage_bot.sh start $BOT_NAME"
        ;;

    start)
        if [ -z "$BOT_NAME" ]; then
            echo "Укажите имя бота."
            exit 1
        fi
        if [ ! -d "$BOT_DIR" ]; then
            echo "Ошибка: бот $BOT_NAME не создан. Сначала выполните create."
            exit 1
        fi
        echo "🚀 Запускаем бота $BOT_NAME..."
        docker run -d \
            --name "scalper-${BOT_NAME,,}" \
            --restart unless-stopped \
            -v "$(pwd)/$BOT_DIR/config:/app/config" \
            -v "$(pwd)/logs/${BOT_NAME,,}:/app/logs" \
            scalper-bot
        echo "✅ Бот $BOT_NAME запущен. Логи: docker logs -f scalper-${BOT_NAME,,}"
        ;;

    stop)
        if [ -z "$BOT_NAME" ]; then
            echo "Укажите имя бота."
            exit 1
        fi
        echo "⏹️ Останавливаем бота $BOT_NAME..."
        docker stop "scalper-${BOT_NAME,,}" || true
        docker rm "scalper-${BOT_NAME,,}" || true
        echo "✅ Бот $BOT_NAME остановлен. Открытые позиции остались на бирже."
        ;;

    list)
        echo "Список ботов:"
        docker ps --filter "name=scalper-" --format "table {{.Names}}\t{{.Status}}"
        ;;

    *)
        echo "Неизвестная команда: $COMMAND"
        ;;
esac