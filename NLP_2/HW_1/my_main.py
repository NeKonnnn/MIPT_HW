import logging
import telebot
from fastapi import FastAPI
import uvicorn
from cross_encoder_model import *  # Импортируем все необходимые функции и переменные
from settings import *  # Импорт настроек

# Logger setup
logger = telebot.logger
telebot.logger.setLevel(logging.INFO)

bot = telebot.TeleBot(API_TOKEN)
app = FastAPI(docs=None, redoc_url=None)

# Обработчики сообщений
@app.post(WEBHOOK_URL_PATH)
def process_webhook(update: dict):
    update = telebot.types.Update.de_json(update)
    bot.process_new_updates([update])

@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    bot.reply_to(message, "Привет, Морти. Зачем пришел?")

@bot.message_handler(func=lambda message: True, content_types=['text'])
def echo_message(message):
    best_response, confidence = find_best_response(message.text, pattern_embeddings, intents_data, tokenizer, embedding_model)
    bot.reply_to(message, best_response)

bot.remove_webhook()
bot.set_webhook(url=WEBHOOK_URL_BASE + WEBHOOK_URL_PATH)

# Запуск веб сервера
if __name__ == "__main__":
    uvicorn.run(app, host=WEBHOOK_LISTEN, port=WEBHOOK_PORT)