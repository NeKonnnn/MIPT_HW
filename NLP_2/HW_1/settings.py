# Пути к файлам и директориям
DATAFILE = "data/data_IN/chatbot_data.json"
MODEL_INFO_DIR = "model_info"

# Параметры модели
MAX_LEN = 256
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
EPOCHS = 3


### ПОМЕНЯТЬ ДАННЫЕ!!!
# Токен для ТГ бота
API_TOKEN = '6709400952:AAGLPiE1sULjwmai70u05_m-gtNP53MWNc4'

WEBHOOK_HOST = '7197-45-83-41-143.ngrok-free.app'
WEBHOOK_PORT = 8000  # Can be 443, 80, 88 or 8443
WEBHOOK_LISTEN = '0.0.0.0'

WEBHOOK_URL_BASE = "https://{}".format(WEBHOOK_HOST)
WEBHOOK_URL_PATH = "/{}/".format(API_TOKEN)