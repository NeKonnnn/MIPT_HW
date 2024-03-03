import numpy as np
import re
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer, T5ForConditionalGeneration,  Trainer, TrainingArguments, DataCollatorForLanguageModeling
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TrainerCallback
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import csv
from nltk.tokenize import sent_tokenize

# Функция для предобработки текста
def preprocess_text_improved(text):
    if isinstance(text, str):
        # Удаление специальных символов и лишних пробелов, кроме знаков пунктуации, которые могут быть важны для контекста
        text = re.sub(r"[^а-яА-Я0-9,.!?'\s]", "", text)
        text = text.strip()
        return text
    else:
        return ""

# Загрузка и предобработка данных
data_path = './data/data_IN/chat_corpus_train.csv'
df = pd.read_csv(data_path)

# Применение функции предобработки к столбцам 'Question' и 'Answer'
df['Question'] = df['Question'].apply(preprocess_text_improved)
df['Answer'] = df['Answer'].apply(preprocess_text_improved)

# Комбинирование вопросов и ответов в одну строку с разделителем
df['combined_text'] = df['Question'] + " [SEP] " + df['Answer']

# Предполагается, что у вас есть DataFrame `df` с колонкой 'combined_text' для обучения
texts = df['combined_text'].tolist()

# Инициализация токенизатора и модели
tokenizer = GPT2Tokenizer.from_pretrained('sberbank-ai/rugpt3large_based_on_gpt2')
model = GPT2LMHeadModel.from_pretrained('sberbank-ai/rugpt3large_based_on_gpt2')

class RickDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = {key: torch.tensor(val) for key, val in encodings.items()}
        self.labels = self.encodings['input_ids'].detach().clone()

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

# Разделение на обучающую и тестовую выборки
train_texts, val_texts = train_test_split(texts, test_size=0.2)

# Токенизация текстов
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=512)

# Создание объектов Dataset
train_dataset = RickDataset(train_encodings)
val_dataset = RickDataset(val_encodings)

# Настройка обучения
training_args = TrainingArguments(
    output_dir='./results',
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch"
)

# Класс для отслеживания потерь
class LossLoggingCallback(TrainerCallback):
    def __init__(self):
        super().__init__()
        self.losses = []
        self.eval_losses = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if 'loss' in logs:
            self.losses.append(logs['loss'])
        if 'eval_loss' in logs:
            self.eval_losses.append(logs['eval_loss'])

# Инициализация объекта для отслеживания потерь
loss_logging_callback = LossLoggingCallback()

# Инициализация Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    callbacks=[loss_logging_callback]
)

# Обучение модели
trainer.train()

# Сохранение модели и токенизатора
model.save_pretrained("./rick_model")
tokenizer.save_pretrained("./rick_model")

# Вывод графиков потерь
plt.figure(figsize=(12, 6))
plt.plot(loss_logging_callback.losses, label='Training Loss')
plt.plot(loss_logging_callback.eval_losses, label='Validation Loss', linestyle='--')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# Загрузка модели и токенизатора
model = GPT2LMHeadModel.from_pretrained("./rick_model").to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
tokenizer = GPT2Tokenizer.from_pretrained("./rick_model")
def find_similar_response(dataset, input_text, threshold=0.8):
    input_text_lower = input_text.lower()
    for index, row in dataset.iterrows():
        question_lower = row['Question'].lower()
        # Проверка наличия вхождения текста вопроса в столбец 'Question'
        if input_text_lower in question_lower:
            return row['Answer']  # Возвращение ответа, соответствующего вопросу
    return None

# --!!улучшения фактологической связности ответа c использованием контекстно-зависимых эмбеддингов!!--

# Инициализация T5 модели и токенизатора для генерации ответа с контекстом
t5_model_name = 't5-large'
t5_tokenizer = T5Tokenizer.from_pretrained(t5_model_name)
t5_model = T5ForConditionalGeneration.from_pretrained(t5_model_name)

# Функция для генерации ответа с использованием модели T5
def generate_answer_with_t5(context, question):
    input_text = f"context: {context} question: {question}"
    input_ids = t5_tokenizer.encode(input_text, return_tensors='pt', add_special_tokens=True)
    output_ids = t5_model.generate(
        input_ids,
        max_length=300,
        num_beams=5,
        early_stopping=True,
        temperature=0.9,  
        top_k=50,  # Ограничение количества кандидатов на каждом шаге
        top_p=0.95,  # Использование nucleus sampling
    )
    answer = t5_tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return answer

# Функция для интеграции T5 
def generate_response(input_text, context=""):
    answer = generate_answer_with_t5(context, input_text)
    return answer

# Пример использования новой функции генерации ответов
context = "Я пошел собирать цветы астры"
question = "какие цветы я пошел собирать?"

answer = generate_response(question, context)
print(f"Ответ: {answer}")

# --!!код инференса модели!!--

def generate_response(input_text, base_max_length=100, temperature=0.8, top_k=50, top_p=0.95, no_repeat_ngram_size=2):
    # Специальный ответ на конкретный вопрос
    if "как тебя зовут" in input_text.lower():
        return "Меня зовут Рик."

    # Попытка найти похожий ответ в датасете
    similar_response = find_similar_response(df, input_text)
    if similar_response:
        return similar_response  # Использование найденного ответа

    # Детализированный контекст о Рике для моделирования его стиля общения
    rick_context = "Ты гениальный учёный, который часто использует сарказм и научные термины. Ты часто называешь людей Морти. Ты готов помогать людям."
    input_text_with_context = rick_context + '"' + input_text + '"'

    model.eval()
    input_ids = tokenizer.encode(input_text_with_context, return_tensors='pt').to(model.device)
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=base_max_length + len(input_ids[0]),
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            no_repeat_ngram_size=no_repeat_ngram_size,
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=1
        )

    response = tokenizer.decode(output[0], skip_special_tokens=True)
    response = response.replace(input_text_with_context, "").strip().strip('"')

    # Окончание на полном предложении
    sentences = sent_tokenize(response)
    if sentences:
        response = sentences[0]

    if not response:  # Если ответ пустой, предложить повторить вопрос
        response = "Повтори-ка, не понял тебя."
    return response

def add_rick_context(input_text):
    rick_context = "Ты гениальный учёный, который часто использует сарказм и научные термины."
    return f"{rick_context} '{input_text}'"

def postprocess_rick_response(response):
    if "наука" in response:
        response += " Наука, бэйби!"
    return response

def interactive_learning(input_text, generated_response):
    print(f"Рик: {generated_response}")
    user_feedback = input("Это похоже на Рика? (да/нет): ").lower()
    if user_feedback == "нет":
        corrected_response = input("Введите более подходящий ответ Рика: ")
        # Добавление исправленного ответа в датасет
        with open('corrections.csv', 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([input_text, corrected_response])

def main():
    while True:
        input_text = input("Вопрос к Рику: ")
        if input_text.lower() == 'выход':
            break

        # Добавление контекста к запросу
        input_text_with_context = add_rick_context(input_text)
        response = generate_response(input_text_with_context)
        response = postprocess_rick_response(response)
        interactive_learning(input_text, response)

if __name__ == "__main__":
    main()

