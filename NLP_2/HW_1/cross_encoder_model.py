import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, BertModel
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
import os
import pickle
import torch.quantization
from settings import *

# Загрузка и предобработка данных
def load_data(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        data = json.load(file)
    texts = []
    intents = []
    intents_data = data['intents']
    for intent in intents_data:
        for pattern in intent['patterns']:
            texts.append(pattern)
            intents.append(intent['tag'])
    return texts, intents, intents_data, data

texts, labels, intents_data, data = load_data(DATAFILE)
df = pd.DataFrame({'pattern': texts, 'response': labels})
train_df, val_df = train_test_split(df, test_size=0.1)

class ChatbotDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, index):
        pattern = str(self.data.iloc[index]['pattern'])
        response = str(self.data.iloc[index]['response'])
        inputs = self.tokenizer.encode_plus(
            pattern, response,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_attention_mask=True,
            truncation=True
        )
        return {
            'input_ids': torch.tensor(inputs['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(inputs['attention_mask'], dtype=torch.long),
            'labels': torch.tensor(1, dtype=torch.float)
        }

    def __len__(self):
        return self.len

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
classification_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classification_model.to(device)

# DataLoader
train_dataset = ChatbotDataset(train_df, tokenizer, MAX_LEN)
val_dataset = ChatbotDataset(val_df, tokenizer, MAX_LEN)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# Оптимизатор
optimizer = AdamW(classification_model.parameters(), lr=LEARNING_RATE)

def calculate_accuracy(loader, model, device):
    model.eval().to(device)
    correct_preds, num_samples = 0, 0
    with torch.no_grad():
        for data in loader:
            input_ids = data['input_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)
            labels = data['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            preds = torch.round(torch.sigmoid(outputs.logits))
            correct_preds += (preds.flatten() == labels).cpu().numpy().sum()
            num_samples += labels.size(0)
    return correct_preds / num_samples

# Обучение и валидация модели
train_losses, val_losses = [], []
for epoch in range(EPOCHS):
    classification_model.train()
    total_loss, total_val_loss = 0, 0
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = classification_model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    
    train_acc = calculate_accuracy(train_loader, classification_model, device)
    val_acc = calculate_accuracy(val_loader, classification_model, device)
    for batch in val_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        with torch.no_grad():
            outputs = classification_model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_val_loss += loss.item()
    
    avg_train_loss = total_loss / len(train_loader)
    avg_val_loss = total_val_loss / len(val_loader)
    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)
    
    print(f'Epoch {epoch+1}/{EPOCHS}, Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}, Train Accuracy: {train_acc:.4f}, Validation Accuracy: {val_acc:.4f}')

# Графики функций потерь
epochs_range = range(1, EPOCHS + 1)
plt.figure(figsize=(12, 6))
plt.plot(epochs_range, train_losses, 'b-', label='Training Loss')
plt.plot(epochs_range, val_losses, 'r-', label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# ---!!!Ускорение модели!!!---
# Подготовка модели к квантованию и квантование модели
model_to_quantize = classification_model.cpu()
model_to_quantize.eval()
quantized_model = torch.quantization.quantize_dynamic(
    model=model_to_quantize, 
    qconfig_spec={torch.nn.Linear}, 
    dtype=torch.qint8
)

# Проверка качества квантованной модели
val_accuracy_quantized = calculate_accuracy(val_loader, quantized_model, 'cpu')
print(f'Validation Accuracy (Quantized): {val_accuracy_quantized:.4f}')

# Сравнение качества до и после квантования
val_accuracy_original = calculate_accuracy(val_loader, classification_model.cpu(), 'cpu')
print(f'Validation Accuracy (Original): {val_accuracy_original:.4f}')


# --!!код инференса модели!!--
# Загрузка модели и токенизатора для эмбеддингов
embedding_model = BertModel.from_pretrained('bert-base-uncased')
embedding_model.eval()

# Функция для получения векторного представления текста
def get_embedding(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()
    return embeddings

# Инициализация модели и токенизатора
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
model.eval()

# Вычисление векторных представлений для всех паттернов
pattern_embeddings = {intent['tag']: np.mean([get_embedding(pattern, tokenizer, model) for pattern in intent['patterns']], axis=0) for intent in data['intents']}

# Функция для нахождения наилучшего ответа (код инференса)
def find_best_response(question, pattern_embeddings, intents, tokenizer, model):
    question_embedding = get_embedding(question, tokenizer, model)
    best_tag, best_distance = None, float('inf')
    for tag, embedding in pattern_embeddings.items():
        distance = cosine(question_embedding, embedding)
        if distance < best_distance:
            best_tag, best_distance = tag, distance
    # Выбор случайного ответа из списка возможных ответов для лучшего тега
    best_responses = next(intent['responses'] for intent in intents if intent['tag'] == best_tag)
    best_response = np.random.choice(best_responses)
    return best_response, 1 - best_distance  # Возвращаем ответ и косинусное сходство

# Пример использования
question = "Как тебя зовут?"
best_response, confidence = find_best_response(question, pattern_embeddings, data['intents'], tokenizer, model)
print(f"Ответ: {best_response} \nВероятность: {confidence:.2f}")


# Создание папки для сохранения модели
if not os.path.exists(MODEL_INFO_DIR):
    os.makedirs(MODEL_INFO_DIR)

# Сохранение модели классификации и токенизатора
classification_model.save_pretrained(os.path.join(MODEL_INFO_DIR, "classification_model"))
tokenizer.save_pretrained(os.path.join(MODEL_INFO_DIR, "tokenizer"))

# Сохранение модели для получения эмбеддингов
embedding_model.save_pretrained(os.path.join(MODEL_INFO_DIR, "embedding_model"))

# Сохранение дополнительных данных
additional_data = {
    "pattern_embeddings": pattern_embeddings,
    "intents_data": intents_data
}
with open(os.path.join(MODEL_INFO_DIR, "additional_data.pkl"), "wb") as f:
    pickle.dump(additional_data, f)