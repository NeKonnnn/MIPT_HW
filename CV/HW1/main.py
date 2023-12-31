# Импорт библиотек
import cv2
import numpy as np
import json
import os
from utils.compute_iou import compute_ious
import matplotlib.pyplot as plt

def segment_fish(img):
    # Применение билатеральной фильтрации для снижения шума, сохраняя при этом края
    img_filtered = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)

    # Повышение резкости изображения
    kernel_sharpening = np.array([[-1, -1, -1], 
                                  [-1, 9, -1], 
                                  [-1, -1, -1]])
    img_sharpened = cv2.filter2D(img_filtered, -1, kernel_sharpening)

    # Конвертация в HSV
    img_hsv = cv2.cvtColor(img_sharpened, cv2.COLOR_BGR2HSV)

    # Разделение на компоненты и коррекция яркости компонента V
    h, s, v = cv2.split(img_hsv)
    v = cv2.equalizeHist(v)

    # Объединение обратно в одно изображение HSV
    img_hsv_equalized = cv2.merge([h, s, v])
    img_hsv_output = cv2.cvtColor(img_hsv_equalized, cv2.COLOR_HSV2BGR)

    # Фильтр Гаусса для уменьшения шума
    img_blurred = cv2.GaussianBlur(img_hsv_output, (5, 5), 0)

    # Повторная конвертация в HSV после коррекции
    img_hsv_corrected = cv2.cvtColor(img_blurred, cv2.COLOR_BGR2HSV)

    # Настройка диапазонов для цветов
    lower_orange = np.array([1, 190, 150])
    upper_orange = np.array([30, 255, 255])
    lower_white = np.array([60, 20, 200])
    upper_white = np.array([145, 150, 255])

    # Создание маски для оранжевого и белого цветов
    mask_orange = cv2.inRange(img_hsv_corrected, lower_orange, upper_orange)
    mask_white = cv2.inRange(img_hsv_corrected, lower_white, upper_white)

    # Объединение цветовых масок
    mask_color = cv2.bitwise_or(mask_orange, mask_white)

    # Добавление детекции краев
    edges = cv2.Canny(img, 100, 200)
    combined_mask = cv2.bitwise_or(mask_color, edges)

    # Морфологические операции с измененными параметрами
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask_opened = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask_closed = cv2.morphologyEx(mask_opened, cv2.MORPH_CLOSE, kernel, iterations=2)
    final_mask = cv2.dilate(mask_closed, kernel, iterations=2)

    return final_mask

def process_images_and_compute_iou(image_paths, mask_paths):
    img_name2mask = {}
    for img_path, mask_path in zip(image_paths, mask_paths):
        # Чтение изображения
        img = cv2.imread(img_path)

        # Сегментация рыбы на изображении
        pred_mask = segment_fish(img)

        # Добавление предсказанной маски в словарь
        img_name = os.path.basename(img_path)
        img_name2mask[img_name] = pred_mask

    # Расчет IoU для всех изображений
    average_iou = compute_ious(img_name2mask, masks_folder)
    return average_iou

if __name__ == "__main__":
     # Определение пути к текущей директории
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Путь к JSON-файлу с аннотациями
    json_path = os.path.join(current_dir, 'dataset', 'train.json')

    # Пути к папкам с изображениями и масками
    images_folder = os.path.join(current_dir, 'dataset', 'train', 'imgs')
    masks_folder = os.path.join(current_dir, 'dataset', 'train', 'masks')

    # Загрузка аннотаций
    with open(json_path, 'r') as file:
        train_data = json.load(file)

    # Подготовка списков путей к изображениям и маскам
    image_paths = [os.path.join(images_folder, img_file) for img_file in train_data]
    mask_paths = [os.path.join(masks_folder, mask_file.replace('.jpg', '.png')) for mask_file in train_data]

    # Вычисление среднего значения IoU для сегментированных изображений
    average_iou = process_images_and_compute_iou(image_paths, mask_paths)
    print(f'Метрика IoU на тренировочном датасете: {round(average_iou, 2)}')


# Функция для наложения маски на изображение
def apply_mask(image, mask):
    return np.where(mask[:, :, None], image, np.zeros_like(image))

# Визуализация результатов и вывод IoU
for img_path, mask_path in zip(image_paths, mask_paths):
    img = cv2.imread(img_path)
    true_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    pred_mask = segment_fish(img)

    # Вычисление IoU для текущего изображения
    iou = compute_ious({os.path.basename(img_path): pred_mask}, masks_folder)
    print(f'IoU для {os.path.basename(img_path)}: {round(iou, 2)}')

    # Наложение маски на изображение
    masked_img = apply_mask(img, pred_mask)

    # Визуализация
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(img[:, :, ::-1])  # Конвертация BGR в RGB
    plt.title('Оригинальное изображение')
    plt.subplot(1, 2, 2)
    plt.imshow(masked_img[:, :, ::-1])
    plt.title('Сегментированное изображение')
    plt.show()