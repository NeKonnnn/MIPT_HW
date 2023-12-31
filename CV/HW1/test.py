import cv2
import numpy as np

def nothing(x):
    pass

# Создаем окно с трекбарами для настройки диапазонов HSV
cv2.namedWindow('Trackbars')
cv2.createTrackbar('Low H', 'Trackbars', 0, 179, nothing)
cv2.createTrackbar('High H', 'Trackbars', 179, 179, nothing)
cv2.createTrackbar('Low S', 'Trackbars', 0, 255, nothing)
cv2.createTrackbar('High S', 'Trackbars', 255, 255, nothing)
cv2.createTrackbar('Low V', 'Trackbars', 0, 255, nothing)
cv2.createTrackbar('High V', 'Trackbars', 255, 255, nothing)
# Дополнительные трекбары для белого цвета
cv2.createTrackbar('Low S White', 'Trackbars', 0, 255, nothing)
cv2.createTrackbar('High V White', 'Trackbars', 255, 255, nothing)

while True:
    # Загружаем изображение
    img = cv2.imread('C:\\Users\\NeKonn\\MFTI\\CV\\dataset\\train\\imgs\\nemo004.jpg')
    img_blurred = cv2.GaussianBlur(img, (5, 5), 0)
    img_hsv = cv2.cvtColor(img_blurred, cv2.COLOR_BGR2HSV)

    # Получаем текущие позиции трекбаров
    low_h = cv2.getTrackbarPos('Low H', 'Trackbars')
    high_h = cv2.getTrackbarPos('High H', 'Trackbars')
    low_s = cv2.getTrackbarPos('Low S', 'Trackbars')
    high_s = cv2.getTrackbarPos('High S', 'Trackbars')
    low_v = cv2.getTrackbarPos('Low V', 'Trackbars')
    high_v = cv2.getTrackbarPos('High V', 'Trackbars')
    low_s_white = cv2.getTrackbarPos('Low S White', 'Trackbars')
    high_v_white = cv2.getTrackbarPos('High V White', 'Trackbars')

    # Создаем маску для белого цвета
    lower_white = np.array([0, low_s_white, 200])
    upper_white = np.array([179, 255, high_v_white])
    mask_white = cv2.inRange(img_hsv, lower_white, upper_white)

    # Показываем оригинальное изображение и маску
    cv2.imshow('Original', img)
    cv2.imshow('Mask', mask_white)

    key = cv2.waitKey(1)
    if key == 27:  # ESC для выхода
        break

cv2.destroyAllWindows()
