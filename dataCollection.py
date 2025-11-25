
'''import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math

import time
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
offset = 20
imgSize = 300
folder = "Data/salem"
counter = 0
while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
        imgCropShape = imgCrop.shape
        aspectRatio = h / w
        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize
        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)
    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg',imgWhite)
        print(counter)
'''



import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

# --- НАСТРОЙКИ ---
folder = "Data/qalaisyn"     # Папка, куда сохранять фото
imgSize = 300             # Размер выходного изображения
offset = 20               # Отступ вокруг руки

cap = cv2.VideoCapture(1)
detector = HandDetector(maxHands=1)
counter = 0

while True:
    success, img = cap.read()
    if not success:
        print("Ошибка: камера не работает")
        break

    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        # --- БЕЗОПАСНАЯ ОБРЕЗКА КАРТИНКИ ---
        y1 = max(0, y - offset)
        y2 = min(img.shape[0], y + h + offset)
        x1 = max(0, x - offset)
        x2 = min(img.shape[1], x + w + offset)

        imgCrop = img[y1:y2, x1:x2]

        # если область пустая — пропускаем
        if imgCrop.size == 0:
            cv2.imshow("Image", img)
            key = cv2.waitKey(1)
            continue

        aspectRatio = h / w

        # --- ВЫРАВНИВАНИЕ ПОД БЕЛЫЙ ФОН ---
        if aspectRatio > 1:
            # вертикальная рука
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize

        else:
            # горизонтальная рука
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize

        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)

    # --- СОХРАНЕНИЕ ИЗОБРАЖЕНИЯ ---
    if key == ord("s"):
        counter += 1
        file_path = f'{folder}/Image_{time.time()}.jpg'
        cv2.imwrite(file_path, imgWhite)
        print(f"Сохранено: {file_path}  |  Всего: {counter}")

    # выход
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
