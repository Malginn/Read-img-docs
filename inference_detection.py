ё  # идет на python >=3.7 and <=3.11
import os

import cv2
import numpy as np
from sklearn.cluster import KMeans
from ultralytics import YOLO

IMGSZ = (1120, 1280)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# путь до папки с классифицированными изображениями
path_to_imgs = "../data_gagarin/data/"


def enhance_image(image):
    r, g, b = cv2.split(image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # 2

    enhanced_r = clahe.apply(r)
    enhanced_g = clahe.apply(g)
    enhanced_b = clahe.apply(b)

    enhanced_image = cv2.merge((enhanced_r, enhanced_g, enhanced_b))
    enhanced_image = cv2.convertScaleAbs(enhanced_image, alpha=1.5, beta=0)

    return enhanced_image


# 3 канала
def rework(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb = enhance_image(image_rgb)
    pixels = image_rgb.reshape((-1, 3))
    n_colors = 2
    kmeans = KMeans(n_clusters=n_colors)
    kmeans.fit(pixels)
    main_colors = kmeans.cluster_centers_.astype(int)
    most_common_color = main_colors[np.argmin(np.bincount(kmeans.labels_))]

    treshold_post_class = 70

    mask = np.any(np.abs(pixels - most_common_color) > treshold_post_class, axis=1)
    result = np.where(mask.reshape(image.shape[:2]), 255, 0).astype(np.uint8)
    result = cv2.bitwise_not(result)

    _, result = cv2.threshold(result, 100, 255, cv2.THRESH_BINARY)  # 180
    return result


def cropp_imgs(preds):
    """
    Функция для вырезания предсказанных Bounding boxes
    из исходных изображений

    input -> предсказания
    output -> папка results с обрезанными картинками
    """
    for iter in range(len(preds)):
        img = preds[iter].orig_img
        try:
            x, y, x_1, y_1 = [
                int(i) for i in list(preds[iter].boxes.xyxy[0].to("cpu").numpy())
            ]

            roi_color = img[y:y_1, x:x_1]

            # image = rework(roi_color)
            # обрезанная картинка в переменной roi_color

            name = preds[iter].path.split("/")[-1]
            cv2.imwrite(f"./results/{name}.jpg", roi_color)
        except:
            continue


# инициализируем модель и загружаем веса
model = YOLO("path/to/model.pt")
# делаем предсказания
preds = model.predict(
    [path_to_imgs + i for i in os.listdir(path_to_imgs)], save=True, imgsz=IMGSZ
)
# вырезаем и сохраняем картинки
cropp_imgs(preds)
