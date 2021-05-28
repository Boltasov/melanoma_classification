import cv2
from PIL import Image
import numpy as np
import tensorflow as tf


def convert_photo(photo_name, RESIZE, sigmaX=10):
    # Преобразует полученные изображения в формат, подходящий для используемой нейросети
    img = np.asarray(Image.open(photo_name).convert("RGB"))
    img = cv2.resize(img, (RESIZE, RESIZE))
    img = img.reshape(1, 224, 224, 3)
    return img



