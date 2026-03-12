import os
import cv2
import numpy as np

RGB_PATH = "data/trene/training/rgb"
MASK_PATH = "data/trene/training/mask"

IMG_SIZE = 256


def load_dataset():

    images = []
    masks = []

    files = sorted(os.listdir(RGB_PATH))

    for file in files:

        rgb_path = os.path.join(RGB_PATH, file)
        mask_path = os.path.join(MASK_PATH, file)

        img = cv2.imread(rgb_path)
        mask = cv2.imread(mask_path, 0)

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))

        img = img / 255.0
        mask = mask / 255.0

        images.append(img)
        masks.append(mask)

    return np.array(images), np.array(masks)