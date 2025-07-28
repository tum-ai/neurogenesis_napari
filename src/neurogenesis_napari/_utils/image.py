import cv2
import numpy as np
from napari.layers import Image
from skimage import img_as_float32


def get_gray_img(image_layer: Image) -> np.ndarray:
    img = img_as_float32(image_layer.data)
    # Remove all dimensions of size 1 (especially relevant for czi files)
    img_gray = np.squeeze(img)
    if img_gray.ndim == 3:
        # if RGBA, drop alpha channel
        if img_gray.shape[2] == 4:
            img_gray = img_gray[..., :3]
        # convert rgb to grayscale
        img_gray = cv2.cvtColor(img_gray, cv2.COLOR_RGB2GRAY)
    return img_gray
