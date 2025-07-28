import math
from collections.abc import Sequence

import numpy as np


def crop_from_bbox(
    img: np.ndarray, bbox: tuple[int, int, int, int], copy: bool = True
) -> np.ndarray:
    min_r, min_c, max_r, max_c = map(int, bbox)

    #  Clamp coords to image bounds in case they exceed
    min_r = max(min_r, 0)
    min_c = max(min_c, 0)
    max_r = min(max_r, img.shape[0])
    max_c = min(max_c, img.shape[1])

    crop = img[min_r:max_r, min_c:max_c]
    return crop.copy() if copy else crop


def enlarge_bbox(
    img_shape: tuple[int, int],  # (H, W)            ── image size
    bbox: Sequence[int],  # (min_r, min_c, max_r, max_c)
    x: float,
) -> tuple[int, int, int, int]:
    if x <= 0:
        raise ValueError("Scale factor x must be positive.")

    H, W = img_shape[:2]
    min_r, min_c, max_r, max_c = map(float, bbox)

    # Current size and centre
    h = max_r - min_r
    w = max_c - min_c
    c_r = min_r + h / 2.0
    c_c = min_c + w / 2.0

    # New half-size after scaling
    half_h = (h * x) / 2.0
    half_w = (w * x) / 2.0

    # Proposed extents
    new_min_r = math.floor(c_r - half_h)
    new_max_r = math.ceil(c_r + half_h)
    new_min_c = math.floor(c_c - half_w)
    new_max_c = math.ceil(c_c + half_w)

    # Clamp to image bounds
    new_min_r = max(new_min_r, 0)
    new_min_c = max(new_min_c, 0)
    new_max_r = min(new_max_r, H)
    new_max_c = min(new_max_c, W)

    # Guarantee at least one pixel in each dimension
    if new_max_r <= new_min_r:
        new_max_r = min(new_min_r + 1, H)
    if new_max_c <= new_min_c:
        new_max_c = min(new_min_c + 1, W)

    return int(new_min_r), int(new_min_c), int(new_max_r), int(new_max_c)
