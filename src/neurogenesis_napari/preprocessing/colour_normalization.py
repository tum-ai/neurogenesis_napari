from pathlib import Path

import cv2
import numpy as np
from skimage.exposure import match_histograms


def normalize_colors(grayscale_img: np.ndarray) -> np.ndarray:
    """
    Normalize the color distribution of an input grayscale image to match a reference image using histogram matching.
    The reference image is a 3D image.

    Args:
        grayscale_img (np.ndarray): The input image to normalize. Must be a 2D grayscale image (H, W).

    Returns:
        normalzed (np.ndarray): The color-normalized image as a uint8 BGR array of shape (H, W, 3).
    """
    ref_path = Path(__file__).parent / "good_img.png"
    reference_img = cv2.imread(str(ref_path))
    image2_bgr = cv2.cvtColor(grayscale_img, cv2.COLOR_GRAY2BGR)
    matched = match_histograms(image2_bgr, reference_img)
    normalized = np.clip(matched, 0, 255).astype(np.uint8)
    return normalized
