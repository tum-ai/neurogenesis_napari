from typing import TypedDict, List
import numpy as np


class TSegmentation(TypedDict):
    masks: np.ndarray
    centroids: List[List[float]]
    bounding_boxes: List[np.ndarray]
