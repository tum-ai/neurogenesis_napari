from typing import TypedDict, List
import numpy as np


TMasks = np.ndarray
TCentroids = List[List[float]]
TBoundingBoxes = List[np.ndarray]


class TSegmentation(TypedDict):
    masks: TMasks
    centroids: TCentroids
    bounding_boxes: TBoundingBoxes