from collections.abc import Sequence

import cv2
import numpy as np

from neurogenesis_napari.segmentation.cs import crop_from_bbox, enlarge_bbox


def bbox_to_rectangle(bbox: Sequence[int]) -> np.ndarray:
    """Convert (min_row, min_col, max_row, max_col) → 4-vertex rectangle poly."""
    r0, c0, r1, c1 = bbox
    return np.array(
        [[r0, c0], [r0, c1], [r1, c1], [r1, c0]],
        dtype=float,
    )


def crop_stack_resize(
    channels: tuple[np.ndarray, ...], bbox: np.ndarray, out_size: int = 224
) -> np.ndarray:
    """Extract the same enlarged bounding‑box from each channel and resize.

    Args:
        channels (tuple[np.ndarray,...]): Sequence of single‑channel images with identical spatial shape.
        bbox (np.ndarray): Polygon representing the nucleus bounding box.
        out_size (int: = 224): Final square size after resizing.

    Returns:
        Stacked patch with shape (len(channels), out_size, out_size).
    """
    flat = [
        bbox[:, 0].min(),
        bbox[:, 1].min(),
        bbox[:, 0].max(),
        bbox[:, 1].max(),
    ]
    big = enlarge_bbox(channels[0].shape, flat, 2)
    crops = [
        cv2.resize(crop_from_bbox(ch, big), (out_size, out_size), cv2.INTER_AREA) for ch in channels
    ]
    return np.stack(crops, -1).transpose(2, 0, 1)
