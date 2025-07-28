from collections.abc import Sequence

import numpy as np


def bbox_to_rectangle(bbox: Sequence[int]) -> np.ndarray:
    """Convert (min_row, min_col, max_row, max_col) → 4-vertex rectangle poly."""
    r0, c0, r1, c1 = bbox
    return np.array(
        [[r0, c0], [r0, c1], [r1, c1], [r1, c0]],
        dtype=float,
    )
