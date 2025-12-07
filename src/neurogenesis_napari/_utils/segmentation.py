from typing import Tuple, Optional
from pathlib import Path
import numpy as np
import json
import hashlib
from napari.layers import Image
from napari.utils.notifications import show_info

from neurogenesis_napari._utils.typing import (
    TMasks,
    TCentroids,
    TBoundingBoxes,
    TSegmentation,
)


SEGMENT_WIDGET_PANEL_KEY = "segment_widget"


def _get_image_hash(image_data: np.ndarray) -> str:
    """Generate a hash of the image data to detect changes."""
    # Sample the image for speed (every 10th pixel)
    sample = image_data[::10, ::10] if image_data.ndim >= 2 else image_data
    return hashlib.md5(sample.tobytes()).hexdigest()[:16]



def _get_sidecar_paths(
    image: Image,
) -> Tuple[Optional[Path], Optional[Path]]:
    source = image.source.path

    if not source:
        return None, None

    source_path = Path(source)
    json_path = source_path.parent / f"{source_path.name}.seg.json"
    masks_path = source_path.parent / f"{source_path.name}.masks.npz"

    return json_path, masks_path


def load_segmentation(
    image: Image,
    gpu: bool,
    model_type: str,
) -> Optional[TSegmentation]:

    json_path, masks_path = _get_sidecar_paths(image)
    
    if not json_path or not masks_path:
        return None

    if not json_path.exists() or not masks_path.exists():
        return None

    try:
        with open(json_path, "r") as f:
            metadata = json.load(f)

        current_hash = _get_image_hash(image.data)
        if metadata.get("image_hash") != current_hash:
            return None
        
        params = metadata.get("parameters", {})
        if params.get("gpu") != gpu or params.get("model_type") != model_type:
            return None

        data = np.load(masks_path, allow_pickle=True)
        masks = data["masks"]
        bboxes = list(data['bboxes'])

        show_info(f"Loaded segmentation from cache ({metadata['num_cells']} cells)")

        segmentation : TSegmentation = {
            "masks": masks,
            "centroids": metadata["centroids"],
            "bounding_boxes": bboxes,
        }

        return segmentation
    except:
        show_info(f"Failed to load segmentation.")
        return None


def save_segmentation(
    image: Image,
    masks: TMasks,
    centroids: TCentroids,
    bounding_boxes: TBoundingBoxes,
    gpu: bool,
    model_type: str,
) -> bool:
    json_path, masks_path = _get_sidecar_paths(image)
    if not json_path or not masks_path:
        return False
    
    try:
        np.savez_compressed(
            masks_path,
            masks=masks,
            bboxes=np.array(bounding_boxes, dtype=object)
        )

        metadata = {
            "image_hash": _get_image_hash(image.data),
            "image_shape": list(np.squeeze(image.data).shape),
            "centroids": centroids,
            "num_cells": len(centroids),
            "parameters": {
                "gpu": gpu,
                "model_type": model_type,
            },
            "version": "1.0"
        }

        with open(json_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        show_info(f"Segmentation cached to {json_path.name}.")
        return True

    except:
        show_info(f"Failed to save segmentation to {json_path.name}.")
        return False
