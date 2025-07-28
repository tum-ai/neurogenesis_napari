import numpy as np
from cellpose.models import Cellpose

# sudo env KMP_DUPLICATE_LIB_OK=TRUE python dapi_cellpose.py
# model = models.CellposeModel(gpu=False, model_type='cyto3') default\


def segment(image: np.ndarray, model: Cellpose) -> np.ndarray:
    imgs = image.copy()
    try:
        pred_masks, _, _ = model.eval(imgs, diameter=None)
    except Exception:  # noqa: BLE001
        pred_masks, _, _, _ = model.eval(imgs, diameter=None)

    return pred_masks
