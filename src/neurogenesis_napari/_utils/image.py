from typing import List
import cv2
import numpy as np
import napari
from napari.layers import Image
from skimage import img_as_float32
from magicgui.widgets import ComboBox


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


def get_image_choices_with_default(widget: ComboBox, patterns: List[str]) -> List[Image]:
    """Return image layers with the default choice listed first, so the widget preselects it.

    Args:
        widget (ComboBox): When used with napari in a ``@magic_factory``, the widget is passed as the first argument.
        patterns (List[str]): Strings to match against layer names. Matching is performed using a simple substring check.

    Returns:
        List[Image]: The list of image layers, with the default layer placed first.
    """
    viewer = getattr(getattr(widget, "parent", None), "viewer", None) or napari.current_viewer()
    if viewer is None:
        return []

    # what’s already selected in the other inputs?
    parent = getattr(widget, "parent", None)
    used = set()
    if parent is not None:
        for name in ("DAPI", "BF", "Tuj1", "RFP"):
            w = getattr(parent, name, None)
            if w is widget or w is None:
                continue
            val = getattr(w, "value", None)
            if isinstance(val, Image):
                used.add(id(val))

    # only keep unused layers
    imgs = [lyr for lyr in viewer.layers if isinstance(lyr, Image)]
    unused_imgs = [lyr for lyr in imgs if id(lyr) not in used]
    if not unused_imgs:
        return []

    # normalize patterns to lowercase for comparison
    lowered = [p.lower() for p in patterns]

    # find the first image layer whose name contains any of the patterns
    default = next(
        (lyr for lyr in unused_imgs if any(p in lyr.name.lower() for p in lowered)),
        unused_imgs[0],  # fallback to the first layer if no match
    )

    # put default first, then others no matter if used or not
    ordered = [default] + [lyr for lyr in imgs if lyr is not default]
    return ordered
