from typing import List, Iterable, Callable, Sequence, Tuple
import cv2
import numpy as np

import napari
from napari.layers import Image, Layer
from napari.utils.events import Event

from skimage import img_as_float32
from magicgui.widgets import ComboBox, FunctionGui
from qtpy.QtCore import QTimer


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


def is_pattern_match(layer: Layer, patterns: List[str]) -> bool:
    """Returns true if the layer is an image and its name matches any of the provided patterns"""
    return isinstance(layer, Image) and any(pattern in layer.name.lower() for pattern in patterns)


def wire_layer_comboboxes_autorefresh(
    w: FunctionGui,
    viewer: napari.Viewer,
    combos: Sequence[Tuple[ComboBox, Iterable[str]]],
) -> Callable[[], None]:
    """Keep multiple ComboBoxes of image layers in sync with the viewer and select sensible defaults.

    Args:
        w (FunctionGui): The widget created by ``magic_factory`` that contains the ComboBoxes.
        viewer (Viewer): The napari viewer whose layers are monitored.
        combos (Sequence[Tuple[ComboBox, Iterable[str]]]): Each entry is ``(combobox, patterns)``,
            where ``patterns`` are substrings used to detect preferred layers (case-insensitive).

    Returns:
        Callable[[], None]: A disposer that disconnects all event listeners when called.
    """
    combos = [(cb, tuple(p.lower() for p in patterns)) for cb, patterns in combos]

    def _refresh(cb: ComboBox, patterns: tuple[str, ...]) -> None:
        cb.reset_choices()
        choices = cb.choices
        if not choices:
            cb.choices = []
            cb.value = None

    def _on_inserted(e: Event) -> None:
        layer = getattr(e, "value", None)
        if layer is None:
            return
        for cb, patterns in combos:
            if is_pattern_match(layer, patterns) and not cb.value:
                cb.reset_choices()
                cb.value = layer
            else:
                _refresh(cb, patterns)

    events = viewer.layers.events
    events.inserted.connect(_on_inserted)
    events.removed.connect(lambda e: [_refresh(cb, p) for cb, p in combos])
    events.reordered.connect(lambda e: [_refresh(cb, p) for cb, p in combos])
    events.changed.connect(lambda e: [_refresh(cb, p) for cb, p in combos])

    def _initial() -> None:
        for cb, patterns in combos:
            _refresh(cb, patterns)
            preferred = next(
                (lyr for lyr in viewer.layers if is_pattern_match(lyr, patterns)), None
            )
            if preferred is not None:
                cb.value = preferred

    QTimer.singleShot(0, _initial)

    def dispose() -> None:
        try:
            events.inserted.disconnect(_on_inserted)
            events.removed.disconnect()
            events.reordered.disconnect()
            events.changed.disconnect()
        except Exception:
            pass

    try:
        w.native.destroyed.connect(lambda *_: dispose())
    except Exception:
        pass

    return dispose


def image_layer_choices(widget: FunctionGui) -> list[Image]:
    viewer = getattr(getattr(widget, "parent", None), "viewer", None) or napari.current_viewer()
    if viewer is None:
        return []
    return [lyr for lyr in viewer.layers if isinstance(lyr, Image)]
