import csv
import numpy as np
from magicgui import magicgui
from qtpy import QtWidgets
from napari.layers import Shapes
from napari import Viewer
from napari.utils.notifications import show_warning, show_error, show_info


def save_layer_bboxes_csv(layer: Shapes) -> None:
    labels = list(layer.properties.get("label", []))
    if len(labels) != len(layer.data):
        show_warning("Prediction layer is missing labels or counts don’t match polygons!")
        return

    path, _ = QtWidgets.QFileDialog.getSaveFileName(
        None,
        "Save bounding boxes and labels",
        "bboxes_labels.csv",
    )
    if not path:
        return

    if not path.lower().endswith(".csv"):
        path += ".csv"

    scale = np.asarray(layer.scale[:2], dtype=float)
    translate = np.asarray(layer.translate[:2], dtype=float)

    rows = []
    for i, (poly, lab) in enumerate(zip(layer.data, labels)):
        pix = (poly - translate) / scale  # (y, x)
        y_min, x_min = pix.min(axis=0)
        y_max, x_max = pix.max(axis=0)
        rows.append(
            [
                i,
                str(lab),
                float(x_min),
                float(y_min),
                float(x_max),
                float(y_max),
                float(x_max - x_min),
                float(y_max - y_min),
            ]
        )

    try:
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "label", "x_min", "y_min", "x_max", "y_max", "width", "height"])
            w.writerows(rows)
        show_info(f"Saved to:\n{path}.")
    except Exception as e:
        show_error(f"Failed to save CSV: {e}")


def attach_saver_dock(viewer: Viewer, layer: Shapes) -> None:
    """Attach a label-saving widget to the viewer.

    Provides a docked button for saving class labels
    with the correspoding bounding boxes.

    Args:
        viewer (Viewer): Napari viewer instance.
        layer (Shapes): Shapes layer containing cell polygons with labels.

    Returns:
        None
    """

    @magicgui(
        call_button="Save labels as CSV",
        persist=True,
        auto_call=False,
    )
    def save_file() -> None:
        save_layer_bboxes_csv(layer)

    viewer.window.add_dock_widget(save_file, area="right", name="Save cell labels")
