from functools import partial

from magicgui import magic_factory
from napari import Viewer
import napari
from napari.layers import Image
from napari.utils.notifications import (
    show_warning,
    show_error,
)
from neurogenesis_napari.widget_utils import (
    ensure_weights,
    get_gray_img,
    setup_cellpose_log_panel,
    wire_layer_comboboxes_autorefresh,
    image_layer_choices,
    start_progress,
    close_progress,
    load_segmentation,
    save_segmentation,
    get_segmentation_layers,
    attach_saver_dock,
    attach_edit_widget,
    attach_inspect_widget,
    classify,
)
from neurogenesis_napari.widgets.segment import (
    _segment_async,
)
from neurogenesis_napari.settings import IDX2LBL


CLASSIFY_WIDGET_PANEL_KEY = "segment_classify_widget"


def _segment_and_classify_widget_impl(
    viewer: Viewer,
    DAPI: Image,
    Tuj1: Image,
    RFP: Image,
    BF: Image,
    reuse_cached_segmentation: bool = True,
    gpu: bool = False,
    model_type: str = "cyto3",
) -> None:
    """Segment nuclei and classify every detected cell in one click.

    Workflow
    --------
    1. **Weight check** – abort early if the required model files are missing.
    2. **Segmentation** – run (or reuse cached) Cellpose‑based segmentation on
       the DAPI channel to obtain bounding‑boxes.
    3. **Patch extraction** – build a 4‑channel patch around each bounding box.
    4. **Prediction** – embed with VAE → classify with nearest‑centroid.
    5. **Visualisation** – add a Shapes layer per class (skipping empties).

    Args:
        DAPI: DAPI channel.
        Tuj1: β‑III‑tubulin channel.
        RFP: RFP channel.
        BF: Bright‑field channel.
        reuse_cached_segmentation (bool: = True): Whether to reuse the already created segmentation layers.

    Returns:
        None
    """
    missing = []
    for name, image in zip(["DAPI", "Tuj1", "RFP", "BF"], [DAPI, Tuj1, RFP, BF], strict=False):
        if image is None:
            missing += [name]

    if missing != []:
        show_warning(f"No {', '.join(missing)} image layer(s) selected. Pick one and retry.")
        return None

    # Ensure model weights are downloaded (runs only once)
    try:
        ensure_weights()
    except Exception as e:  # noqa: BLE001
        show_warning(f"Failed to download model weights: {e}")
        return None

    seg = None

    if reuse_cached_segmentation:
        # check memory
        seg = DAPI.metadata.get("segmentation")

        # check disk
        if seg is None:
            seg = load_segmentation(DAPI, gpu, model_type)
            if seg is not None:
                DAPI.metadata["segmentation"] = seg

    if seg is not None:
        bounding_boxes = seg["bounding_boxes"]

        if not bounding_boxes:
            show_warning("No nuclei detected → nothing to classify.")
            return None

        prediction_layer = classify(DAPI, BF, Tuj1, RFP, bounding_boxes)
        viewer.add_layer(prediction_layer)  # User already has segmentation layers
        attach_edit_widget(viewer, prediction_layer, IDX2LBL)
        attach_inspect_widget(viewer, prediction_layer, IDX2LBL, DAPI, BF, Tuj1, RFP)
        attach_saver_dock(viewer, prediction_layer)

        return None

    dapi_gray = get_gray_img(DAPI)

    setup_cellpose_log_panel(
        viewer, panel_key=CLASSIFY_WIDGET_PANEL_KEY, dock_title="Cellpose logs - Segment + Classify"
    )
    worker = _segment_async(dapi_gray, CLASSIFY_WIDGET_PANEL_KEY, gpu, model_type)

    def _on_done(result) -> None:
        pred_masks, centroids, bounding_boxes = result

        if not bounding_boxes:
            show_warning("No nuclei detected → nothing to classify.")
            return None

        DAPI.metadata["segmentation"] = {
            "masks": pred_masks,
            "centroids": centroids,
            "bounding_boxes": bounding_boxes,
        }

        save_segmentation(DAPI, pred_masks, centroids, bounding_boxes, gpu, model_type)

        segmentation_layers = get_segmentation_layers(DAPI, pred_masks, centroids, bounding_boxes)
        prediction_layer = classify(DAPI, BF, Tuj1, RFP, bounding_boxes)
        for layer in segmentation_layers + [prediction_layer]:
            viewer.add_layer(layer)
        attach_edit_widget(viewer, prediction_layer, IDX2LBL)
        attach_inspect_widget(viewer, prediction_layer, IDX2LBL, DAPI, BF, Tuj1, RFP)
        attach_saver_dock(viewer, prediction_layer)

    pbar = {"obj": None}

    worker.started.connect(partial(start_progress, pbar))
    worker.returned.connect(_on_done)
    worker.errored.connect(lambda e: show_error(f"Cellpose failed: {e}"))
    worker.finished.connect(partial(close_progress, pbar))

    worker.start()
    return None


_factory = magic_factory(
    call_button="Segment + Classify",
    DAPI={
        "widget_type": "ComboBox",
        "choices": image_layer_choices,
        "nullable": True,
    },
    BF={
        "widget_type": "ComboBox",
        "choices": image_layer_choices,
        "nullable": True,
    },
    Tuj1={
        "widget_type": "ComboBox",
        "choices": image_layer_choices,
        "nullable": True,
    },
    RFP={
        "widget_type": "ComboBox",
        "choices": image_layer_choices,
        "nullable": True,
    },
)(_segment_and_classify_widget_impl)


def segment_and_classify_widget():
    w = _factory()
    _ = wire_layer_comboboxes_autorefresh(
        w,
        viewer=napari.current_viewer(),
        combos=[
            (w.DAPI, ["dapi"]),
            (w.BF, ["bf", "bright", "brightfield"]),
            (w.Tuj1, ["Tuj1"]),
            (w.RFP, ["RFP"]),
        ],
    )
    return w
