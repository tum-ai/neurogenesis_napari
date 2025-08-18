import pickle
from functools import lru_cache
from typing import List

import cv2
import numpy as np
import torch
from magicgui import magic_factory, magicgui
from napari import Viewer
from napari.layers import Image, Layer, Shapes
from napari.utils.notifications import (
    show_warning,
    show_error,
)
from sklearn.neighbors import NearestCentroid

from neurogenesis_napari._utils import (
    ensure_weights,
    get_gray_img,
    get_weight_path,
    setup_cellpose_log_panel,
)
from neurogenesis_napari.classification.representation_based.vae import (
    VAE,
    generate_latent_representation,
)
from neurogenesis_napari.segmentation.cs import crop_from_bbox, enlarge_bbox
from neurogenesis_napari.widgets.segment import (
    _get_segmentation_layers,
    _segment_async,
)

PALETTE = {
    "Astrocyte": "magenta",
    "Dead Cell": "gray",
    "Neuron": "cyan",
    "OPC": "lime",
}
IDX2LBL = {0: "Astrocyte", 1: "Dead Cell", 2: "Neuron", 3: "OPC"}
CLASSES = list(PALETTE)


@lru_cache
def load_models(vae_wts: str, clf_wts: str) -> tuple[VAE, NearestCentroid]:
    """Load the pretrained models *once* and cache them.

    Args:
        vae_wts (str): Path to the ``.pth`` state‑dict of the VAE.
        clf_wts (str): Path to the pickled scikit‑learn classifier (``.pkl``).

    Returns:
        Tuple containing the VAE and the classifier instance.
    """
    vae = VAE().eval()
    vae.load_state_dict(torch.load(vae_wts, map_location="cpu"))
    with open(clf_wts, "rb") as f:
        clf = pickle.load(f)
    return vae, clf


def classify_patch(patch: np.ndarray, vae: VAE, clf: NearestCentroid) -> str:
    """Predict the cell type of a *single* 4‑channel patch.

    Args:
        patch (np.ndarray): Array of shape (C, 224, 224) with values in [0, 1].
        vae: The pretrained vae.
        clf: A fitted scikit‑learn classifier.

    Returns:
        One of {"Astrocyte", "Dead Cell", "Neuron", "OPC"}.
    """
    z = generate_latent_representation(patch, vae)
    return IDX2LBL[int(clf.predict(z)[0])]


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


def classify(
    DAPI: np.ndarray,
    BF: np.ndarray,
    Tuj1: np.ndarray,
    RFP: np.ndarray,
    bounding_boxes: List[np.ndarray],
) -> List[Layer]:
    """Classify nuclei *polygons* into cell types and return per-class shape layers.

    Args:
        DAPI (np.ndarray): DAPI channel.
        Tuj1 (np.ndarray): β‑III‑tubulin channel.
        RFP (np.ndarray):  RFP channel.
        BF (np.ndarray):   Bright‑field channel.
        bounding_boxes (List[np.ndarray]): List of nucleus polygons in pixel coordinates.

    Returns:
        A list of layers (one per predicted class) containing the corresponding polygons.
    """

    def prep(layer: Image) -> np.ndarray:
        return cv2.normalize(get_gray_img(layer), None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)

    chans = tuple(map(prep, (DAPI, BF, Tuj1, RFP)))

    vae, clf = load_models(
        str(get_weight_path("vae", "TL_FT_bigvae3.pth")),
        str(get_weight_path("classifier", "NearestCentroid.pkl")),
    )

    labels = []

    for bbox in bounding_boxes:
        patch = crop_stack_resize(chans, bbox)
        pred = classify_patch(patch, vae, clf)
        labels.append(pred)

    layer = Shapes(
        data=bounding_boxes,
        shape_type="polygon",
        properties={"label": labels},
        name="Predictions",
        edge_width=4,
        face_color=[0, 0, 0, 0],
        scale=DAPI.scale[-2:],
        translate=DAPI.translate[-2:],
        edge_color="label",
        edge_color_cycle=list(PALETTE.values()),
        text={
            "text": "{label}",
            "size": 5,
            "anchor": "upper_left",
            "translation": [0, 0],
        },
    )

    return [layer]


def _set_label(layer: Shapes, label: str):
    sel = list(layer.selected_data)
    if not sel:
        show_warning("Select one or more cell polygons first.")
        return
    for idx in sel:
        layer.properties["label"][idx] = label
        layer.text.values[idx] = label
    layer.refresh()


def add_label_hotkeys(layer: Shapes):
    @layer.bind_key("1")
    def _1(event=None):
        _set_label(layer, "Astrocyte")

    @layer.bind_key("2")
    def _2(event=None):
        _set_label(layer, "Dead Cell")

    @layer.bind_key("3")
    def _3(event=None):
        _set_label(layer, "Neuron")

    @layer.bind_key("4")
    def _4(event=None):
        _set_label(layer, "OPC")


def attach_edit_widget(viewer: Viewer, layer: Shapes) -> None:
    @magicgui(
        class_label={"widget_type": "ComboBox", "choices": CLASSES},
        call_button="Apply",
        persist=True,
        auto_call=False,
    )
    def edit_label(class_label: str = "Neuron"):
        _set_label(layer, class_label)

    # keep dropdown synced with current selection
    def _sync_dropdown(event=None):
        sel = list(layer.selected_data)
        if sel:
            edit_label.class_label.value = layer.properties["label"][sel[0]]

    layer.events.connect(_sync_dropdown)
    viewer.window.add_dock_widget(edit_label, area="right", name="Edit cell label")

    add_label_hotkeys(layer)


@magic_factory(
    call_button="Segment + Classify",
)
def segment_and_classify_widget(
    viewer: Viewer,
    DAPI: Image | None = None,
    Tuj1: Image | None = None,
    RFP: Image | None = None,
    BF: Image | None = None,
    reuse_cached_segmentation: bool = True,
    gpu: bool = False,
    model_type: str = "cyto3",
) -> List[Layer]:
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
        RFP:  RFP channel.
        BF:   Bright‑field channel.
        reuse_cached_segmentation (bool: = True): Whether to reuse the already created segmentation layers.

    Returns:
        A list of :class:`napari.layers.Shapes` layers, one per predicted cell class.
    """
    missing = []
    for name, image in zip(["DAPI", "Tuj1", "RFP", "BF"], [DAPI, Tuj1, RFP, BF], strict=False):
        if image is None:
            missing += [name]

    if missing != []:
        show_warning(f"No {', '.join(missing)} image layer(s) selected. Pick one and retry.")
        return []

    # Ensure model weights are downloaded (runs only once)
    try:
        ensure_weights()
    except Exception as e:  # noqa: BLE001
        show_warning(f"Failed to download model weights: {e}")
        return []

    seg = DAPI.metadata.get("segmentation")

    if reuse_cached_segmentation and seg is not None:
        bounding_boxes = seg["bounding_boxes"]

        if not bounding_boxes:
            show_warning("No nuclei detected → nothing to classify.")
            return []

        classification_layers = classify(DAPI, BF, Tuj1, RFP, bounding_boxes)
        return classification_layers  # User already has segmentation layers

    dapi_gray = get_gray_img(DAPI)

    dock_panel_key = "segment_classify_widget"
    setup_cellpose_log_panel(
        viewer, panel_key=dock_panel_key, dock_title="Cellpose logs - Segment + Classify"
    )
    worker = _segment_async(dapi_gray, gpu, model_type, dock_panel_key)

    def _on_done(result) -> None:
        pred_masks, centroids, bounding_boxes = result

        if not bounding_boxes:
            show_warning("No nuclei detected → nothing to classify.")
            return []

        DAPI.metadata["segmentation"] = {
            "masks": pred_masks,
            "centroids": centroids,
            "bounding_boxes": bounding_boxes,
        }

        segmentation_layers = _get_segmentation_layers(DAPI, pred_masks, centroids, bounding_boxes)
        classification_layers = classify(DAPI, BF, Tuj1, RFP, bounding_boxes)
        for layer in segmentation_layers + classification_layers:
            viewer.add_layer(layer)

        for layer in classification_layers:
            if isinstance(layer, Shapes) and layer.name == "Cells":
                attach_edit_widget(viewer, layer)

    worker.returned.connect(_on_done)
    worker.errored.connect(lambda e: show_error(f"Cellpose failed: {e}"))
    worker.start()

    return []
