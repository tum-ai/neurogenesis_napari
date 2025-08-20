import pickle
from functools import lru_cache
from typing import List

import cv2
import numpy as np
import torch
from magicgui import magic_factory
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
from neurogenesis_napari.widgets._edit_prediction import attach_edit_widget

PALETTE = {
    "Astrocyte": "magenta",
    "Dead Cell": "gray",
    "Neuron": "cyan",
    "OPC": "lime",
}
IDX2LBL = {0: "Astrocyte", 1: "Dead Cell", 2: "Neuron", 3: "OPC"}
PREDICTION_LAYER_NAME = "Predictions"
CLASSIFY_WIDGET_PANEL_KEY = "segment_classify_widget"


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
) -> Layer:
    """Classify nuclei *polygons* into cell types and return per-class shape layers.

    Args:
        DAPI (np.ndarray): DAPI channel.
        Tuj1 (np.ndarray): β‑III‑tubulin channel.
        RFP (np.ndarray): RFP channel.
        BF (np.ndarray): Bright‑field channel.
        bounding_boxes (List[np.ndarray]): List of nucleus polygons in pixel coordinates.

    Returns:
        A layer containing all prediction polygons.
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
        name=PREDICTION_LAYER_NAME,
        edge_width=4,
        face_color=[0, 0, 0, 0],
        scale=DAPI.scale[-2:],
        translate=DAPI.translate[-2:],
        edge_color="label",
        # TODO: non-determinstic color assignment
        edge_color_cycle=list(PALETTE.values()),
        text={
            "text": "{label}",
            "size": 5,
            "anchor": "upper_left",
            "translation": [0, 0],
        },
    )

    return layer


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

    seg = DAPI.metadata.get("segmentation")

    if reuse_cached_segmentation and seg is not None:
        bounding_boxes = seg["bounding_boxes"]

        if not bounding_boxes:
            show_warning("No nuclei detected → nothing to classify.")
            return None

        prediction_layer = classify(DAPI, BF, Tuj1, RFP, bounding_boxes)
        viewer.add_layer(prediction_layer)  # User already has segmentation layers
        attach_edit_widget(viewer, prediction_layer, IDX2LBL)

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

        segmentation_layers = _get_segmentation_layers(DAPI, pred_masks, centroids, bounding_boxes)
        prediction_layer = classify(DAPI, BF, Tuj1, RFP, bounding_boxes)
        for layer in segmentation_layers + [prediction_layer]:
            viewer.add_layer(layer)
        attach_edit_widget(viewer, prediction_layer, IDX2LBL)

    worker.returned.connect(_on_done)
    worker.errored.connect(lambda e: show_error(f"Cellpose failed: {e}"))
    worker.start()

    return None
