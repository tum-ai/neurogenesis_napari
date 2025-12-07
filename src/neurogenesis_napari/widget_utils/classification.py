import pickle
from functools import lru_cache
from typing import List

import cv2
import numpy as np
import torch
from napari.layers import Image, Layer, Shapes
from sklearn.neighbors import NearestCentroid

from neurogenesis_napari.widget_utils import (
    get_gray_img,
    crop_stack_resize,
    get_weight_path,
)
from neurogenesis_napari.classification.representation_based.vae import (
    VAE,
    generate_latent_representation,
)
from neurogenesis_napari.settings import (
    PALETTE,
    IDX2LBL,
)

PREDICTION_LAYER_NAME = "Predictions"


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
