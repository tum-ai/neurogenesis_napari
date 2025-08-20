from collections.abc import Callable
from typing import Union
from unittest.mock import patch

import numpy as np
import pytest
from napari.layers import Image, Layer

from neurogenesis_napari.widgets import (
    normalize_and_denoise_widget,
    segment_and_classify_widget,
    segment_widget,
)

NONE_CASES = [
    (
        normalize_and_denoise_widget,
        "neurogenesis_napari.widgets.normalize_and_denoise.show_warning",
        {"BF": None},
        "No BF image layer selected. Pick one and retry.",
        None,
    ),
    (
        segment_widget,
        "neurogenesis_napari.widgets.segment.show_warning",
        {"DAPI": None},
        "No DAPI image layer selected. Pick one and retry.",
        None,
    ),
    (
        segment_and_classify_widget,
        "neurogenesis_napari.widgets.segment_and_classify.show_warning",
        {"DAPI": None, "Tuj1": None, "RFP": None, "BF": None},
        "No DAPI, Tuj1, RFP, BF image layer(s) selected. Pick one and retry.",
        None,
    ),
]


@pytest.mark.parametrize(
    "factory, patch_target, kwargs, expected_msg, expected_result",
    NONE_CASES,
    ids=[
        "normalize+denoise",
        "segment",
        "segment+classify",
    ],
)
def test_widgets_warn_on_missing_layers(
    factory: Callable,
    patch_target: str,
    kwargs: dict[str, None],
    expected_msg: str,
    expected_result: Union[list, None],
) -> None:
    with patch(patch_target) as mock_warning:
        # Get the widget factory and call its underlying function
        widget = factory()
        result = widget(**kwargs)
    mock_warning.assert_called_once_with(expected_msg)
    assert result == expected_result


def test_normalize_and_denoise_widget(img: Image, make_napari_viewer, qtbot) -> None:
    # We test for all test cases from img fixture
    # since theoretically it should work on any kind of image
    viewer = make_napari_viewer()
    bf_layer = viewer.add_image(
        img.data,
        name=img.name,
    )
    widget = normalize_and_denoise_widget()
    widget(viewer=viewer, BF=bf_layer)
    expected_name = f"{bf_layer.name}_denoised"

    def layer_added():
        return any(layer.name == expected_name for layer in viewer.layers)

    qtbot.waitUntil(layer_added, timeout=500000)
    denoised_layer = viewer.layers[expected_name]
    assert isinstance(denoised_layer, Image)

    # Ensure that scale and translate are preserved
    assert (denoised_layer.scale == img.scale[-2:]).all()
    assert (denoised_layer.translate == img.translate[-2:]).all()

    # Must be a gray img with size maintained
    # NOTE: some img cases are have more dims,
    # but we must end up with two
    assert denoised_layer.ndim == 2
    if img.name == "astronaut":
        expected_spatial = img.data.shape[:2]
    else:
        expected_spatial = img.data.shape[-2:]

    assert denoised_layer.data.shape == expected_spatial


def test_segment_widget(img: Image, make_napari_viewer, qtbot) -> None:
    pytest.skip()
    # TODO: we should be able to simulate segmentation results
    if img.name != "sample_czi_ch0":
        pytest.skip("Test is only for DAPI channel.")

    # Sanity check that segmentation results are not there beforehand
    assert "segmentation" not in img.metadata

    widget = segment_widget()
    results = widget(DAPI=img)

    # Make sure segmentation metadata is saved to img
    assert "segmentation" in img.metadata
    assert all(
        res in img.metadata["segmentation"]
        for res in ["masks", "centroids", "bounding_boxes"]
    )

    # We expect layers for masks, centroids and bboxes
    assert len(results) == 3

    # Ensure the layers are correct
    for i, layer in enumerate(results):
        assert isinstance(layer, Layer)
        assert (layer.scale == img.scale[-2:]).all()
        assert (layer.translate == img.translate[-2:]).all()

        if i == 0:  # Pred mask layer
            assert (layer.data == img.metadata["segmentation"]["masks"]).all()
        elif i == 1:  # Centroids
            assert (layer.data == img.metadata["segmentation"]["centroids"]).all()
        else:  # Bboxes
            assert (layer.data == img.metadata["segmentation"]["bounding_boxes"]).all()
