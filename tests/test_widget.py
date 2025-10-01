from collections.abc import Callable
from typing import Union, List
from functools import partial
from unittest.mock import patch

import pytest
from napari.layers import Image, Layer, Labels, Points, Shapes
from napari import Viewer

from neurogenesis_napari.widgets import (
    normalize_and_denoise_widget,
    segment_and_classify_widget,
    segment_widget,
)

from neurogenesis_napari.widgets.normalize_and_denoise import _normalize_and_denoise_widget_impl
from neurogenesis_napari.widgets.segment import _segment_widget_impl, _get_segmentation_layers
from neurogenesis_napari.widgets.segment_and_classify import _segment_and_classify_widget_impl

NONE_CASES = [
    (
        _normalize_and_denoise_widget_impl,
        "neurogenesis_napari.widgets.normalize_and_denoise.show_warning",
        {"BF": None},
        "No BF image layer selected. Pick one and retry.",
        None,
    ),
    (
        _segment_widget_impl,
        "neurogenesis_napari.widgets.segment.show_warning",
        {"DAPI": None},
        "No DAPI image layer selected. Pick one and retry.",
        None,
    ),
    (
        _segment_and_classify_widget_impl,
        "neurogenesis_napari.widgets.segment_and_classify.show_warning",
        {"DAPI": None, "Tuj1": None, "RFP": None, "BF": None},
        "No DAPI, Tuj1, RFP, BF image layer(s) selected. Pick one and retry.",
        None,
    ),
]


@pytest.mark.parametrize(
    "impl_f, patch_target, kwargs, expected_msg, expected_result",
    NONE_CASES,
    ids=[
        "normalize+denoise",
        "segment",
        "segment+classify",
    ],
)
def test_widgets_warn_on_missing_layers(
    impl_f: Callable,
    patch_target: str,
    kwargs: dict[str, None],
    expected_msg: str,
    expected_result: Union[list, None],
) -> None:
    with patch(patch_target) as mock_warning:
        result = impl_f(viewer=None, **kwargs)
    mock_warning.assert_called_once_with(expected_msg)
    assert result == expected_result


@pytest.mark.parametrize("img", ["astronaut_rgb", "sample_czi_ch0"], indirect=True)
def test_normalize_and_denoise_widget(img: Image, make_napari_viewer: Viewer, qtbot) -> None:
    # We test for all test cases from img fixture
    # since theoretically it should work on any kind of image
    viewer = make_napari_viewer()
    bf_layer = viewer.add_image(
        img.data,
        name=img.name,
    )
    widget = normalize_and_denoise_widget()
    widget(viewer=viewer, BF=bf_layer)
    expected_names = [f"{bf_layer.name}_denoised"]

    qtbot.waitUntil(partial(_expected_layers_added, viewer, expected_names), timeout=50000)
    denoised_layer = viewer.layers[expected_names[0]]
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


def test_get_segmentation_layers(img: Image, sample_segmentation) -> None:
    masks, centroids, boxes = sample_segmentation

    layers = _get_segmentation_layers(
        img=img,
        pred_masks=masks,
        centroids=centroids,
        bounding_boxes=boxes,
    )

    assert len(layers) == 3

    # Ensure the layers are correct
    for i, layer in enumerate(layers):
        assert isinstance(layer, Layer)
        assert (layer.scale == img.scale[-2:]).all()
        assert (layer.translate == img.translate[-2:]).all()

        if i == 0:  # Mask layer
            assert isinstance(layer, Labels)
            assert (layer.data == masks).all()
            assert layer.name == f"{img.name}_masks"
        elif i == 1:  # Centroids
            assert isinstance(layer, Points)
            assert (layer.data == centroids).all()
            assert layer.name == f"{img.name}_centroids"
        else:  # Bboxes
            assert isinstance(layer, Shapes)
            assert (layer.data == boxes).all()
            assert layer.name == f"{img.name}_boxes"


@pytest.mark.parametrize("img", ["sample_czi_ch0"], indirect=True)
def test_segment_widget_adds_layers_and_metadata(img: Image, make_napari_viewer: Viewer, qtbot, fast_segment_worker) -> None:
    viewer = make_napari_viewer()
    dapi_layer = viewer.add_image(
        img.data,
        name="DAPI",
    )
    # NOTE: this does not actually run segmentation, see fast_segment_worker
    widget = segment_widget()
    widget(viewer=viewer, DAPI=dapi_layer)
    expected_names = ["DAPI_masks", "DAPI_centroids", "DAPI_boxes"]
    
    qtbot.waitUntil(partial(_expected_layers_added, viewer, expected_names), timeout=50000)

    # Make sure segmentation metadata is saved to img
    assert "segmentation" in dapi_layer.metadata
    assert all(
        res in dapi_layer.metadata["segmentation"]
        for res in ["masks", "centroids", "bounding_boxes"]
    )

    # Number of layers must be 4 (original + masks, centroids, bboxes)
    assert len(viewer.layers) == 4


def _expected_layers_added(viewer: Viewer, expected_names: List[str]) -> bool:
    existing = {layer.name for layer in viewer.layers}
    return all(name in existing for name in expected_names)
