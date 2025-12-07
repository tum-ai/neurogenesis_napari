from pathlib import Path
from typing import Tuple, List

import numpy as np
import pytest
import napari
from napari.layers import Image, Shapes
from napari_czifile2 import napari_get_reader
from napari.qt.threading import create_worker
from skimage import data


from neurogenesis_napari.widget_utils.classification import PALETTE


def _czi_channel_params_via_plugin(
    path: Path = Path(__file__).parent / "sample.czi",
) -> list[tuple[str, Image]]:
    """Load a sample czi file through the napari_czifile2 plugin reader
    and mimic how napari splits an image with a channel_axist into separate
    per channels arrays (without squeezing any other axes).
    """
    reader = napari_get_reader(str(path))
    ((data, add_kwargs, _),) = reader(str(path))
    ch_axis = add_kwargs["channel_axis"]
    per_channel_arrays = []
    for c in range(data.shape[ch_axis]):
        plane = np.take(data, c, axis=ch_axis)
        per_channel_arrays.append(plane)

    params = []
    for c, arr in enumerate(per_channel_arrays):
        params.append(
            (
                f"sample_czi_ch{c}",
                (lambda a=arr, c_=c: Image(a, name=f"sample_czi_ch{c_}")),
            )
        )
    return params

_ALL_IMG_PARAMS = [
    ("camera_gray", lambda: Image(data.camera(), name="camera")),
    ("astronaut_rgb", lambda: Image(data.astronaut(), name="astronaut", rgb=True)),
    ("synthetic", lambda: Image(np.random.rand(32, 42), name="synthetic")),
    *_czi_channel_params_via_plugin(),
]
_ID2PARAM = {p[0]: p for p in _ALL_IMG_PARAMS}


@pytest.fixture(params=_ALL_IMG_PARAMS, ids=[p[0] for p in _ALL_IMG_PARAMS])
def img(request: pytest.FixtureRequest) -> Image:
    param = request.param
    if isinstance(param, str):      
        param = _ID2PARAM[param]
    return param[1]()


@pytest.fixture
def make_napari_viewer(qtbot):
    """Provide a fresh non-interactive napari viewer for tests."""
    viewer = napari.Viewer(show=False)
    qtbot.addWidget(viewer.window._qt_window)
    return lambda: viewer


@pytest.fixture
def sample_segmentation() -> Tuple[np.ndarray, List[List[float]], List[np.ndarray]]:
    masks = np.load(Path(__file__).parent / "sample_masks.npy")
    centroids = np.load(Path(__file__).parent / "sample_centroids.npy")
    boxes = np.load(Path(__file__).parent / "sample_boxes.npy")
    return masks, centroids, boxes


@pytest.fixture
def sample_bbox_shapes_layer(img: Image) -> Shapes:
    boxes = np.load(Path(__file__).parent / "sample_boxes.npy")
    labels = ["OPC" for _ in range(len(boxes))] 
    layer = Shapes(
        data=boxes,
        shape_type="polygon",
        properties={"label": labels},
        name=f"{img.name}_test",
        edge_width=4,
        face_color=[0, 0, 0, 0],
        scale=img.scale[-2:],
        translate=img.translate[-2:],
        edge_color="label",
        edge_color_cycle=list(PALETTE.values()),
        text={
            "text": "{label}",
            "size": 5,
            "anchor": "upper_left",
            "translation": [0, 0],
        },
    )
    return layer


@pytest.fixture
def fast_segment_worker(monkeypatch, sample_segmentation) -> None:
    """Patch _segment_async so tests don't download models or run Cellpose."""
    masks, centroids, boxes = sample_segmentation

    def fake_segment_async(img_gray, panel_key, gpu=False, model_type="cyto3"):
        def run():
            return masks, centroids, boxes
        return create_worker(run) # mimics @thread_worker return

    monkeypatch.setattr(
        "neurogenesis_napari.widgets.segment._segment_async",
        fake_segment_async,
        raising=True,
    )


@pytest.fixture(autouse=True)
def mock_progress_functions(monkeypatch) -> None:
    """Mock progress functions to prevent hanging in tests."""
    def mock_start_progress(pbar):
        pbar["obj"] = None 
    
    def mock_close_progress(pbar):
        pass
    
    for widget in ["normalize_and_denoise", "segment"]:
        monkeypatch.setattr(f"neurogenesis_napari.widgets.{widget}.start_progress", mock_start_progress)
        monkeypatch.setattr(f"neurogenesis_napari.widgets.{widget}.close_progress", mock_close_progress)
