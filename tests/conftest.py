from pathlib import Path

import numpy as np
import pytest
from napari.layers import Image
from napari_czifile2 import napari_get_reader
from skimage import data


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


@pytest.fixture(
    params=[
        ("camera_gray", lambda: Image(data.camera(), name="camera")),
        (
            "astronaut_rgb",
            lambda: Image(data.astronaut(), name="astronaut", rgb=True),
        ),
        ("synthetic", lambda: Image(np.random.rand(32, 42), name="synthetic")),
        *_czi_channel_params_via_plugin(),
    ],
    ids=lambda p: p[0],
)
def img(request: pytest.FixtureRequest) -> Image:
    return request.param[1]()
