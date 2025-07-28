from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
from napari.layers import Image

from neurogenesis_napari._utils.geometry import bbox_to_rectangle
from neurogenesis_napari._utils.image import get_gray_img
from neurogenesis_napari._utils.model_hub import FILE_MAP, ensure_weights


@pytest.mark.parametrize(
    "bbox",
    [
        [1, 2, 4, 6],
    ],
)
def test_bbox_to_rectangle(bbox: Sequence[int]) -> None:
    expected = np.array([[1, 2], [1, 6], [4, 6], [4, 2]], dtype=float)
    result = bbox_to_rectangle(bbox)
    assert (expected == result).all()


def test_get_gray_img(img: Image) -> np.ndarray:
    # Sanity checks on input shape
    if img.name.startswith("sample_czi"):
        assert img.data.ndim == 4
    else:
        expected_ndim = {"camera": 2, "synthetic": 2, "astronaut": 3}[img.name]
        assert img.data.ndim == expected_ndim

    result = get_gray_img(img)

    # Must be a gray img with size maintained
    assert result.ndim == 2
    if img.name == "astronaut":
        expected_spatial = img.data.shape[:2]
    else:
        expected_spatial = img.data.shape[-2:]

    assert result.shape == expected_spatial


def test_ensure_weights_downloads_all(tmp_path: Path) -> None:
    contents: dict[str, bytes] = {}

    def fake_downloader(repo_id: str, filename: str) -> str:
        # Simulate remote cache file path
        cache_file = tmp_path / f"cache_{filename}"
        data = f"{repo_id}:{filename}".encode()
        cache_file.write_bytes(data)
        contents[filename] = data
        return str(cache_file)

    ensure_weights(
        base_dir=tmp_path, downloader=fake_downloader, repo_id="FAKE/REPO"
    )

    # Assert directories & files
    for category, filenames in FILE_MAP.items():
        cat_dir = tmp_path / category
        assert cat_dir.is_dir()
        for fname in filenames:
            path = cat_dir / fname
            assert path.is_file()
            assert path.read_bytes() == contents[fname]
