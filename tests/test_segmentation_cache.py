"""Tests for segmentation caching (sidecar files)."""
from pathlib import Path
import json
import numpy as np
import pytest
from unittest.mock import Mock
from napari.layers import Image

from neurogenesis_napari.widget_utils.segmentation import (
    save_segmentation,
    load_segmentation,
    _get_image_hash,
    _get_sidecar_paths,
)


def test_save_segmentation_creates_sidecar_files(tmp_path, sample_segmentation) -> None:
    """Test that save_segmentation creates .seg.json and .masks.npz files."""
    masks, centroids, boxes = sample_segmentation
    
    image_file = tmp_path / "test_image.czi"
    image_file.write_text("fake czi data")
    
    img = Mock()
    img.source.path = str(image_file)
    img.data = np.random.rand(512, 512)
    
    result = save_segmentation(
        image=img,
        masks=masks,
        centroids=centroids.tolist(),
        bounding_boxes=boxes.tolist(),
        gpu=False,
        model_type="cyto3"
    )
    
    assert result is True
    
    json_path = tmp_path / "test_image.czi.seg.json"
    masks_path = tmp_path / "test_image.czi.masks.npz"
    
    assert json_path.exists()
    assert masks_path.exists()


def test_save_segmentation_json_content(tmp_path, sample_segmentation) -> None:
    """Test that the JSON metadata contains correct information."""
    masks, centroids, boxes = sample_segmentation
    
    image_file = tmp_path / "test.czi"
    image_file.write_text("fake")
    
    img = Mock()
    img.source.path = str(image_file)
    img.data = np.random.rand(512, 512)
    
    save_segmentation(
        image=img,
        masks=masks,
        centroids=centroids.tolist(),
        bounding_boxes=boxes.tolist(),
        gpu=True,  # with GPU=True
        model_type="cyto2"
    )
    
    json_path = tmp_path / "test.czi.seg.json"
    with open(json_path, 'r') as f:
        metadata = json.load(f)
    
    assert "image_hash" in metadata
    assert "image_shape" in metadata
    assert "centroids" in metadata
    assert "num_cells" in metadata
    assert "parameters" in metadata
    assert "version" in metadata
    
    assert metadata["num_cells"] == len(centroids)
    assert metadata["centroids"] == centroids.tolist()
    assert metadata["parameters"]["gpu"] is True
    assert metadata["parameters"]["model_type"] == "cyto2"
    assert metadata["version"] == "1.0"
    assert metadata["image_shape"] == [512, 512]


def test_save_segmentation_masks_content(tmp_path, sample_segmentation) -> None:
    """Test that the .npz file contains masks and bboxes."""
    masks, centroids, boxes = sample_segmentation
    
    image_file = tmp_path / "test.czi"
    image_file.write_text("fake")
    
    img = Mock()
    img.source.path = str(image_file)
    img.data = np.random.rand(512, 512)
    
    save_segmentation(
        image=img,
        masks=masks,
        centroids=centroids.tolist(),
        bounding_boxes=boxes.tolist(),
        gpu=False,
        model_type="cyto3"
    )
    
    masks_path = tmp_path / "test.czi.masks.npz"
    data = np.load(masks_path, allow_pickle=True)
    
    assert "masks" in data
    assert "bboxes" in data
    assert np.array_equal(data["masks"], masks)
    assert len(data["bboxes"]) == len(boxes)


def test_save_and_load_roundtrip(tmp_path, sample_segmentation) -> None:
    """Test that we can save and then load back the same segmentation."""
    masks, centroids, boxes = sample_segmentation
    
    image_file = tmp_path / "roundtrip.czi"
    image_file.write_text("fake")
    
    img = Mock()
    img.source.path = str(image_file)
    img.data = np.random.rand(512, 512)
    
    # Save
    save_result = save_segmentation(
        image=img,
        masks=masks,
        centroids=centroids.tolist(),
        bounding_boxes=boxes.tolist(),
        gpu=False,
        model_type="cyto3"
    )
    assert save_result is True
    
    # Load
    loaded = load_segmentation(
        image=img,
        gpu=False,
        model_type="cyto3"
    )
    
    assert loaded is not None
    assert "masks" in loaded
    assert "centroids" in loaded
    assert "bounding_boxes" in loaded
    
    assert np.array_equal(loaded["masks"], masks)
    assert loaded["centroids"] == centroids.tolist()
    assert len(loaded["bounding_boxes"]) == len(boxes)


def test_save_segmentation_no_source_path(sample_segmentation) -> None:
    """Test that save_segmentation returns False when image has no source path."""
    masks, centroids, boxes = sample_segmentation
    
    img = Mock()
    img.source.path = None
    
    result = save_segmentation(
        image=img,
        masks=masks,
        centroids=centroids.tolist(),
        bounding_boxes=boxes.tolist(),
        gpu=False,
        model_type="cyto3"
    )
    
    assert result is False


def test_load_segmentation_with_wrong_params(tmp_path, sample_segmentation) -> None:
    """Test that load fails when parameters don't match."""
    masks, centroids, boxes = sample_segmentation
    
    image_file = tmp_path / "test.czi"
    image_file.write_text("fake")
    
    img = Mock()
    img.source.path = str(image_file)
    img.data = np.random.rand(512, 512)
    
    save_segmentation(
        image=img,
        masks=masks,
        centroids=centroids.tolist(),
        bounding_boxes=boxes.tolist(),
        gpu=False,
        model_type="cyto3"
    )
    
    # try to load with different params
    loaded = load_segmentation(
        image=img,
        gpu=True, 
        model_type="cyto3"
    )
    
    assert loaded is None


def test_load_segmentation_with_changed_image(tmp_path, sample_segmentation) -> None:
    """Test that load fails when image data has changed."""
    masks, centroids, boxes = sample_segmentation
    
    image_file = tmp_path / "test.czi"
    image_file.write_text("fake")
    
    img = Mock()
    img.source.path = str(image_file)
    img.data = np.random.rand(256, 256)
    
    save_segmentation(
        image=img,
        masks=masks,
        centroids=centroids.tolist(),
        bounding_boxes=boxes.tolist(),
        gpu=False,
        model_type="cyto3"
    )
    
    img_modified = Mock()
    img_modified.source.path = str(image_file)
    img_modified.data = np.random.rand(256, 256)
    
    loaded = load_segmentation(
        image=img_modified,
        gpu=False,
        model_type="cyto3"
    )
    
    assert loaded is None


def test_load_segmentation_nonexistent_files(tmp_path) -> None:
    """Test that load returns None when sidecar files don't exist."""
    image_file = tmp_path / "nosidecar.czi"
    image_file.write_text("fake")
    
    img = Mock()
    img.source.path = str(image_file)
    img.data = np.random.rand(512, 512)
    
    loaded = load_segmentation(
        image=img,
        gpu=False,
        model_type="cyto3"
    )
    
    assert loaded is None


def test_get_sidecar_paths() -> None:
    """Test _get_sidecar_paths helper function."""
    img = Mock()
    img.source.path = "/data/experiment.czi"
    
    json_path, masks_path = _get_sidecar_paths(img)
    
    assert json_path == Path("/data/experiment.czi.seg.json")
    assert masks_path == Path("/data/experiment.czi.masks.npz")


def test_get_sidecar_paths_no_source() -> None:
    """Test _get_sidecar_paths returns None when no source."""
    img = Mock()
    img.source.path = None
    
    json_path, masks_path = _get_sidecar_paths(img)
    
    assert json_path is None
    assert masks_path is None


def test_get_image_hash_consistency() -> None:
    """Test that _get_image_hash produces consistent results."""
    img_data = np.random.rand(512, 512)
    
    hash1 = _get_image_hash(img_data)
    hash2 = _get_image_hash(img_data)
    
    assert hash1 == hash2
    assert len(hash1) == 16 


def test_get_image_hash_detects_changes() -> None:
    """Test that _get_image_hash changes when image changes."""
    img1 = np.random.rand(512, 512)
    img2 = np.random.rand(512, 512)
    
    hash1 = _get_image_hash(img1)
    hash2 = _get_image_hash(img2)
    
    assert hash1 != hash2