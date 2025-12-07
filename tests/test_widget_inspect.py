import pytest

from neurogenesis_napari.widget_utils import _extract_cell_patch

@pytest.mark.parametrize("img", ["sample_czi_ch0", "sample_czi_ch1"], indirect=True)
def test_extract_cell_path(img, sample_bbox_shapes_layer) -> None:
    # sanity check for fixture
    assert sample_bbox_shapes_layer.name == f"{img.name}_test"
    patch, layer_properties = _extract_cell_patch(
        layer=sample_bbox_shapes_layer,
        bbox_index=0,
        DAPI=img,
        BF=img, 
        Tuj1=img,
        RFP=img,
    )
    assert patch.ndim == 3
    assert patch.shape[0] == 4
    assert all(channel_name == img.name for channel_name in layer_properties.keys())
