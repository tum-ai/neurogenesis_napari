import numpy as np
from magicgui import magic_factory
from napari.qt.threading import thread_worker
from napari import Viewer
from napari.layers import Image, Labels, Layer, Points, Shapes
from napari.utils.notifications import (
    show_error,
    show_warning,
)
from skimage.measure import regionprops

from neurogenesis_napari._utils import (
    bbox_to_rectangle,
    get_gray_img,
    setup_cellpose_log_panel,
    log_context,
)


@thread_worker
def _segment_async(
    img_gray: np.ndarray,
    gpu: bool = False,
    model_type: str = "cyto3",
    panel_key: str = "segment_widget",
) -> tuple[np.ndarray, list[list[float]], list[np.ndarray]]:
    """Segment *img_gray* with Cellpose and derive centroids + bounding boxes. Route the logs to a separate context associated with the panel key.
    Args:
        img_gray (np.ndarray): 2‑D numpy array.
        gpu (bool: False): If ``True`` and a CUDA device is available, run Cellpose on GPU.
        model_type (str: = "cyto3"):  Name of the pretrained Cellpose model to load.
        panel_key (str: = "segment_widget): Panel key of the log context.

    Returns:
        pred_masks
        centroids
        bounding_boxes
    """
    from cellpose import models
    from neurogenesis_napari.segmentation.dapi_cellpose import segment

    # route logs from this thread to the matching dock only
    with log_context(panel_key):
        model = models.Cellpose(gpu=gpu, model_type=model_type)
        pred_masks = segment(img_gray, model)

    regions = regionprops(pred_masks)
    centroids = []
    bounding_boxes = []
    for region in regions:
        centroids.append([float(region.centroid[0]), float(region.centroid[1])])
        bounding_boxes.append(bbox_to_rectangle(region.bbox))

    return pred_masks, centroids, bounding_boxes


def _get_segmentation_layers(
    img: Image,
    pred_masks: np.ndarray,
    centroids: list[list[float]],
    bounding_boxes: list[np.ndarray],
) -> list[Layer]:
    labels_layer = Labels(
        data=pred_masks,
        name=f"{img.name}_masks",
        scale=img.scale[-2:],
        translate=img.translate[-2:],
    )

    points_layer = Points(
        data=np.asarray(centroids),
        name=f"{img.name}_centroids",
        size=30,
        face_color="yellow",
        opacity=0.8,
        scale=img.scale[-2:],
        translate=img.translate[-2:],
    )

    boxes_layer = Shapes(
        data=bounding_boxes,
        name=f"{img.name}_boxes",
        shape_type="polygon",
        edge_color="lime",
        face_color=[0, 0, 0, 0],
        edge_width=4,
        scale=img.scale[-2:],
        translate=img.translate[-2:],
    )

    return [labels_layer, points_layer, boxes_layer]


@magic_factory(
    call_button="Segment",
)
def segment_widget(
    viewer: Viewer,
    DAPI: Image | None = None,
    gpu: bool = False,
    model_type: str = "cyto3",
) -> list[Layer]:
    """Run segmentation and add three visual layers to Napari.

    Args:
        DAPI (Image): DAPI channel.
        gpu (bool: = False): Forwarded to "_get_bounding_boxes". Use GPU if available.
        model_type (str: = "cyto3"): Which Cellpose model weights to load.

    Returns:
        A list of mask Labels, centroid Points, and bounding box Shapes layers (in that order).
    """
    if DAPI is None:
        show_warning("No DAPI image layer selected. Pick one and retry.")
        return []

    img_gray = get_gray_img(DAPI)

    dock_panel_key = "segment_widget"
    setup_cellpose_log_panel(
        viewer,
        panel_key=dock_panel_key,
        dock_title="Cellpose logs - Segment",
    )
    worker = _segment_async(img_gray, gpu, model_type, dock_panel_key)

    def _on_done(result) -> None:
        pred_masks, centroids, bounding_boxes = result

        DAPI.metadata["segmentation"] = {
            "masks": pred_masks,
            "centroids": centroids,
            "bounding_boxes": bounding_boxes,
        }

        segmentation_layers = _get_segmentation_layers(DAPI, pred_masks, centroids, bounding_boxes)
        for layer in segmentation_layers:
            viewer.add_layer(layer)

    worker.returned.connect(_on_done)
    worker.errored.connect(lambda e: show_error(f"Cellpose failed: {e}"))
    worker.start()

    return []
