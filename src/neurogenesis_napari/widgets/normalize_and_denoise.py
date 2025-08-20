import napari
import numpy as np
from magicgui import magic_factory
from napari.qt.threading import thread_worker
from napari import Viewer
from napari.layers import Image
from napari.utils.notifications import show_warning, show_error

from neurogenesis_napari._utils import get_gray_img, log_context, setup_cellpose_log_panel
from neurogenesis_napari.preprocessing.cellpose_denoising import denoise_image
from neurogenesis_napari.preprocessing.colour_normalization import normalize_colors


DENOISE_WIDGET_PANEL_KEY = "denoise_widget"


@thread_worker
def _denoise_async(
    normalized_img: np.ndarray,
    panel_key: str,
) -> np.ndarray:
    """Denoise *img_gray* with Cellpose. Route the logs to a separate context associated with the panel key.
    Args:
        img_gray (np.ndarray): 2‑D numpy array.
        panel_key (str): Panel key of the log context.

    Returns:
        pred_masks
        centroids
        bounding_boxes
    """
    # route logs from this thread to the matching dock only
    with log_context(panel_key):
        denoised_img = denoise_image(normalized_img)
    return denoised_img


@magic_factory(
    call_button="Normalize + Denoise",
)
def normalize_and_denoise_widget(
    viewer: Viewer,
    BF: Image | None = None,
) -> None:
    """Normalise colour and denoise the selected bright‑field layer.

    Args:
        BF (Image):   Bright‑field channel.

    Returns:
        None
    """
    if BF is None:
        show_warning("No BF image layer selected. Pick one and retry.")
        return None

    img_gray = get_gray_img(BF)

    try:
        normalized = normalize_colors(img_gray)
    except ValueError as e:
        napari.utils.notifications.show_error(f"Normalisation failed: {e}")
        return

    setup_cellpose_log_panel(
        viewer,
        panel_key=DENOISE_WIDGET_PANEL_KEY,
        dock_title="Cellpose logs - Segment",
    )

    worker = _denoise_async(normalized, DENOISE_WIDGET_PANEL_KEY)

    def _on_done(result) -> None:
        denoised = np.squeeze(result)
        denoised_img = Image(
            data=denoised,
            name=f"{BF.name}_denoised",
            scale=BF.scale[-2:],
            translate=BF.translate[-2:],
        )
        viewer.add_layer(denoised_img)

    worker.returned.connect(_on_done)
    worker.errored.connect(lambda e: show_error(f"Cellpose denoising failed: {e}"))
    worker.start()

    return None
