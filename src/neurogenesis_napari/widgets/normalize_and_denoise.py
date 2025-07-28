import napari
import numpy as np
from magicgui import magic_factory
from napari.layers import Image
from napari.utils.notifications import show_warning

from neurogenesis_napari._utils import get_gray_img
from neurogenesis_napari.preprocessing.cellpose_denoising import denoise_image
from neurogenesis_napari.preprocessing.colour_normalization import normalize_colors


@magic_factory(
    call_button="Normalize + Denoise",
)
def normalize_and_denoise_widget(
    BF: Image | None = None,
) -> Image:
    """Normalise colour and denoise the selected bright‑field layer.

    Args:
        BF (Image):   Bright‑field channel.

    Returns:
        denoised (Image): A new layer containing the denoised image.
        The name is suffixed with `_denoised` and the spatial scale is inherited from the input.
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

    try:
        denoised = denoise_image(normalized)
    except ValueError as e:
        napari.utils.notifications.show_error(f"Denoising failed: {e}")
        return

    denoised = np.squeeze(denoised)

    return Image(
        data=denoised,
        name=f"{BF.name}_denoised",
        scale=BF.scale[-2:],
        translate=BF.translate[-2:],
    )
