import numpy as np
from cellpose import denoise


def denoise_image(image: np.ndarray) -> np.ndarray:
    """
    Apply Cellpose's denoising model to an input image.

    This function uses the CellposeDenoiseModel (with model_type 'cyto3' and restore_type 'denoise_cyto3')
    to denoise the input image. The model is run on GPU if available. The function returns the denoised
    image as a numpy array, with shape (H, W, 1).

    Args:
        image (np.ndarray): The input image to denoise. Should be a 3D image.

    Returns:
        denoised_img (np.ndarray) The denoised image of shape (H, W, 1).
    """
    dn_model = denoise.CellposeDenoiseModel(
        gpu=True, model_type="cyto3", restore_type="denoise_cyto3"
    )
    _, _, _, denoised_img = dn_model.eval(image, channels=[0, 0])
    return denoised_img
