from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Union

from huggingface_hub import hf_hub_download

REPO_ID = "Anuun/tumai-hemlholtz"

FILE_MAP: Mapping[str, list[str]] = {
    "cellpose": [
        "cyto_dapi_finetuned",
    ],
    "classifier": [
        "NearestCentroid.pkl",
        "label_encoder.pkl",
    ],
    "SAM": [
        "denoised_cellSAM2_large_2000.torch",
        "sam2.1_hiera_large.pt",
        "Tuj_cellSAM2_small_3000.torch",
    ],
    "vae": [
        "TL_FT_bigvae3.pth",
    ],
}


def get_weights_dir() -> Path:
    """Get the base weights directory."""
    return Path(__file__).parent.parent / "weights"


def get_weight_path(category: str, filename: str) -> Path:
    """Get the full path to a specific weight file.

    Args:
        category: The category (e.g., 'vae', 'classifier', 'SAM', 'cellpose')
        filename: The filename within that category

    Returns:
        Full path to the weight file
    """
    return get_weights_dir() / category / filename


Downloader = Callable[..., str]


def ensure_weights(
    base_dir: Union[Path, None] = None,
    repo_id: str = REPO_ID,
    mapping: Mapping[str, list[str]] = FILE_MAP,
    downloader: Downloader = hf_hub_download,
) -> None:
    """Ensure all weight files are present under base_dir."""
    base_dir = base_dir or get_weights_dir()
    base_dir.mkdir(exist_ok=True)

    for subdir, files in mapping.items():
        tgt_dir = base_dir / subdir
        tgt_dir.mkdir(parents=True, exist_ok=True)

        for filename in files:
            target_path = tgt_dir / filename
            cached_path = downloader(repo_id=repo_id, filename=filename)
            data = Path(cached_path).read_bytes()
            target_path.write_bytes(data)
