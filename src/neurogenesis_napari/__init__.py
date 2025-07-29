try:
    from ._version import __version__
except ImportError:
    __version__ = "0.1.0a1.post2"

from neurogenesis_napari.widgets import (
    normalize_and_denoise_widget,
    segment_and_classify_widget,
    segment_widget,
)

__all__ = (
    "normalize_and_denoise_widget",
    "segment_widget",
    "segment_and_classify_widget",
)
