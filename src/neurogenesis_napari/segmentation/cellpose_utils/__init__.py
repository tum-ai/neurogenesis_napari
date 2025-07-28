# segmentation/cellpose_utils/__init__.py
import sys as _sys

_sys.modules.setdefault("cellpose_utils", _sys.modules[__name__])

from cellpose_utils.version import version, version_str  # ← unchanged

# …rest of the original code…
