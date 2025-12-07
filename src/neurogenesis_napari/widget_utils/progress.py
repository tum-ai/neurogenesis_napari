from typing import Dict, Union
from napari.utils import progress
from qtpy import QtWidgets, QtCore


def start_progress(pbar: Dict[str, Union[None, progress]]) -> None:
    """Start a napari progress indicator and show wait cursor.

    Args:
        pbar (Dict[str, Union[None, progress]]): Dictionary to store progress object.
            The progress object will be stored in pbar["obj"].

    Returns:
        None
    """
    pbar["obj"] = progress()
    QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)


def close_progress(pbar: Dict[str, progress]) -> None:
    """Close the napari progress indicator and restore normal cursor.

    Args:
        pbar (Dict[str, progress]): Dictionary containing the progress object
            at pbar["obj"].

    Returns:
        None
    """
    pbar["obj"].close()
    QtWidgets.QApplication.restoreOverrideCursor()
