from typing import Dict, Union
from napari.utils import progress
from qtpy import QtWidgets, QtCore


def start_progress(pbar: Dict[str, Union[None, progress]]) -> None:
    pbar["obj"] = progress()
    QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)


def close_progress(pbar: Dict[str, progress]) -> None:
    pbar["obj"].close()
    QtWidgets.QApplication.restoreOverrideCursor()
