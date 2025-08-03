# logging_to_napari.py
import logging
import weakref
import threading
from contextlib import contextmanager
from typing import Iterable
from qtpy.QtCore import QObject, Signal
from qtpy.QtWidgets import QPlainTextEdit

_REGISTRY: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()
_LOG_CTX = threading.local()  # holds .panel_key for the current worker thread


@contextmanager
def log_context(panel_key: str):
    """Set the current panel_key for the calling thread while inside the context."""
    prev = getattr(_LOG_CTX, "panel_key", None)
    _LOG_CTX.panel_key = panel_key
    try:
        yield
    finally:
        _LOG_CTX.panel_key = prev


class _LogEmitter(QObject):
    line = Signal(str)


class _PanelKeyFilter(logging.Filter):
    """Only allow records that were emitted while this thread's context matches."""

    def __init__(self, panel_key: str):
        super().__init__()
        self._panel_key = panel_key

    def filter(self, record: logging.LogRecord) -> bool:
        return getattr(_LOG_CTX, "panel_key", None) == self._panel_key


class NapariLogHandler(logging.Handler):
    """Thread-safe handler that forwards log records to a QPlainTextEdit in napari."""

    def __init__(self, emitter: _LogEmitter, panel_key: str):
        super().__init__()
        self.emitter = emitter
        self.panel_key = panel_key
        self.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s — %(message)s", datefmt="%H:%M:%S")
        )
        # filter to only this panel's context
        self.addFilter(_PanelKeyFilter(panel_key))

    def emit(self, record):
        try:
            msg = self.format(record)
        except Exception:
            msg = record.getMessage()
        self.emitter.line.emit(msg)


def _viewer_state(viewer) -> dict:
    state = _REGISTRY.get(viewer)
    if state is None:
        state = {}
        _REGISTRY[viewer] = state
    return state


def _has_handler_for_key(logger: logging.Logger, panel_key: str) -> bool:
    return any(
        isinstance(h, NapariLogHandler) and getattr(h, "panel_key", None) == panel_key
        for h in logger.handlers
    )


def setup_cellpose_log_panel(
    viewer,
    *,
    panel_key: str,
    dock_title: str | None = None,
    area: str = "right",
    logger_names: Iterable[str] = ("cellpose", "cellpose.models"),
    logger_level: int = logging.INFO,
):
    dock_title = dock_title or f"Cellpose logs — {panel_key}"

    vstate = _viewer_state(viewer)
    if panel_key in vstate:
        return vstate[panel_key]["dock"]

    log_widget = QPlainTextEdit()
    log_widget.setReadOnly(True)
    dock = viewer.window.add_dock_widget(log_widget, name=dock_title, area=area)

    emitter = _LogEmitter()
    emitter.line.connect(log_widget.appendPlainText)
    handler = NapariLogHandler(emitter, panel_key)
    handler.setLevel(logger_level)

    targets = [logging.getLogger(name) for name in logger_names]
    for lg in targets:
        lg.setLevel(logger_level)
        if not _has_handler_for_key(lg, panel_key):
            lg.addHandler(handler)
        lg.propagate = False

    vstate[panel_key] = {
        "dock": dock,
        "handler": handler,
        "emitter": emitter,
        "widget": log_widget,
        "targets": targets,
    }

    def _cleanup(*_):
        st = _REGISTRY.pop(viewer, {})
        for _, entry in st.items():
            for lg in entry["targets"]:
                try:
                    lg.removeHandler(entry["handler"])
                except Exception:
                    pass
            try:
                entry["dock"].close()
            except Exception:
                pass

    try:
        viewer.events.destroy.connect(_cleanup)
    except Exception:
        try:
            viewer.events.close.connect(_cleanup)
        except Exception:
            pass

    return dock
