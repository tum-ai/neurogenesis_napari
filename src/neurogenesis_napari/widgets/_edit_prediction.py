from typing import Dict
from magicgui import magicgui
from napari import Viewer
from napari.layers import Shapes
from napari.utils.notifications import (
    show_warning,
)


def _set_label(layer: Shapes, label: str) -> None:
    """Assign a label to the currently selected *cell polygons* in a Shapes layer.

    If no polygons are selected, shows a warning and exits. For each selected
    polygon, updates both the layer's ``properties["label"]`` and the displayed
    text to the given label, then refreshes the layer.

    Args:
        layer (Shapes): Napari Shapes layer containing cell polygons.
        label (str): Label to assign to all selected polygons.

    Returns:
        None
    """
    selected_polys = list(layer.selected_data)
    if not selected_polys:
        show_warning("Select one or more cell polygons first.")
        return
    for idx in selected_polys:
        layer.properties["label"][idx] = label
        layer.text.values[idx] = label
    layer.refresh_colors()
    layer.refresh()


def add_label_hotkeys(layer: Shapes, idx2lbl: Dict[int, str]) -> None:
    """Bind number keys to assign predefined labels to selected polygons.

    Hotkeys:
        0 → "Astrocyte"
        1 → "Dead Cell"
        2 → "Neuron"
        3 → "OPC"

    Pressing a key will call `_set_label` on all currently selected polygons
    in the given Shapes layer.

    Args:
        layer (Shapes): The Napari Shapes layer containing cell polygons.
        idx2lbl (Dict[int, str]): Dict of available idx (key) and label pairs.

    Returns:
        None
    """
    # NOTE: maybe the keys should be incremented by one, since 0 is at the other end on the keyboard
    for key in sorted(idx2lbl.keys()):
        lbl = idx2lbl[key]

        def _handler(event=None, lbl=lbl) -> None:
            _set_label(layer, lbl)

        layer.bind_key(str(key), _handler)


def attach_edit_widget(viewer: Viewer, layer: Shapes, idx2lbl: Dict[int, str]) -> None:
    """Attach an interactive label-editing widget to the viewer.

    Provides a docked dropdown (combo box) for selecting a class label
    from the given list and applying it to the currently selected polygons.
    The dropdown is automatically synced with the label of the first
    selected polygon. Also enables hotkey support for quick labeling
    via `add_label_hotkeys`.

    Args:
        viewer (Viewer): Napari viewer instance.
        layer (Shapes): Shapes layer containing cell polygons.
        idx2lbl (Dict[int, str]): Dict of available idx and label pairs.

    Returns:
        None
    """

    @magicgui(
        class_label={"widget_type": "ComboBox", "choices": idx2lbl.values()},
        call_button="Apply",
        persist=True,
        auto_call=False,
    )
    def edit_label(class_label: str = "Neuron") -> None:
        _set_label(layer, class_label)

    # keep dropdown synced with current selection
    def _sync_dropdown(event=None) -> None:
        sel = list(layer.selected_data)
        if sel:
            edit_label.class_label.value = layer.properties["label"][sel[0]]

    layer.events.connect(_sync_dropdown)
    viewer.window.add_dock_widget(edit_label, area="right", name="Edit cell label")

    add_label_hotkeys(layer, idx2lbl)
