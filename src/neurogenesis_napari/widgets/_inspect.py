from typing import Dict, Optional, List, Tuple, Any
import numpy as np
from magicgui import magicgui
from napari import Viewer
from napari.layers import Shapes, Image
from napari.utils.notifications import show_warning, show_info

from neurogenesis_napari._utils import crop_stack_resize

# Global dictionary to store inspection state for each viewer
_inspection_states: Dict[Viewer, Dict[str, Any]] = {}

# Global dictionary to store widget references for each viewer
_viewer_widgets: Dict[Viewer, Dict[str, Any]] = {}

# Global dictionary to store cleanup functions for each viewer
_viewer_cleanup_functions: Dict[Viewer, callable] = {}


def _add_images_to_viewer(
    viewer: Viewer, patch_data: np.ndarray, layer_properties: Dict[str, Dict[str, Any]]
) -> None:
    """Add image layers to the viewer with their properties."""
    for i, channel_name in enumerate(layer_properties.keys()):
        channel_data = patch_data[i]
        properties = layer_properties[channel_name]

        viewer.add_image(
            channel_data,
            name=channel_name,
            colormap=properties["colormap"],
            contrast_limits=properties["contrast_limits"],
            gamma=properties["gamma"],
            opacity=properties["opacity"],
            blending=properties["blending"],
        )


def _create_inspection_viewer(
    patch_data: np.ndarray,
    current_label: str,
    idx2lbl: Dict[int, str],
    original_layer: Shapes,
    bbox_index: int,
    layer_properties: Dict[str, Dict[str, Any]],
    progress_info: str,
    all_bbox_indices: List[int] = None,
    current_index: int = 0,
    image_layers: Dict[str, Image] = None,
) -> None:
    """Create a new Napari viewer window for inspecting a single cell."""

    viewer = Viewer(title=f"Cell Inspection - {current_label} {progress_info}")

    # Add cleanup handler for when the viewer is closed
    def cleanup_viewer():
        """Clean up global state when viewer is closed."""
        if viewer in _inspection_states:
            del _inspection_states[viewer]
        if viewer in _viewer_widgets:
            del _viewer_widgets[viewer]
        if viewer in _viewer_cleanup_functions:
            del _viewer_cleanup_functions[viewer]

    # Store cleanup function in global dictionary
    _viewer_cleanup_functions[viewer] = cleanup_viewer

    # Connect to Qt close event for cleanup
    # to manually cleanup the viewer to avoid ghost windows
    def on_close_event(event):
        cleanup_viewer()
        from qtpy.QtWidgets import QWidget

        QWidget.closeEvent(viewer.window.qt_viewer, event)

    viewer.window.qt_viewer.closeEvent = on_close_event

    _add_images_to_viewer(viewer, patch_data, layer_properties)

    _add_label_editor_to_viewer(
        viewer,
        current_label,
        idx2lbl,
        original_layer,
        bbox_index,
        all_bbox_indices,
        current_index,
        image_layers,
    )


def _update_inspection_viewer(
    viewer: Viewer,
    patch_data: np.ndarray,
    current_label: str,
    bbox_index: int,
    layer_properties: Dict[str, Dict[str, Any]],
    progress_info: str,
    current_index: int = 0,
) -> None:
    """Update an existing inspection viewer with new cell data."""

    viewer.title = f"Cell Inspection - {current_label} {progress_info}"

    # Clear existing images
    viewer.layers.clear()

    # Add channels for next cell
    _add_images_to_viewer(viewer, patch_data, layer_properties)

    _update_label_editor_widget(
        viewer,
        current_label,
        bbox_index,
        current_index,
    )


def _add_label_editor_to_viewer(
    viewer: Viewer,
    current_label: str,
    idx2lbl: Dict[int, str],
    original_layer: Shapes,
    bbox_index: int,
    all_bbox_indices: List[int] = None,
    current_index: int = 0,
    image_layers: Dict[str, Image] = None,
) -> None:
    """Add a label editing widget to the inspection viewer."""

    # Store navigation state in global dictionary
    _inspection_states[viewer] = {
        "current_label": current_label,
        "idx2lbl": idx2lbl,
        "original_layer": original_layer,
        "bbox_index": bbox_index,
        "all_bbox_indices": all_bbox_indices,
        "current_index": current_index,
        "image_layers": image_layers,
    }

    @magicgui(
        new_label={"widget_type": "ComboBox", "choices": list(idx2lbl.values())},
        call_button="Update Label",
        persist=True,
        auto_call=False,
    )
    def update_label(
        new_label: str = current_label,
    ) -> None:
        """Update the label in the original prediction layer."""
        state = _inspection_states[viewer]

        # Propagate label update to the original predictions layer
        state["original_layer"].properties["label"][state["bbox_index"]] = new_label
        state["original_layer"].text.values[state["bbox_index"]] = new_label
        state["original_layer"].refresh_colors()
        state["original_layer"].refresh()

        total_cells = len(state["all_bbox_indices"])
        current_pos = state["current_index"] + 1
        progress_info = f"({current_pos}/{total_cells})" if total_cells > 1 else ""
        viewer.title = f"Cell Inspection - {new_label} {progress_info}"

        # Update stored state
        state["current_label"] = new_label
        show_info(f"Label updated to: {new_label}")

    def go_to_next():
        """Navigate to the next cell."""
        state = _inspection_states[viewer]

        if state["current_index"] < len(state["all_bbox_indices"]) - 1:

            next_bbox_index = state["all_bbox_indices"][state["current_index"] + 1]
            next_label = state["original_layer"].properties["label"][next_bbox_index]

            # Extract patch for next cell
            patch_data, layer_properties = _extract_cell_patch(
                state["original_layer"],
                next_bbox_index,
                state["image_layers"]["DAPI"],
                state["image_layers"]["BF"],
                state["image_layers"]["Tuj1"],
                state["image_layers"]["RFP"],
            )

            if patch_data is not None:
                progress_info = f"({state['current_index'] + 2}/{len(state['all_bbox_indices'])})"
                _update_inspection_viewer(
                    viewer,
                    patch_data,
                    next_label,
                    next_bbox_index,
                    layer_properties,
                    progress_info,
                    state["current_index"] + 1,
                )
        else:
            show_info("This is the last cell, you can close the window.")

    def go_to_previous():
        """Navigate to the previous cell."""
        state = _inspection_states[viewer]

        if state["current_index"] > 0:

            prev_bbox_index = state["all_bbox_indices"][state["current_index"] - 1]
            prev_label = state["original_layer"].properties["label"][prev_bbox_index]

            # Extract patch for previous cell
            patch_data, layer_properties = _extract_cell_patch(
                state["original_layer"],
                prev_bbox_index,
                state["image_layers"]["DAPI"],
                state["image_layers"]["BF"],
                state["image_layers"]["Tuj1"],
                state["image_layers"]["RFP"],
            )

            if patch_data is not None:
                progress_info = f"({state['current_index']}/{len(state['all_bbox_indices'])})"
                _update_inspection_viewer(
                    viewer,
                    patch_data,
                    prev_label,
                    prev_bbox_index,
                    layer_properties,
                    progress_info,
                    state["current_index"] - 1,
                )
        else:
            show_info("This is the first cell.")

    # Create separate navigation buttons
    from qtpy.QtWidgets import QPushButton, QVBoxLayout, QWidget

    nav_widget = QWidget()
    nav_layout = QVBoxLayout()

    next_btn = QPushButton("Next Cell")
    prev_btn = QPushButton("Previous Cell")

    next_btn.clicked.connect(go_to_next)
    prev_btn.clicked.connect(go_to_previous)

    nav_layout.addWidget(next_btn)
    nav_layout.addWidget(prev_btn)
    nav_widget.setLayout(nav_layout)

    # Set the current label as default
    update_label.new_label.value = current_label

    # Hide navigation buttons if only one cell
    if not all_bbox_indices or len(all_bbox_indices) <= 1:
        next_btn.setVisible(False)
        prev_btn.setVisible(False)

    # Add dock widgets and store references
    label_dock = viewer.window.add_dock_widget(update_label, area="right", name="Edit Label")
    nav_dock = viewer.window.add_dock_widget(nav_widget, area="right", name="Navigation")

    # Store widget references for later updates
    _viewer_widgets[viewer] = {
        "update_label_widget": update_label,
        "label_dock": label_dock,
        "nav_dock": nav_dock,
    }


def _update_label_editor_widget(
    viewer: Viewer,
    current_label: str,
    bbox_index: int,
    current_index: int = 0,
) -> None:
    """Update the label editor widget with new cell data."""
    # Update stored state
    _inspection_states[viewer].update(
        {
            "current_label": current_label,
            "bbox_index": bbox_index,
            "current_index": current_index,
        }
    )

    # Update the label editor widget using stored reference
    update_label_widget = _viewer_widgets[viewer]["update_label_widget"]
    if hasattr(update_label_widget, "new_label"):
        update_label_widget.new_label.value = current_label


def _extract_cell_patch(
    layer: Shapes,
    bbox_index: int,
    DAPI: Image,
    BF: Image,
    Tuj1: Image,
    RFP: Image,
) -> Optional[Tuple[Optional[np.ndarray], Dict[str, Dict[str, Any]]]]:
    """Extract the 4-channel patch for a specific bounding box."""
    try:
        # Get the bounding box
        bbox = layer.data[bbox_index]

        # Squeeze out singleton dimensions (important for CZI files)
        channels: List[np.ndarray] = []
        layer_properties: Dict[str, Dict[str, Any]] = {}

        for img in [DAPI, BF, Tuj1, RFP]:
            # Remove dimensions of size 1
            squeezed = np.squeeze(img.data)
            if squeezed.ndim == 3:
                if squeezed.shape[2] == 4:
                    squeezed = squeezed[..., :3]
            channels.append(squeezed)
            layer_properties[img.name] = {
                "colormap": img.colormap,
                "contrast_limits": img.contrast_limits,
                "gamma": img.gamma,
                "opacity": img.opacity,
                "blending": img.blending,
            }

        patch = crop_stack_resize(tuple(channels), bbox)
        return patch, layer_properties
    except Exception as e:
        show_warning(f"Failed to extract cell patch: {e}")
        return None


def attach_inspect_widget(
    viewer: Viewer,
    prediction_layer: Shapes,
    idx2lbl: Dict[int, str],
    DAPI: Image,
    BF: Image,
    Tuj1: Image,
    RFP: Image,
) -> None:
    """Attach an inspection widget that sequentially opens
    all or selected cells in a new window. The window has docked
    button for updating the class label for a cell, along with two
    buttons that allow to go to the previous and the next cells.

    Args:
        viewer (Viewer): Napari viewer instance.
        prediction_layer (Shapes): Shapes layer containing cell polygons with labels.
        idx2lbl (Dict[int, str]): Dictionary mapping indices to labels.
        DAPI (Image): DAPI channel.
        BF (Image): Bright‑field channel.
        Tuj1 (Image): β‑III‑tubulin channel.
        RFP (Image): RFP channel.

    Returns:
        None
    """

    @magicgui(
        call_button="Inspect Selected",
        inspect_all={"widget_type": "CheckBox", "text": "Inspect all cells (not just selected)"},
    )
    def inspect_cell(inspect_all: bool = False) -> None:
        if inspect_all:
            bbox_indices = list(range(len(prediction_layer.data)))
            if not bbox_indices:
                show_warning("No cells found to inspect. The prediction layer seems to be empty.")
                return
        else:
            bbox_indices = list(prediction_layer.selected_data)
            if not bbox_indices:
                show_warning(
                    "Please select one or more cells to inspect, or check 'Inspect all cells'."
                )
                return

        # Always start with the first cell
        bbox_index = bbox_indices[0]
        current_label = prediction_layer.properties["label"][bbox_index]

        # Extract patch
        patch_data, layer_properties = _extract_cell_patch(
            prediction_layer, bbox_index, DAPI, BF, Tuj1, RFP
        )

        if patch_data is None:
            return

        # Create image layers dictionary for navigation
        # TODO: dynamically resolve
        image_layers = {
            "DAPI": DAPI,
            "BF": BF,
            "Tuj1": Tuj1,
            "RFP": RFP,
        }

        # Create inspection viewer with navigation info
        progress_info = f"(1/{len(bbox_indices)})" if len(bbox_indices) > 1 else ""
        _create_inspection_viewer(
            patch_data,
            current_label,
            idx2lbl,
            prediction_layer,
            bbox_index,
            layer_properties,
            progress_info,
            bbox_indices,
            0,
            image_layers,
        )

    viewer.window.add_dock_widget(inspect_cell, area="right", name="Inspect Cells")
