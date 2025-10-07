from typing import Dict, Optional, List, Tuple, Any
import numpy as np
from magicgui import magicgui
from napari import Viewer
from napari.layers import Shapes, Image
from napari.utils.notifications import show_warning, show_info
from qtpy import QtWidgets

from neurogenesis_napari._utils import (
    get_gray_img,
    crop_stack_resize,
)


def _create_inspection_viewer(
    patch_data: np.ndarray, 
    current_label: str, 
    idx2lbl: Dict[int, str],
    original_layer: Shapes,
    bbox_index: int,
    layer_properties: Dict[str, Dict[str, Any]], 
    progress_info: str = "",
    all_bbox_indices: List[int] = None,
    current_index: int = 0,
    image_layers: Dict[str, Image] = None,
) -> Viewer:
    """Create a new Napari viewer window for inspecting a single cell."""
    # Create new viewer with progress info
    title = f"Cell Inspection - {current_label}"
    if progress_info:
        title = f"{title} {progress_info}"
    viewer = Viewer(title=title)
        
    for i, channel_name in enumerate(layer_properties.keys()):
        channel_data = patch_data[i]
        properties = layer_properties[channel_name]
        
        viewer.add_image(
            channel_data,
            name=channel_name,
            colormap=properties['colormap'],
            contrast_limits=properties['contrast_limits'],
            gamma=properties['gamma'],
            opacity=properties['opacity'],
            blending=properties['blending'],
        )
    
    # Add label editing widget with navigation
    _add_label_editor_to_viewer(
        viewer, current_label, idx2lbl, original_layer, bbox_index, 
        all_bbox_indices, current_index, image_layers
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
    
    # Store the original title with progress info
    original_title = viewer.title
    
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
        # Handle label update
        original_layer.properties["label"][bbox_index] = new_label
        original_layer.text.values[bbox_index] = new_label
        original_layer.refresh_colors()
        original_layer.refresh()
        
        # Update title but preserve progress info
        base_title = f"Cell Inspection - {new_label}"
        if "(" in original_title and ")" in original_title:
            # Extract progress info from original title
            progress_part = original_title[original_title.find("("):original_title.find(")") + 1]
            viewer.title = f"{base_title} {progress_part}"
        else:
            viewer.title = base_title
            
        show_info(f"Label updated to: {new_label}")
    
    def go_to_next():
        """Navigate to the next cell."""
        if all_bbox_indices and current_index < len(all_bbox_indices) - 1:
            # Clean up dock widgets before closing
            _cleanup_dock_widgets(viewer)
            # Close current viewer
            viewer.close()
            
            # Get next cell data
            next_bbox_index = all_bbox_indices[current_index + 1]
            next_label = original_layer.properties["label"][next_bbox_index]
            
            # Extract patch for next cell
            patch_data, layer_properties = _extract_cell_patch(
                original_layer, next_bbox_index, 
                image_layers['DAPI'], image_layers['BF'], 
                image_layers['Tuj1'], image_layers['RFP']
            )
            
            if patch_data is not None:
                progress_info = f"({current_index + 2}/{len(all_bbox_indices)})"
                _create_inspection_viewer(
                    patch_data, next_label, idx2lbl, original_layer, 
                    next_bbox_index, layer_properties, progress_info,
                    all_bbox_indices, current_index + 1, image_layers
                )
        else:
            show_info("This is the last cell.")
    
    def go_to_previous():
        """Navigate to the previous cell."""
        if all_bbox_indices and current_index > 0:
            # Clean up dock widgets before closing
            _cleanup_dock_widgets(viewer)
            # Close current viewer
            viewer.close()
            
            # Get previous cell data
            prev_bbox_index = all_bbox_indices[current_index - 1]
            prev_label = original_layer.properties["label"][prev_bbox_index]
            
            # Extract patch for previous cell
            patch_data, layer_properties = _extract_cell_patch(
                original_layer, prev_bbox_index,
                image_layers['DAPI'], image_layers['BF'], 
                image_layers['Tuj1'], image_layers['RFP']
            )
            
            if patch_data is not None:
                progress_info = f"({current_index}/{len(all_bbox_indices)})"
                _create_inspection_viewer(
                    patch_data, prev_label, idx2lbl, original_layer, 
                    prev_bbox_index, layer_properties, progress_info,
                    all_bbox_indices, current_index - 1, image_layers
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
    
    # Store references to dock widgets for cleanup
    label_dock = viewer.window.add_dock_widget(update_label, area="right", name="Edit Label")
    nav_dock = viewer.window.add_dock_widget(nav_widget, area="right", name="Navigation")
    
    # Store dock widget references in viewer for cleanup
    viewer._inspect_dock_widgets = [label_dock, nav_dock]
    
    # Add cleanup when viewer is closed manually
    def on_viewer_close(event):
        _cleanup_dock_widgets(viewer)
    
    viewer.window.qt_window.closeEvent = on_viewer_close


def _cleanup_dock_widgets(viewer: Viewer) -> None:
    """Properly clean up dock widgets before closing the viewer."""
    if hasattr(viewer, '_inspect_dock_widgets'):
        for dock_widget in viewer._inspect_dock_widgets:
            try:
                # Remove the dock widget from the window
                viewer.window.remove_dock_widget(dock_widget)
                # Delete the widget to free memory
                if hasattr(dock_widget, 'widget') and dock_widget.widget():
                    dock_widget.widget().deleteLater()
            except Exception:
                # Ignore errors during cleanup
                pass
        # Clear the reference
        delattr(viewer, '_inspect_dock_widgets')


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
                'colormap': img.colormap,
                'contrast_limits': img.contrast_limits,
                'gamma': img.gamma,
                'opacity': img.opacity,
                'blending': img.blending,
            }

        patch = crop_stack_resize(tuple(channels), bbox)        
        return patch, layer_properties       
    except Exception as e:
        show_warning(f"Failed to extract cell patch: {e}")
        return None


def attach_inspect_widget(
    viewer: Viewer, 
    layer: Shapes, 
    idx2lbl: Dict[int, str], 
    DAPI: Image,
    BF: Image,
    Tuj1: Image,
    RFP: Image,
) -> None:
    """Attach an inspection widget that opens selected cells in a new window.
    
    Args:
        viewer: Main Napari viewer
        layer: Prediction shapes layer
        idx2lbl: Dictionary mapping indices to labels
    """
    
    @magicgui(
        call_button="Inspect Selected",
        inspect_all={"widget_type": "CheckBox", "text": "Inspect all cells (not just selected)"}
    )
    def inspect_cell(inspect_all: bool = False) -> None:
        """Open selected cell(s) in a new inspection window."""
        # Get bounding boxes to inspect
        if inspect_all:
            # Inspect all bounding boxes
            bbox_indices = list(range(len(layer.data)))
            if not bbox_indices:
                show_warning("No cells found to inspect.")
                return
        else:
            # Get selected bounding boxes
            bbox_indices = list(layer.selected_data)
            if not bbox_indices:
                show_warning("Please select one or more cells to inspect, or check 'Inspect all cells'.")
                return
        
        # Start with the first cell
        bbox_index = bbox_indices[0]
        current_label = layer.properties["label"][bbox_index]
        
        # Extract patch
        patch_data, layer_properties = _extract_cell_patch(layer, bbox_index, DAPI, BF, Tuj1, RFP)
        
        if patch_data is None:
            return
        
        # Create image layers dictionary for navigation
        image_layers = {
            'DAPI': DAPI,
            'BF': BF,
            'Tuj1': Tuj1,
            'RFP': RFP,
        }
        
        # Create inspection viewer with navigation info
        progress_info = f"(1/{len(bbox_indices)})" if len(bbox_indices) > 1 else ""
        _create_inspection_viewer(
            patch_data, current_label, idx2lbl, layer, bbox_index, 
            layer_properties, progress_info, bbox_indices, 0, image_layers
        )
        
        show_info(f"Opened inspection window for {current_label}. Use Next/Previous buttons to navigate.")
    
    viewer.window.add_dock_widget(inspect_cell, area="right", name="Inspect Cells")