# TUMai Helmholtz Napari Plugin

[![License MIT](https://img.shields.io/pypi/l/neurogenesis-napari.svg?color=green)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/neurogenesis-napari.svg?color=green)](https://pypi.org/project/neurogenesis-napari)
[![Python Version](https://img.shields.io/pypi/pyversions/neurogenesis-napari.svg?color=green)](https://python.org)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/neurogenesis-napari)](https://napari-hub.org/plugins/neurogenesis-napari)

This plugin provides one-click color normalization, denoising, and Cellpose-based nuclear segmentation.

## Key Features

| Widget | Function | Input | Output |
|--------|----------|-------|---------|
| **Normalize + Denoise** | Color normalization and denoising | Bright-field image | Processed image |
| **Segment** | Nuclear segmentation | DAPI/nuclear stain | Masks, centroids, bounding boxes |
| **Segment + Classify** | End-to-end cell analysis | 4-channel images | Cell segmentation + classification |

## Quick Start

### Installation

```bash
pip install neurogenesis-napari
```

Or install through napari:
1. Open napari
2. Go to `Plugins` → `Install/Uninstall Plugins`
3. Search for **"TumAI Histology Toolkit"**
4. Click Install

### Basic Usage

1. **Load your images** into napari
2. **Select the appropriate widget** from the `Plugins` menu
3. **Choose your image layers** from the dropdown menus
4. **Click the action button** to process

The plugin will automatically download required AI models on first use.

---

## Widget Documentation

### Normalize + Denoise

**Purpose**: Standardizes color variations and reduces noise in bright-field images.

#### Usage
1. Load a bright-field image into napari
2. Open `Plugins` → `Normalize and Denoise`
3. Select your bright-field image from the **BF** dropdown
4. Click **"Normalize + Denoise"**

#### What it does
- **Color Normalization**: Adjusts colors against an internal reference to standardize appearance across different images/scanners
- **Denoising**: Removes noise while preserving important cellular structures
- **Output**: Creates a new layer named `{original_name}_denoised`

---

### Segment

**Purpose**: Detects and segments individual cell nuclei using Cellpose.

#### Usage
1. Load a nuclear staining image (DAPI) into napari
2. Open `Plugins` → `Segment`
3. Select your nuclear image from the **DAPI** dropdown
4. Optionally adjust:
   - **GPU**: Enable for faster processing
   - **Model**: Choose Cellpose model (`cyto3` default)
5. Click **"Segment Nuclei"**

#### What it does
- **Segmentation**: Uses Cellpose to identify individual nuclei
- **Creates 3 new layers**:
  - `{name}_masks`: Segmentation masks
  - `{name}_centroids`: Center points of each detected cell
  - `{name}_bboxes`: Bounding boxes around each cell

---

### Segment + Classify

**Purpose**: Complete pipeline that segments nuclei AND classifies cell types in multi-channel images.

#### Usage
1. Load a **4-channel image** into napari as separate layers:
   - **DAPI**: Nuclear staining
   - **Tuj1**: β-III-tubulin
   - **RFP**: Red fluorescent protein marker
   - **BF**: Bright-field
2. Open `Plugins` → `Segment and Classify`
3. Select each channel from the respective dropdowns
4. Choose **Reuse cached**:
   - **True**: Reuse previous segmentation (faster) from the segment widget
   - **False**: Perform fresh segmentation
5. Click **"Segment + Classify"**

#### What it does
1. **Segmentation**: Does segmentation same as the segment widget above
2. **Feature extraction**: Uses a Variational Autoencoder (VAE) to extract features
3. **Classification**: Nearest-centroid classifier assigns cell types

#### Output
Creates colored bounding box layers for each detected cell type:
- **🟣 Astrocytes** (magenta boxes)
- **⚫ Dead Cells** (gray boxes)
- **🔵 Neurons** (cyan boxes)
- **🟢 OPCs** (lime boxes)

Layer names show counts: `{count}_{cell_type}s` (e.g., `23_Neurons`)

---

### Supported Image Formats
- `.czi` (via napari-czifile2)
- `.png`, `.jpg`
