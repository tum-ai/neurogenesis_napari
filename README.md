# TUM.ai Neurogenesis Napari Plugin

[![License MIT](https://img.shields.io/pypi/l/neurogenesis-napari.svg?color=green)](https://github.com/tum-ai/neurogenesis_napari/blob/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/neurogenesis-napari.svg?color=green)](https://pypi.org/project/neurogenesis-napari)
[![Python Version](https://img.shields.io/pypi/pyversions/neurogenesis-napari.svg?color=green)](https://python.org)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/neurogenesis-napari)](https://napari-hub.org/plugins/neurogenesis-napari)

A napari plugin for automated nuclear segmentation and neural cell type classification in neurogenesis research. Supports classification of astrocytes, neurons, oligodendrocyte precursor cells (OPCs), and dead cells from multi-channel fluorescence microscopy images.

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
3. Search for "TUM.ai Neurogenesis Toolkit"
4. Click Install

### Basic Usage

1. Load your images into napari
2. Select the appropriate widget from the `Plugins` menu
3. Choose your image layers from the dropdown menus
4. Click the action button to process

Model weights are automatically downloaded on first use.

---

## Widget Documentation

### Normalize + Denoise

Standardizes color variations and reduces noise in bright-field microscopy images.

#### Usage
1. Load a bright-field image into napari
2. Open `Plugins` → `Normalize and Denoise`
3. Select your bright-field image from the **BF** dropdown
4. Click **"Normalize + Denoise"**

#### What it does
- **Color Normalization**: Histogram matching against an internal reference to standardize appearance across different acquisitions
- **Denoising**: Removes noise while preserving cellular structures using Cellpose
- **Output**: Creates a new layer named `{original_name}_denoised`

---

### Segment

Detects and segments individual cell nuclei from DAPI-stained images using Cellpose.

#### Usage
1. Load a DAPI/nuclear staining image into napari
2. Open `Plugins` → `Segment`
3. Select your DAPI image from the **DAPI** dropdown
4. Optionally adjust:
   - **GPU**: Enable for faster processing (if CUDA available)
   - **Model**: Choose Cellpose model (`cyto3` default)
5. Click **"Segment"**

#### What it does
- **Segmentation**: Uses Cellpose to identify individual nuclei
- **Creates 3 new layers**:
  - `{name}_masks`: Segmentation masks for each nucleus
  - `{name}_centroids`: Center points of detected nuclei
  - `{name}_bboxes`: Bounding boxes (polygons) around each nucleus

---

### Segment + Classify

End-to-end pipeline that segments nuclei and classifies neural cell types in multi-channel fluorescence images.

#### Usage
1. Load a **4-channel image** into napari as separate layers:
   - **DAPI**: Nuclear staining (DAPI/Hoechst)
   - **Tuj1**: β-III-tubulin (neuronal marker)
   - **RFP**: Red fluorescent protein marker
   - **BF**: Bright-field
2. Open `Plugins` → `Segment and Classify`
3. Select each channel from the respective dropdowns
4. Choose **Reuse cached segmentation**:
   - **True** (default): Reuse previous segmentation if available (faster)
   - **False**: Perform fresh segmentation
5. Click **"Segment + Classify"**

#### How it works
1. **Segmentation**: Cellpose-based nuclear segmentation on DAPI channel
2. **Feature extraction**: Variational Autoencoder (VAE) extracts features from 4-channel patches around each nucleus
3. **Classification**: Nearest-centroid classifier assigns cell types based on learned centroids

#### Output
Creates colored polygon overlays for each detected cell:
- Astrocytes (magenta)
- Neurons (cyan)
- OPCs - Oligodendrocyte Precursor Cells (lime)
- Dead Cells (gray)

The classification results can be manually corrected through an interactive interface. Select any cell and use keyboard shortcuts to reassign its type: Shift+A (Astrocyte), Shift+N (Neuron), Shift+O (OPC), Shift+D (Dead Cell).

---

---

## Technical Details

### Supported Image Formats
- `.czi` (Zeiss microscopy files, via napari-czifile2)
- `.tiff`, `.tif`
- `.png`, `.jpg`

### Cell Classification Model
- **Feature extraction**: Variational Autoencoder (VAE) with 2304-dimensional latent space
- **Classifier**: Scikit-learn Nearest Centroid
- **Input**: 224×224 pixel patches from 4 channels (DAPI, BF, Tuj1, RFP)
- **Output**: 4 cell type classes

### Requirements
- Python ≥ 3.10
- CUDA-capable GPU (optional, for faster processing)
- Model weights are automatically downloaded on first use via Hugging Face Hub

---

## Citation

If you use this plugin in your research, please cite:

```
TUM.ai Neurogenesis Napari Plugin
Technical University of Munich
https://github.com/tum-ai/neurogenesis_napari
```

---

## License

MIT License - see [LICENSE](LICENSE) file for details.
