# TUM.ai Neurogenesis Toolkit

A napari plugin for automated nuclear segmentation and neural cell type classification in neurogenesis research.

## Features

### Widgets

| Widget | Function | Input |
|--------|----------|-------|
| **Normalize + Denoise** | Color normalization and denoising | Bright-field images |
| **Segment** | Nuclear segmentation with Cellpose | DAPI/nuclear staining |
| **Segment + Classify** | Complete segmentation and classification pipeline | 4-channel images (DAPI, Tuj1, RFP, BF) |

### Cell Type Classification

The plugin classifies detected cells into four categories:
- Astrocytes
- Neurons
- OPCs (Oligodendrocyte Precursor Cells)
- Dead Cells

Classification uses a Variational Autoencoder (VAE) for feature extraction followed by nearest-centroid classification.

## Installation

```bash
pip install neurogenesis-napari
```

Alternatively, install through napari: `Plugins` → `Install/Uninstall Plugins` → search for "TUM.ai Neurogenesis Toolkit"
