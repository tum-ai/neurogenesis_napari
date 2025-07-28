# TumAI Helmholtz Toolkit for napari

Brings common pre-processing and segmentation steps for digital pathology directly into napari.

---

## Key features

| Step | What it does | Where to click |
|------|--------------|----------------|
| **Normalize + Denoise** | *✓* Colour normalisation against an internal reference <br> | `Normalize + Denoise` |
| **Segment nuclei** | *✓* Runs Cellpose (`cyto3`) on the active image layer <br> *✓* Adds masks, centroids, and bounding-boxes as separate layers <br> *✓* (Optional) overlays ground-truth boxes loaded from a CSV | `Segment Nuclei`|

All layers retain the scale/translate metadata of the original image, so you can continue analysing or exporting them downstream.

---

## Installation

```bash
pip install tumai-napari
# or, inside napari:
#   Plugins ➜ Install/Uninstall Plugins ➜ search for “TumAI Histology Toolkit”
