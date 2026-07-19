---
title: Brain Tumor Detection
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: "1.37.0"
app_file: app.py
pinned: false
---

# Brain Tumor Segmentation & Classification

## Overview
This app runs a two-stage pipeline on a single MRI slice: it first classifies the
scan as one of
- **Glioma**
- **Meningioma**
- **Pituitary Tumor**
- **No Tumor**

and, only if a tumor is detected, segments it pixel-by-pixel and draws its
bounding box.

> [!WARNING]
>
> This tool is intended for demonstration purposes only. It should not be used as a substitute for professional medical diagnosis.

## Features
- **Two-stage pipeline:** a dedicated classifier gates a dedicated segmenter — the segmentation model only runs on scans the classifier believes contain a tumor.
- **Interactive UI:** upload an MRI image and see the tumor region overlaid on the original scan with its bounding box, predicted type, and confidence breakdown.

## Model
- **Classifier** (`models/classifier.py`): `ConvNext`, a from-scratch ConvNeXt-Tiny implementation, predicts one of the four classes above from the full MRI slice.
- **Segmenter** (`models/unet.py`): `AttentionUNet`, a from-scratch Attention U-Net, predicts a per-pixel tumor mask. It only runs on scans the classifier labels as a tumor type.

Select layers in both models use decorrelated backpropagation (`models/decor.py`,
`DecorConv2d`/`DecorLinear`) rather than plain `Conv2d`/`Linear`, which decorrelates
a layer's input on the fly to counteract the way correlated inputs skew gradient
descent. Placement is chosen per-layer based on how well-conditioned that layer's
online correlation estimate is (a function of batch size and spatial resolution at
that layer), not applied uniformly.

At inference (`models/inference.py`), the segmenter's predicted mask's largest
connected component gives the bounding box shown in the UI.

## Dataset
Training uses [BRISC2025](https://arxiv.org/abs/2506.14318) (Fateh et al., 2025), a
6,000-slice T1-weighted MRI dataset with physician-reviewed pixel-level masks,
expected to be downloaded locally into `brisc2025/` at the project root:

```
brisc2025/
├─ classification_task/{train,test}/{glioma,meningioma,pituitary,no_tumor}/*.jpg
└─ segmentation_task/{train,test}/{images/*.jpg, masks/*.png}
```

`preprocessing/dataset.py` builds classification samples from `classification_task`
(all four classes) and segmentation samples from `segmentation_task` (tumor
scans only, since `no_tumor` scans have no mask and are never passed to the
segmenter). BRISC2025 ships its own stratified train/test split, which is used
as-is; a stratified validation slice is carved out of the train split.

## Installation

```bash
git clone https://github.com/KhoaNguyen02/brain-tumor-detection.git
cd brain-tumor-detection
pip install -r requirements.txt
```

Download BRISC2025 and place it at `brisc2025/` in the project root (matching the
layout above) before training.

## Training

```bash
python training.py
```

Trains `ConvNext` and `AttentionUNet` in sequence, saving weights and history to
`pretrained/Classifier/` and `pretrained/Segmenter/` respectively. See
`train_colab.ipynb` for a Colab-ready version (GPU runtime, Drive-backed
checkpoints, `kagglehub` dataset download).

## Running the demo

```bash
streamlit run app.py
```

Upload an MRI image; the app classifies it and, if a tumor is present, overlays
the segmented region on the original scan along with its bounding box, type, and
confidence.
