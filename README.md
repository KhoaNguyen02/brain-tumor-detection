# Brain Tumor Detection

## Overview
This project aims to implement a complete pipeline to scan through MRI images to detect and segment different types of tumors including:
- **Glioma**
- **Meningioma**
- **Pituitary Tumor**

> [!WARNING]
>
> This tool is intended for demonstration purposes only. It should not be used as a substitute for professional medical diagnosis.

## Features
- **Two-stage pipeline:** a dedicated classifier gates a dedicated segmenter — the segmentation model only runs on scans the classifier believes contain a tumor.
- **Interactive UI:** upload an MRI image and see the tumor region overlaid on the original scan with its bounding box, predicted type, and confidence breakdown.

## Model
- **Classifier** (`models/classifier.py`): `ConvNext`, a from-scratch ConvNeXt-Tiny implementation, predicts one of the four classes above from the full MRI slice.
- **Segmenter** (`models/unet.py`): `AttentionUNet`, a from-scratch Attention U-Net, predicts a tumor mask. It only runs on scans the classifier labels as a tumor type.

Some convolution layers in both models use decorrelation (`DecorConv2d`,
`models/decor.py`) instead of plain `Conv2d`, which decorrelates a layer's
input to counteract the way correlated inputs ruin gradient descent.

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
`pretrained/Classifier/` and `pretrained/Segmenter/` respectively.

## Running the demo

```bash
streamlit run app.py
```

Upload an MRI image; the model classifies it and, if a tumor is present, overlays
the segmented region on the original scan along with its bounding box, type, and
confidence.
