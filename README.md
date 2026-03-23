# Object Detection in Off-Road Terrain

A deep learning project for detecting obstacles in off-road environments using YOLO (Ultralytics), developed as a Master's thesis.

## Project Overview

This repository contains the complete implementation of a CNN-based object detection system designed to identify obstacles in challenging off-road terrain. The system leverages computer vision techniques to enable autonomous navigation in unstructured environments.

> **Note:** The dataset images and trained model weights are not included due to their large file size. See the [Setup](#setup) section for instructions on how to obtain them.

## Repository Structure

```
DP_2026/
├── DP/
│   ├── train_code/       # Model training scripts
│   ├── predict_code/     # Inference pipeline for obstacle detection
│   ├── val_code/         # Model validation and evaluation scripts
│   ├── track_code/       # Object tracking implementation
│   └── utils/            # Utility functions and helper modules
├── runs/detect/          # Detection results, training and validation outputs
├── val_results/          # Detailed validation metrics
├── requirements.txt
└── README.md
```

## Technology Stack

- **Language:** Python
- **Detection Framework:** [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- **Annotation Tool:** [CVAT](https://github.com/cvat-ai/cvat)
- **Application:** Off-road obstacle detection and object tracking

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/spilkovab/DP_2026.git
cd DP_2026
```

### 2. Install dependencies

It is recommended to use a virtual environment:

```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Download model weights

Pretrained model weights are not included in the repository. Download them from the [Releases page](https://github.com/spilkovab/DP_2026/releases) and place them in the following location:

```
DP_2026/
└── DP/
    └── weights/
        └── best.pt     # place downloaded weights here
```

### 4. Prepare your dataset

The dataset was annotated using [CVAT](https://github.com/cvat-ai/cvat). If you want to use your own data, annotate it in YOLO format and place it in the following structure:

```
DP_2026/
└── dataset/
    ├── images/
    │   ├── train/
    │   └── val/
    └── labels/
        ├── train/
        └── val/
    └── data.yaml
    └── train.txt
    └── val.txt
    └── test.txt
```

## Usage

### Run inference on an image or video

```bash
python DP/predict_code/predict.py --source path/to/your/image_or_video
```

### Train the model

```bash
python DP/train_code/train.py --data path/to/data.yaml --epochs 100
```

### Validate the model

```bash
python DP/val_code/val.py --weights DP/weights/best.pt --data path/to/data.yaml
```

## References

- Boris Sekachev, Nikita Manovich, et al. (2020). *Computer Vision Annotation Tool (CVAT)* [Computer software]. https://github.com/cvat-ai/cvat
- Ultralytics YOLO. https://github.com/ultralytics/ultralytics
