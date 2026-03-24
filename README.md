# Object Detection in Off-Road Terrain

A deep learning project for detecting obstacles in off-road environments using YOLO (Ultralytics), developed as a Master's thesis.

## Project Overview

This repository contains the complete implementation of a CNN-based object detection system designed to identify obstacles in challenging off-road terrain. The system leverages computer vision techniques to enable autonomous navigation in unstructured environments. For full architecture overview see [ARCHITECTURE.md](github.com/spilkovab/DP_2026/blob/main/ARCHITECTURE.md)

> **Note:** The dataset images and trained model weights are not included due to their large file size. See the [Drive folder](https://drive.google.com/drive/folders/13HK3aNuS0XGXs0uKUe35SjNUQdC3RddV?usp=sharing) where the latest iteration of dataset and weights of **model_K3** are saved. For previous versions of the dataset or weights for other models please contact me at *TODO: email*

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

Pretrained model weights are not included in the repository. Download them from the [Drive folder](https://drive.google.com/drive/folders/13HK3aNuS0XGXs0uKUe35SjNUQdC3RddV?usp=sharing).

```
DP_2026/
└── DP/
    └── runs/detect/
        └── model_K3/
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

### Run inference on a video
 
Before running, open `DP/predict_code/predict_video.py` and set the following paths at the top of the file:
 
```python
MODEL_NAME = 'model_K3'                         # name of your model folder under runs/detect/
VIDEO_PATH = "path/to/your/input_video.MOV"     # path to input video
OUTPUT_PATH = "path/to/your/output_video.mp4"   # path for the annotated output video
```
 
The script expects the model weights to be located at:
```
runs/detect/<MODEL_NAME>/weights/best.pt
```

If you placed the weights in different directory, you need to edit the paths in the following line of code:
```python
model = YOLO("path/to/your/weights/best.pt")
```

Then run:
 
```bash
python DP/predict_code/predict_video.py
```
 
The script will:
- Save the annotated video to the specified `OUTPUT_PATH`
- Display the inference in a window in real time — press `q` to stop early

The `predict_video.py` and `tracker_plot_new.py` scripts use a `draw_custom_annotations` function from `utils/visualization.py`. Please refer to the code documentation for more information on how to correctly use this function.

### Training the model
This script was used for training the latest version (model_K). To replicate the process follow these steps:

Before running, open `DP/train_code/train_model_K.py` and set the following variables at the top of the file:
 
```python
DATA = 'data_06'        # name of your dataset folder under dataset/
MODEL_NAME = 'model_K'  # name for the new model - use a unique name to avoid overwriting
```
 
The script expects your dataset to be located at:
```
dataset/<DATA>/
```
 
Then run:
 
```bash
python DP/train_code/train_model_K.py
```
 
The script will:
- Download `yolo11s.pt` automatically on first run (pretrained YOLO11 small weights)
- Train with image size 640, batch size 8, AugMix augmentation, and multi-scale training
- Use early stopping with a patience of 150 epochs
- Run validation automatically after training and print metrics
- Save all results to `runs/detect/<MODEL_NAME>/`
 
> **Note:** Training requires a CUDA-capable GPU. The base model `yolo11s.pt` will be downloaded automatically by Ultralytics on first use.
 
### Validate the model
 
Validation runs automatically at the end of training. To run it separately on an already trained model:
 
```bash
python DP/val_code/val_model_K.py
```

### Track objects in a video
 
Before running, open `DP/track_code/tracker_plot_new.py` and set the following variables at the top of the file:
 
```python
model_name     = 'model_J'                       # name of your model folder under runs/detect/
video_path     = "vidz/palacak_08_cut.MP4"       # path to input video
save_path_video = "track_results/annotated_tracking_model_J.mp4"  # annotated output video
save_path_graph = "track_results/trajectory_graph_model_J.mp4"    # trajectory graph output
```
 
Then run:
 
```bash
python DP/track_code/track.py
```
 
The script will:
- Run YOLO tracking (`persist=True`) on each frame and assign consistent IDs across frames
- Save two output videos: an annotated video with bounding boxes and labels, and a separate trajectory graph showing the movement paths of each tracked object
- Display both windows in real time — press `q` to stop early
- Track up to the last 100 positions per object, with trajectories color-coded by class

The class color mapping is:

| Class ID | Color |
|----------|-------|
| 0 | Green |
| 1 | Yellow |
| 2 | Purple |
| 3 | Blue |
| 4 | Light pink |

> **Note:** Make sure the track_results/ output directory exists before running, or the video writers will silently fail.

## References

- Boris Sekachev, Nikita Manovich, et al. (2020). *Computer Vision Annotation Tool (CVAT)* [Computer software]. https://github.com/cvat-ai/cvat
- Ultralytics YOLO. https://github.com/ultralytics/ultralytics
