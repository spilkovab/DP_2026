# Object Detection in Off-Road Terrain

A comprehensive deep learning project for detecting obstacles in off-road environments using convolutional neural networks, developed as a Master's thesis.

## Project Overview

This repository contains the complete implementation of a CNN-based object detection system designed to identify obstacles in challenging off-road terrain. The system leverages computer vision techniques to enable autonomous navigation in unstructured environments.

**Note:** The dataset images and trained model weights are not included due to their large file size.

## Dataset

The dataset was collected and manually annotated using the **CVAT (Computer Vision Annotation Tool)** software.

**Reference:** Boris Sekachev, Nikita Manovich, et al. (2020). Computer Vision Annotation Tool (CVAT) [Computer software]. Available at https://github.com/cvat-ai/cvat

## Repository Structure

### Core Components

- **DP/** - Main codebase directory
  - **train_code/** - Model training implementations and scripts
  - **predict_code/** - Inference pipeline for obstacle detection
  - **val_code/** - Model validation and evaluation scripts
  - **track_code/** - Object tracking implementation
  - **utils/** - Utility functions and helper modules

### Results and Outputs

- **runs/detect/** - Detection results, model training and validation results
- **val_results/** - Detailed validation metrics

## Technology Stack

- **Language:** Python
- **Core Framework:** Convolutional Neural Networks - YOLO Ultralytics (TODO: reference)
- **Application:** Off-road obstacle detection and object tracking

## Key Features

- End-to-end CNN architecture for object detection
- Robust performance in challenging off-road environments
- Integrated object tracking capabilities
- Comprehensive validation pipeline with detailed metrics
- Modular code structure for easy customization and extension
