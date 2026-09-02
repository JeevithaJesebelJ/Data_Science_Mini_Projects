# License Plate Detection using YOLO11

A computer vision project that detects license plates in vehicle images using YOLO11 object detection.

## 📌 Project Overview

This project uses a dataset containing vehicle images and Pascal VOC XML annotations. The annotations were converted into YOLO format, followed by training a YOLO11 Nano model for license plate detection.

The project includes:

- XML annotation processing
- Bounding box extraction
- Pascal VOC to YOLO format conversion
- Train-validation dataset splitting
- YOLO11 Nano model training
- Model evaluation
- License plate prediction and visualization

## 📂 Dataset

- Total annotated images: **225**
- Training images: **180**
- Validation images: **45**
- Number of classes: **1**
- Class: **License Plate**

The original annotations were provided in Pascal VOC XML format.

## 🤖 Model

The final model uses **YOLO11 Nano**, a lightweight object detection model suitable for efficient training and inference.

Training was performed using CPU.

## 📊 Model Performance

The trained YOLO model achieved approximately:

| Metric | Score |
|--------|-------|
| Precision | 87.7% |
| Recall | 97.8% |
| mAP@50 | 97.3% |
| mAP@50-95 | 63.5% |

## 🔄 Workflow

```text
Vehicle Images + XML Annotations
            ↓
Bounding Box Extraction
            ↓
Convert Pascal VOC XML → YOLO Format
            ↓
Train / Validation Split
            ↓
YOLO11 Nano Training
            ↓
Model Evaluation
            ↓
License Plate Detection