# Earth Terrain Image Segmentation using U-Net

A deep learning-based image segmentation project for generating segmentation masks from Earth terrain images using a CPU-friendly U-Net architecture.

## Project Overview

This project uses a U-Net convolutional neural network to perform pixel-level image segmentation on Earth terrain images.

The model takes a terrain image as input and predicts a corresponding binary segmentation mask.

The complete pipeline includes:

- Dataset preparation
- Image and mask pairing
- Image preprocessing and resizing
- Normalization
- Train-validation split
- U-Net model development
- Model training
- Dice Score evaluation
- IoU evaluation
- Visual comparison of actual and predicted masks

## Dataset

The dataset used for this project is:

**Earth Terrain, Height, and Segmentation Map Images**

Dataset source:

https://www.kaggle.com/datasets/tpapp157/earth-terrain-height-and-segmentation-map-images

The dataset contains terrain images along with their corresponding segmentation maps.

For this project, 5,000 image-mask pairs were used for model training and validation.

## Model Architecture

The project uses a lightweight U-Net architecture designed to train efficiently on CPU.

### Input

- Image size: `128 × 128`
- Channels: `3`
- Input shape: `(128, 128, 3)`

### Architecture

The network consists of:

- Encoder blocks
- Max pooling layers
- Bottleneck layer
- Decoder blocks
- Skip connections
- Final 1×1 convolution
- Sigmoid activation for binary segmentation

The model uses a smaller number of filters than a standard large U-Net to keep CPU training practical while maintaining good segmentation performance.

## Training

The model was trained using:

- Framework: TensorFlow / Keras
- Optimizer: Adam
- Loss function: Binary Crossentropy
- Batch size: 4
- Maximum epochs: 20
- Input resolution: 128 × 128
- Hardware: CPU

Early stopping, learning-rate reduction, and model checkpointing were used during training.

## Results

The trained model achieved the following results on the validation dataset:

| Metric | Score |
|--------|-------|
| Validation Dice Score | **99.70%** |
| Validation IoU | **99.39%** |

### Dice Score

The Dice Score measures the overlap between the predicted segmentation mask and the ground-truth mask.

**Validation Dice Score: 0.99696**

### Intersection over Union

IoU measures the intersection between the predicted and actual masks relative to their union.

**Validation IoU: 0.99394**

## Sample Predictions

The model's predictions were compared against the ground-truth segmentation masks.

The predicted masks closely follow the actual segmentation boundaries across the validation samples.

![alt text](<Screenshot 2026-09-19 190742.png>)

## Project Structure

```text
image-segmentation/
│
├── segmentation_model_final.keras
├── image_segmentation.ipynb
├── README.md
└── .gitignore