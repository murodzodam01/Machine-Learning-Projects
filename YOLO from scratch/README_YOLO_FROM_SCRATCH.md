# YOLO Object Detector From Scratch

This README accompanies A from-scratch YOLOv1-style object detector implemented with PyTorch and PyTorch Lightning. The model is trained to detect three fruit classes:

- Apple
- Orange
- Banana

## What the notebook contains

The notebook covers the complete basic detection pipeline:

1. Downloads the fruit image dataset and XML annotations.
2. Converts Pascal VOC bounding boxes to normalized `(x_center, y_center, width, height)` coordinates.
3. Builds `7 × 7 × 13` YOLO target tensors.
4. Defines the convolutional YOLO model.
5. Implements IoU and the YOLOv1 loss.
6. Trains the model with PyTorch Lightning.
7. Runs inference on a validation image and visualizes predicted boxes.

For each grid cell, the 13 output values are arranged as follows:

| Indices | Meaning |
| --- | --- |
| `0:3` | Class scores |
| `3` | Confidence for bounding box 1 |
| `4:8` | Coordinates for bounding box 1 |
| `8` | Confidence for bounding box 2 |
| `9:13` | Coordinates for bounding box 2 |

## Requirements

The notebook is designed to run in Google Colab or another Jupyter environment with Python and PyTorch installed. A CUDA GPU is recommended for training.

Main dependencies:

```text
torch
torchvision
pytorch-lightning
albumentations
opencv-python
xmltodict
pandas
numpy
Pillow
matplotlib
scikit-learn
gdown
```

The notebook installs `xmltodict`, `pytorch-lightning`, and `gdown`. Depending on the environment, the remaining packages may already be available.

## How to run

1. Open `YOLO_FROM_SCRATCH.ipynb` in Google Colab or Jupyter.
2. If possible, enable a GPU runtime.
3. Run the cells from top to bottom.
4. Allow the dataset download and extraction cell to finish.
5. Run the training cell before running prediction.
6. Run the final visualization cells.

Training is configured for 20 epochs with a batch size of 4. This can take a significant amount of time on CPU.

## Visualization

The final image uses:

- Green rectangles for ground-truth boxes.
- Red rectangles for predicted boxes.

Predictions with confidence below `0.25` are not displayed. This threshold can be changed in the final visualization cell:

```python
if confidence < 0.25:
    continue
```

## Notes and limitations

- The project is educational and does not reproduce every detail of the original YOLOv1 implementation.
- Only one object can be stored as the target for a particular grid cell. Additional objects whose centers fall in the same cell are ignored.
- Good results depend on successful training. Running only the prediction cells without first training the model will not produce meaningful detections.

## Files

- `YOLO_FROM_SCRATCH.ipynb` — corrected notebook.
- `README_YOLO_FROM_SCRATCH.md` — project documentation.
