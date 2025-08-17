# YOLOv11 Poo Detection for Mobile (LiteRT)

A deep neural network project for poo detection optimized for mobile devices using YOLOv11 and LiteRT format.

## Overview

This project implements a YOLOv11-based poo detection system optimized for mobile deployment using LiteRT. The model is trained on a custom dataset containing poo examples and exported to LiteRT format for efficient on-device inference.

## Features

- **Custom Dataset**: Preprocessed from semantic masks containing poo annotations to YOLO format bounding box annotations
- **YOLOv11 Fine-tuning**: Leverages state-of-the-art object detection architecture
- **Mobile Optimization**: Exports to LiteRT format for Android integration
- **Automated Pipeline**: Complete workflow from poo data preprocessing to model deployment

## Project Structure

```
project_root/
├── data/
│   ├── boxed_640x640/          # Processed dataset (640x640 cutouts)
│   │   ├── images/             # Train/val/test image splits
│   │   ├── labels/             # YOLO-format annotations
│   │   ├── dataset.yaml        # YOLO training configuration
│   │   └── README.md           # Dataset documentation
│   ├── evaluation/             # Verification outputs
│   │   └── boxed_640x640/      # Visualized images with bounding boxes
│   └── semantic_masks/         # Source dataset
│       ├── images/             # Original images from cameras A & B
│       └── masks/              # Corresponding bitmap masks
├── models/                     # Trained model outputs
├── src/
│   ├── data/                   # Modular data processing utilities
│   │   ├── __init__.py         # Package initialization
│   │   ├── preprocess.py       # Main preprocessing pipeline
│   │   ├── mask_utils.py       # Object center detection utilities
│   │   ├── cutout_utils.py     # Image cutout generation utilities
│   │   ├── yolo_utils.py       # YOLO label conversion utilities
│   │   ├── dataset_utils.py    # Dataset configuration utilities
│   │   └── verify_labels.py    # Image-label verification tool
│   └── models/
│       ├── train.py            # Model training script
│       ├── eval.py             # Evaluation and visualization
│       └── export_litert.py    # LiteRT export/quantization
├── README.md
├── SPECS.md                    # Detailed project specifications
└── requirements.txt            # Python dependencies
```

## Dataset Information

### Source Data
- **Location**: `data/semantic_masks/`
- **Dataset A**: 1000x1000 pixel cutouts with masks
- **Dataset B**: 720x960 images with poo masks
- **Format**: Bitmap masks indicating poo presence

### Processed Data
- **Location**: `data/boxed_640x640/`
- **Format**: 640x640 pixel cutouts with YOLO poo annotations
- **Split**: 80/10/10 (train/val/test)
- **Labels**: Normalized poo bounding boxes in YOLO format
- **Class**: Single class 'poo' with ID 0 for fine-tuning

## Getting Started

### Prerequisites

- Python 3.12
- Virtual environment: `/home/pato/.venvs/deepoo-yolo11-litert`

### Installation

1. Activate the virtual environment:
```bash
source /home/pato/.venvs/deepoo-yolo11-litert/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Usage

#### 1. Data Preprocessing
Process semantic masks containing poo annotations to generate YOLO-format dataset:
```bash
python src/data/preprocess.py
```

#### 2. Model Training
Train YOLOv11 model on the processed poo dataset:
```bash
# Basic training with required arguments
python src/models/train.py --data data/boxed_640x640/dataset.yaml --output models/

# Advanced training with custom parameters
python src/models/train.py --data data/boxed_640x640/dataset.yaml --output models/ \
    --model yolo11s.pt --epochs 200 --batch 16 --device 0

# Training with early stopping and custom experiment name
python src/models/train.py --data data/boxed_640x640/dataset.yaml --output models/ \
    --patience 30 --name poo_detection_v1
```

#### 3. Model Evaluation
Evaluate poo detection performance and visualize predictions:
```bash
# Basic evaluation
python src/models/eval.py --model models/best.pt --data data/boxed_640x640/dataset.yaml

# Evaluation with visualization of 10 examples
python src/models/eval.py --model models/best.pt --data data/boxed_640x640/dataset.yaml --visualize 10

# Advanced evaluation with custom parameters
python src/models/eval.py --model models/best.pt --data data/boxed_640x640/dataset.yaml \
    --split test --visualize 5 --conf 0.3 --iou 0.5 --device 0
```

#### 4. Verify Dataset (Optional)
Visualize images with bounding boxes for quality control:
```bash
python src/data/verify_labels.py val <image_filename>
```

#### 5. Export to LiteRT
Convert trained model to optimized LiteRT format:
```bash
# Basic export to LiteRT
python src/models/export_litert.py --model models/best.pt --output models/

# Export with INT8 quantization for maximum mobile efficiency
python src/models/export_litert.py --model models/best.pt --output models/ --quantize --data data/boxed_640x640/dataset.yaml

# Advanced export with optimizations
python src/models/export_litert.py --model models/best.pt --output models/ \
    --quantize --data data/boxed_640x640/dataset.yaml --optimize --simplify
```

## Model Training

### Training Script Features
The `src/models/train.py` script provides a comprehensive training interface:

**Model Variants:**
- `yolo11n.pt`: Nano (fastest, smallest)
- `yolo11s.pt`: Small (balanced)
- `yolo11m.pt`: Medium (higher accuracy)
- `yolo11l.pt`: Large (best accuracy)
- `yolo11x.pt`: Extra Large (maximum accuracy)

**Key Parameters:**
- `--epochs`: Training epochs (default: 100)
- `--imgsz`: Input image size (default: 640)
- `--batch`: Batch size (-1 for auto-detection)
- `--device`: GPU/CPU selection (auto-detection if empty)
- `--patience`: Early stopping patience (default: 50)
- `--workers`: Data loading threads (default: 8)

**Features:**
- Automatic best model checkpoint saving
- Training metrics display and logging
- Path validation and error handling
- Experiment organization with project/name structure

## Model Architecture

**YOLOv11** was selected for its:
- State-of-the-art object detection performance on mobile and edge devices
- Optimal accuracy-to-efficiency ratio for real-time object detection
- Pre-trained weights for effective transfer learning
- Built-in support for LiteRT optimization and export

## Key Processing Parameters

- `MIN_SEGMENT_DIST = 64`: Minimum distance between object segments (10% of cutout size)
- `MIN_CUTOUT_DIST = 64`: Minimum distance between cutout centers (10% of cutout size)
- **Image Size**: 640x640 pixels for training and inference
- **Data Split**: 80% training, 10% validation, 10% testing

## LiteRT Export and Mobile Integration

### Export Features
The `export_litert.py` script provides comprehensive model export capabilities:

```bash
# Basic LiteRT export
python src/models/export_litert.py --model models/best.pt --output models/

# Quantized export for maximum mobile efficiency
python src/models/export_litert.py --model models/best.pt --output models/ --quantize --data data/boxed_640x640/dataset.yaml
```

**Export Options:**
- `--quantize`: Enable INT8 quantization for 4x smaller models
- `--optimize`: Apply mobile-specific optimizations
- `--simplify`: Simplify model graph for better compatibility
- `--imgsz`: Input image size (should match training)
- `--batch`: Batch size (1 recommended for mobile)

### Android Integration
The exported LiteRT model is optimized for Android integration with:
- Full INT8 quantization for maximum efficiency (up to 4x size reduction)
- Optimized inference pipeline for mobile hardware
- TensorFlow Lite Android API compatibility
- Automatic model inspection and integration guidance

**Integration Steps:**
1. Copy the exported `.tflite` file to your Android app's assets folder
2. Use TensorFlow Lite Android API to load the model
3. Input format: RGB images normalized to [0,1] at 640x640 resolution
4. Output format: YOLO detection format (bounding boxes, scores, classes)
5. Model supports real-time inference on mobile devices

## Project Milestones

- [x] Project setup and specifications
- [x] Dataset preprocessing (modular implementation completed)
- [x] Dataset verification tools
- [x] Model training (comprehensive CLI implementation completed)
- [x] Model evaluation (with visualization capabilities completed)
- [x] LiteRT export and quantization (with Android integration guidance)

## Verification and Evaluation Tools

### Model Evaluation
The `eval.py` script provides comprehensive model evaluation with visualization:

```bash
# Basic evaluation with metrics
python src/models/eval.py --model models/best.pt --data data/boxed_640x640/dataset.yaml

# Evaluation with prediction visualization
python src/models/eval.py --model models/best.pt --data data/boxed_640x640/dataset.yaml --visualize 10
```

**Features:**
- Comprehensive evaluation metrics using Ultralytics API
- Configurable confidence and IoU thresholds
- Random sample visualization from any dataset split
- Side-by-side ground truth vs prediction comparison
- Automatic output organization with experiment tracking
- Shared visualization utilities with verification script

**Key Parameters:**
- `--visualize N`: Number of examples to visualize (0 to disable)
- `--split`: Dataset split to evaluate (train/val/test)
- `--conf`: Confidence threshold for predictions (default: 0.25)
- `--iou`: IoU threshold for NMS (default: 0.45)
- `--output`: Directory for visualization outputs

### Dataset Verification
The `verify_labels.py` script helps validate poo image-label consistency:

```bash
# Basic usage
python src/data/verify_labels.py val image_001.jpg

# Hide all labels (show only bounding boxes)
python src/data/verify_labels.py val image_001.jpg --no-labels

# Show only class names (no indices)
python src/data/verify_labels.py val image_001.jpg --no-indices

# Show only class indices (no names)
python src/data/verify_labels.py val image_001.jpg --no-names
```

**Features:**
- Visualizes poo bounding boxes with customizable labels
- Saves verification images to `data/evaluation/boxed_640x640/<split>/`
- Provides detailed console output with poo box coordinates and sizes
- Supports train/val/test splits

## Dependencies

Key libraries used in this project:
- **numpy==2.3.2**: Numerical computing
- **opencv-python==4.12.0.88**: Image processing and computer vision
- **pyyaml==6.0.2**: Configuration file handling
- **scikit-learn==1.7.1**: Data splitting utilities
- **pillow==11.3.0**: Image processing for verification
- **ultralytics**: YOLO11 implementation and training
- **Python 3.12**: Core runtime environment

See `requirements.txt` for complete dependency list.

## License

This project is developed for mobile poo detection research and development.