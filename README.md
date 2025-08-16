# DNN Object Detection for Mobile (LiteRT)

A deep neural network project for object detection optimized for mobile devices using YOLO11 and LiteRT format.

## Overview

This project fine-tunes a YOLO11 model for object detection using custom datasets captured with mobile cameras. The final model is exported to LiteRT format for efficient on-device inference, outputting bounding boxes for detected objects.

## Features

- **Custom Dataset Processing**: Converts semantic masks to YOLO-format bounding box annotations
- **YOLO11 Fine-tuning**: Leverages state-of-the-art object detection architecture
- **Mobile Optimization**: Exports to LiteRT format for Android integration
- **Automated Pipeline**: Complete workflow from data preprocessing to model deployment

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
- **Dataset B**: 720x960 images with masks
- **Format**: Bitmap masks indicating object presence

### Processed Data
- **Location**: `data/boxed_640x640/`
- **Format**: 640x640 pixel cutouts with YOLO annotations
- **Split**: 80/10/10 (train/val/test)
- **Labels**: Normalized bounding boxes in YOLO format
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
Convert semantic masks to YOLO-format dataset:
```bash
python src/data/preprocess.py
```

#### 2. Model Training
Train the YOLO11 model:
```bash
python src/models/train.py
```

#### 3. Model Evaluation
Evaluate model performance and visualize predictions:
```bash
python src/models/eval.py
```

#### 4. Verify Dataset (Optional)
Visualize images with bounding boxes for quality control:
```bash
python src/data/verify_labels.py val <image_filename>
```

#### 5. Export to LiteRT
Convert trained model to optimized LiteRT format:
```bash
python src/models/export_litert.py
```

## Model Architecture

**YOLO11** was selected for its:
- State-of-the-art performance on mobile and edge devices
- Optimal accuracy-to-efficiency ratio for object detection
- Pre-trained weights for effective transfer learning
- Built-in support for LiteRT optimization and export

## Key Processing Parameters

- `MIN_SEGMENT_DIST = 64`: Minimum distance between object segments (10% of cutout size)
- `MIN_CUTOUT_DIST = 64`: Minimum distance between cutout centers (10% of cutout size)
- **Image Size**: 640x640 pixels for training and inference
- **Data Split**: 80% training, 10% validation, 10% testing

## Mobile Integration

The exported LiteRT model is optimized for Android integration with:
- Full INT8 quantization for maximum efficiency
- Optimized inference pipeline for mobile hardware
- Bounding box output format compatible with mobile apps

## Project Milestones

- [x] Project setup and specifications
- [x] Dataset preprocessing (modular implementation completed)
- [x] Dataset verification tools
- [ ] Model training
- [ ] Model evaluation
- [ ] LiteRT export and quantization
- [ ] Android app integration
- [ ] Mobile testing and validation

## Verification and Evaluation Tools

### Dataset Verification
The `verify_labels.py` script helps validate image-label consistency:

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
- Visualizes bounding boxes with customizable labels
- Saves verification images to `data/evaluation/boxed_640x640/<split>/`
- Provides detailed console output with box coordinates and sizes
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

This project is developed for mobile object detection research and development.