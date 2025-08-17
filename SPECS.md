# Project Specification: Poo Detection for Mobile (LiteRT)

## 1. Overview
Fine-tune a YOLOv11 model for poo detection using a custom dataset captured with mobile cameras. The final model is exported to LiteRT format optimized for on-device inference, outputting bounding boxes for detected poo objects.

## 2. Datasets

### Source Dataset
- **Location:** data/semantic_masks
- **Original Datasets:**
  - **Dataset A:** 1000x1000 pixel cutouts from images from camera A and corresponding masks
  - **Dataset B:** 720x960 images from camera B and corresponding masks
- **Subfolders:**
   - **images:** images from camera A and B
   - **masks:** masks from camera A and B
- **Annotations:** Bitmap masks indicating poo presence

### Processed Dataset
- **Location:** data/boxed_640x640
- **Subfolders:**
   - **images:** 640x640 pixel cutouts derived from source images
   - **labels:** YOLO-format labels generated from the corresponding masks
- **Split:** 80/10/10 (train/val/test)


## 3. Dataset Processing Strategy
- The preprocessing script `src/data/preprocess.py` iterates over source masks to identify centers of all poo segments.
- Data is split into train/val/test (80/10/10) using a fixed seed for reproducibility.
- Centers closer than `MIN_SEGMENT_DIST` pixels are iteratively merged by replacing each pair with its midpoint until no pair violates the threshold.
- For each center, assign a 640x640 square whose center is as close as possible to the poo center while remaining fully inside the image.
- Filter cutout centers closer than `MIN_CUTOUT_DIST` pixels by merging to a midpoint that still keeps the 640x640 window inside image bounds.
- If no valid square exists for an image, generate a random 640x640 square inside the image.
- Crop image–mask pairs by the squares and save to:
  - `data/boxed_640x640/images/{train,val,test}`
  - `data/boxed_640x640/labels/{train,val,test}` (YOLO `*.txt` files)
- Label file format (per line): `class_id x_center y_center width height` with all values normalized to [0, 1].
- Class configuration: Single class 'poo' with ID 0 for fine-tuning from pre-trained YOLO11 weights.
- Also generate `data/boxed_640x640/dataset.yaml` (fields: `path`, `train`, `val`, `test`, `names`).
- Also generate `data/boxed_640x640/README.md` documenting the dataset.

Key constants used in preprocessing:
- `MIN_SEGMENT_DIST = 64` # 10% of the side length of the cutout
- `MIN_CUTOUT_DIST = 64` # 10% of the side length of the cutout

## 4. Model Architecture

### Selected Model: YOLO11
- **Chosen Model:** A model variant with the best accuracy-to-efficiency ratio for object detection on mobile and edge devices. 
- **Rationale:**
  - State of the art for object detection on mobile and edge devices
  - Strong accuracy-to-efficiency ratio for object detection
  - Pre-trained weights available for transfer learning
  - Infrastructure for optimization and export to LiteRT
- **Action:**
  - Use the Ultralytics `ultralytics` Python library to fine-tune, evaluate, and export the model to LiteRT.

## 5. Mobile Integration
- The exported LiteRT model is optimized for Android integration with full INT8 quantization for maximum efficiency
- Optimized inference pipeline for mobile hardware
- Bounding box output format compatible with mobile apps

## 6. Directory Structure
```
project_root/
├── data/
│   ├── boxed_640x640/
│   │   ├── images/
│   │   │   ├── test/
│   │   │   ├── train/
│   │   │   └── val/
│   │   └── labels/
│   │       ├── test/
│   │       ├── train/
│   │       └── val/
│   └── semantic_masks/
│       ├── images/
│       └── masks/
├── models/
├── src/
│   ├── data/
│   ├── models/
│   └── utils/
├── README.md
└── SPECS.md
```

## 7. Scripts and Utilities

### Data Processing
- `src/data/preprocess.py`: Main preprocessing pipeline (modular implementation)
- `src/data/mask_utils.py`: Poo center detection and merging utilities
- `src/data/cutout_utils.py`: Image cutout generation and filtering utilities
- `src/data/yolo_utils.py`: YOLO label format conversion utilities
- `src/data/dataset_utils.py`: Dataset configuration and management utilities
- `src/data/verify_labels.py`: Image-label verification and visualization tool

### Model Training and Export
- `src/models/train.py`: Comprehensive YOLOv11 training pipeline with CLI interface
  - Supports all YOLOv11 model variants (n, s, m, l, x)
  - Configurable training parameters (epochs, batch size, image size)
  - Auto batch size detection and device selection
  - Early stopping with patience control
  - Automatic best model checkpoint saving
  - Training metrics display and logging
- `src/models/eval.py`: Comprehensive evaluation script with Ultralytics API integration
  - Model performance metrics calculation
  - Configurable confidence and IoU thresholds
  - Random sample visualization from any dataset split
  - Ground truth vs prediction comparison
  - Shared visualization utilities with verify_labels.py
  - Automatic output organization and experiment tracking
- `src/models/export_litert.py`: Comprehensive LiteRT export script with quantization
  - INT8 quantization for maximum mobile efficiency (4x size reduction)
  - Mobile-specific optimizations and graph simplification
  - Automatic model inspection and Android integration guidance
  - Configurable export parameters (image size, batch size, device)
  - TensorFlow Lite compatibility validation

## 8. Dependencies
- Python 3.12
- numpy==2.3.2
- opencv-python==4.12.0.88
- pyyaml==6.0.2
- scikit-learn==1.7.1
- pillow==11.3.0
- ultralytics (for YOLO11 training and export)
- (Complete list in `requirements.txt`)

## 9. Verification and Evaluation Tools

### Dataset Verification
- `src/data/verify_labels.py`: Visualizes images with bounding boxes for manual inspection
- Usage: `python src/data/verify_labels.py <split> <image_filename>`
- CLI options: `--no-labels`, `--no-names`, `--no-indices` for customizing label display
- Outputs saved to: `data/evaluation/boxed_640x640/<split>/verified_<imagename>`

## 10. Training Configuration
The training script supports comprehensive configuration options:

### Command Line Interface
```bash
# Basic usage
python src/models/train.py --data data/boxed_640x640/dataset.yaml --output models/

# Full parameter example
python src/models/train.py \
    --data data/boxed_640x640/dataset.yaml \
    --output models/ \
    --model yolo11s.pt \
    --epochs 200 \
    --imgsz 640 \
    --batch 16 \
    --device 0 \
    --workers 8 \
    --patience 50 \
    --save-period 10 \
    --project runs/detect \
    --name poo_detection_experiment
```

### Training Parameters
- **Model Selection**: Choose from yolo11n.pt (fastest) to yolo11x.pt (most accurate)
- **Auto Configuration**: Automatic batch size and device detection
- **Early Stopping**: Configurable patience for optimal training
- **Checkpoint Management**: Automatic best model saving and periodic checkpoints
- **Experiment Tracking**: Organized output with project/name structure

## 11. Evaluation Configuration
The evaluation script supports comprehensive model assessment:

### Command Line Interface
```bash
# Basic evaluation
python src/models/eval.py --model models/best.pt --data data/boxed_640x640/dataset.yaml

# Evaluation with visualization
python src/models/eval.py --model models/best.pt --data data/boxed_640x640/dataset.yaml --visualize 10

# Advanced evaluation with custom parameters
python src/models/eval.py \
    --model models/best.pt \
    --data data/boxed_640x640/dataset.yaml \
    --split test \
    --visualize 5 \
    --conf 0.3 \
    --iou 0.5 \
    --device 0 \
    --output data/evaluation/predictions
```

### Evaluation Features
- **Metrics Calculation**: Comprehensive performance metrics via Ultralytics API
- **Visualization**: Random sample visualization with ground truth vs prediction comparison
- **Flexible Configuration**: Configurable confidence/IoU thresholds and dataset splits
- **Output Management**: Organized visualization outputs with experiment tracking
- **Shared Utilities**: Reuses visualization functions from verify_labels.py for consistency

## 13. LiteRT Export Configuration
The export script provides comprehensive model optimization for mobile deployment:

### Command Line Interface
```bash
# Basic LiteRT export
python src/models/export_litert.py --model models/best.pt --output models/

# Quantized export with calibration dataset
python src/models/export_litert.py --model models/best.pt --output models/ --quantize --data data/boxed_640x640/dataset.yaml

# Advanced export with all optimizations
python src/models/export_litert.py \
    --model models/best.pt \
    --output models/ \
    --quantize \
    --data data/boxed_640x640/dataset.yaml \
    --optimize \
    --simplify \
    --imgsz 640 \
    --batch 1
```

### Export Features
- **INT8 Quantization**: Reduces model size by up to 4x while maintaining accuracy
- **Mobile Optimizations**: Graph optimizations specifically for mobile hardware
- **Model Simplification**: Removes unnecessary operations for better compatibility
- **Automatic Validation**: TensorFlow Lite compatibility checking
- **Android Integration**: Automatic guidance for Android app integration

### Android Integration Guidance
- Automatic file size and format reporting
- Input/output tensor specifications
- Integration code examples and best practices
- Performance optimization recommendations

## 14. Milestones
1. ✅ Preprocess datasets (modular implementation completed)
2. ✅ Train model (comprehensive CLI implementation completed)
3. ✅ Evaluate model (with visualization capabilities completed)
4. ✅ Export model to LiteRT (with quantization and Android integration)

