#!/usr/bin/env python3
"""
YOLOv11 Model Evaluation Script

This script evaluates a trained YOLOv11 model using the Ultralytics API and provides
visualization capabilities for model predictions on dataset samples.

Usage:
    python src/models/eval.py --model models/best.pt --data data/boxed_640x640/dataset.yaml
    python src/models/eval.py --model models/best.pt --data data/boxed_640x640/dataset.yaml --visualize 10
"""

import argparse
import os
import sys
import random
from pathlib import Path
from typing import List
from ultralytics import YOLO
import yaml

# Import visualization utilities from verify_labels
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data'))
from verify_labels import draw_bounding_boxes, parse_yolo_labels

try:
    from PIL import Image
except ImportError:
    print("Error: PIL (Pillow) is required. Install with: pip install pillow")
    sys.exit(1)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate YOLOv11 model and visualize predictions",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model checkpoint (.pt file)"
    )
    
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to dataset YAML configuration file"
    )
    
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate on"
    )
    
    parser.add_argument(
        "--visualize",
        type=int,
        default=0,
        help="Number of examples to visualize with predictions (0 to disable)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="data/evaluation/predictions",
        help="Directory to save visualization outputs"
    )
    
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for predictions"
    )
    
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="IoU threshold for NMS"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Device to use for evaluation (e.g., 0, 1, cpu). Empty string for auto-detection"
    )
    
    parser.add_argument(
        "--save-txt",
        action="store_true",
        help="Save prediction results in YOLO format"
    )
    
    parser.add_argument(
        "--save-conf",
        action="store_true",
        help="Save confidence scores in prediction files"
    )
    
    return parser.parse_args()


def load_dataset_config(data_path: str) -> dict:
    """Load dataset configuration from YAML file."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset YAML file not found: {data_path}")
    
    with open(data_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def get_random_images(dataset_path: str, split: str, n_samples: int) -> List[str]:
    """Get random image filenames from the specified split."""
    images_dir = os.path.join(dataset_path, 'images', split)
    
    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    
    # Get all image files
    image_files = [f for f in os.listdir(images_dir) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not image_files:
        raise ValueError(f"No image files found in {images_dir}")
    
    # Return random sample
    n_samples = min(n_samples, len(image_files))
    return random.sample(image_files, n_samples)


def visualize_predictions(model: YOLO, dataset_config: dict, dataset_path: str, 
                         split: str, n_samples: int, output_dir: str, 
                         conf_threshold: float, iou_threshold: float):
    """Visualize model predictions on random dataset samples."""
    print(f"\nVisualizing {n_samples} predictions from {split} split...")
    
    # Create output directory
    output_path = Path(output_dir) / split
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get random images
    image_files = get_random_images(dataset_path, split, n_samples)
    
    # Get class names from dataset config
    class_names = {i: name for i, name in enumerate(dataset_config.get('names', []))}
    
    for i, image_file in enumerate(image_files):
        print(f"Processing {i+1}/{len(image_files)}: {image_file}")
        
        # Construct paths
        image_path = os.path.join(dataset_path, 'images', split, image_file)
        label_file = os.path.splitext(image_file)[0] + '.txt'
        label_path = os.path.join(dataset_path, 'labels', split, label_file)
        
        # Load image
        try:
            image = Image.open(image_path)
            img_width, img_height = image.size
        except Exception as e:
            print(f"Error loading image {image_file}: {e}")
            continue
        
        # Get ground truth labels
        gt_boxes = parse_yolo_labels(label_path, img_width, img_height)
        
        # Run prediction
        try:
            results = model.predict(
                image_path,
                conf=conf_threshold,
                iou=iou_threshold,
                verbose=False
            )
            
            # Extract prediction boxes
            pred_boxes = []
            if results and len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                for j in range(len(boxes)):
                    # Get box coordinates (xyxy format)
                    x1, y1, x2, y2 = boxes.xyxy[j].cpu().numpy()
                    class_id = int(boxes.cls[j].cpu().numpy())
                    confidence = float(boxes.conf[j].cpu().numpy())
                    
                    pred_boxes.append((class_id, int(x1), int(y1), int(x2), int(y2), confidence))
        
        except Exception as e:
            print(f"Error running prediction on {image_file}: {e}")
            pred_boxes = []
        
        # Create visualization with ground truth
        img_with_gt = draw_bounding_boxes(
            image, gt_boxes, class_names,
            show_labels=True, show_names=True, show_indices=False
        )
        
        # Create visualization with predictions
        # Convert pred_boxes format to match draw_bounding_boxes expected format
        pred_boxes_for_draw = [(class_id, x1, y1, x2, y2) for class_id, x1, y1, x2, y2, _ in pred_boxes]
        img_with_pred = draw_bounding_boxes(
            image, pred_boxes_for_draw, class_names,
            show_labels=True, show_names=True, show_indices=False
        )
        
        # Save visualizations
        base_name = os.path.splitext(image_file)[0]
        gt_output_path = output_path / f"{base_name}_gt.jpg"
        pred_output_path = output_path / f"{base_name}_pred.jpg"
        
        img_with_gt.save(gt_output_path)
        img_with_pred.save(pred_output_path)
        
        # Print prediction summary
        print(f"  Ground truth boxes: {len(gt_boxes)}")
        print(f"  Predicted boxes: {len(pred_boxes)}")
        if pred_boxes:
            avg_conf = sum(conf for _, _, _, _, _, conf in pred_boxes) / len(pred_boxes)
            print(f"  Average confidence: {avg_conf:.3f}")
        
        print(f"  Saved: {gt_output_path}")
        print(f"  Saved: {pred_output_path}")


def main():
    """Main evaluation function."""
    args = parse_arguments()
    
    try:
        # Validate model path
        if not os.path.exists(args.model):
            raise FileNotFoundError(f"Model file not found: {args.model}")
        
        # Load dataset configuration
        dataset_config = load_dataset_config(args.data)
        dataset_path = dataset_config.get('path', os.path.dirname(args.data))
        
        print(f"Loading model: {args.model}")
        print(f"Dataset configuration: {args.data}")
        print(f"Dataset path: {dataset_path}")
        print(f"Evaluation split: {args.split}")
        
        # Load model
        model = YOLO(args.model)
        
        # Set device if specified
        if args.device:
            model.to(args.device)
        
        # Run evaluation
        print("\nRunning model evaluation...")
        eval_args = {
            "data": args.data,
            "split": args.split,
            "conf": args.conf,
            "iou": args.iou,
            "save_txt": args.save_txt,
            "save_conf": args.save_conf,
        }
        
        # Add device if specified
        if args.device:
            eval_args["device"] = args.device
        
        print("\nEvaluation Configuration:")
        for key, value in eval_args.items():
            print(f"  {key}: {value}")
        
        # Run validation
        results = model.val(**eval_args)
        
        # Print evaluation metrics
        print("\nEvaluation Results:")
        if hasattr(results, 'results_dict'):
            metrics = results.results_dict
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    print(f"  {key}: {value:.4f}")
        
        # Visualize predictions if requested
        if args.visualize > 0:
            visualize_predictions(
                model, dataset_config, dataset_path, args.split,
                args.visualize, args.output, args.conf, args.iou
            )
            print(f"\nVisualization outputs saved to: {args.output}/{args.split}/")
        
        print("\nEvaluation completed successfully!")
        
    except Exception as e:
        print(f"Error during evaluation: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()