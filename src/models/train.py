#!/usr/bin/env python3
"""
YOLOv11 Model Training Script

This script trains a YOLOv11 model for object detection following the Ultralytics documentation.
It accepts a dataset path and output path for saving the best checkpoint.

Usage:
    python src/models/train.py --data path/to/dataset.yaml --output path/to/save/model
    python src/models/train.py --data data/boxed_640x640/dataset.yaml --output models/
"""

import argparse
import os
import sys
from pathlib import Path
from ultralytics import YOLO


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train YOLOv11 model for object detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to dataset YAML configuration file"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Directory path where to save the best model checkpoint"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11n.pt",
        help="Model to use for training (yolo11n.pt, yolo11s.pt, yolo11m.pt, yolo11l.pt, yolo11x.pt)"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size for training"
    )
    
    parser.add_argument(
        "--batch",
        type=int,
        default=-1,
        help="Batch size (-1 for auto batch size)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Device to use for training (e.g., 0, 1, cpu). Empty string for auto-detection"
    )
    
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of worker threads for data loading"
    )
    
    parser.add_argument(
        "--patience",
        type=int,
        default=50,
        help="Epochs to wait for no improvement before early stopping"
    )
    
    parser.add_argument(
        "--save-period",
        type=int,
        default=10,
        help="Save checkpoint every n epochs"
    )
    
    parser.add_argument(
        "--project",
        type=str,
        default="runs/detect",
        help="Project directory for saving runs"
    )
    
    parser.add_argument(
        "--name",
        type=str,
        default="train",
        help="Experiment name"
    )
    
    return parser.parse_args()


def validate_paths(data_path: str, output_path: str):
    """Validate input and output paths."""
    # Check if dataset YAML exists
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset YAML file not found: {data_path}")
    
    # Create output directory if it doesn't exist
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Dataset configuration: {data_path}")
    print(f"Model output directory: {output_path}")


def main():
    """Main training function."""
    args = parse_arguments()
    
    try:
        # Validate paths
        validate_paths(args.data, args.output)
        
        # Load model
        print(f"Loading model: {args.model}")
        model = YOLO(args.model)  # Load pretrained model (recommended for training)
        
        # Configure training parameters
        train_args = {
            "data": args.data,
            "epochs": args.epochs,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "workers": args.workers,
            "patience": args.patience,
            "save_period": args.save_period,
            "project": args.project,
            "name": args.name,
        }
        
        # Add device if specified
        if args.device:
            train_args["device"] = args.device
        
        print("\nTraining Configuration:")
        for key, value in train_args.items():
            print(f"  {key}: {value}")
        print()
        
        # Train the model
        print("Starting training...")
        results = model.train(**train_args)
        
        # Get the best model path from training results
        best_model_path = model.trainer.best
        
        # Copy best model to specified output directory
        if best_model_path and os.path.exists(best_model_path):
            output_model_path = os.path.join(args.output, "best.pt")
            
            # Copy the best model
            import shutil
            shutil.copy2(best_model_path, output_model_path)
            
            print("\nTraining completed successfully!")
            print(f"Best model saved to: {output_model_path}")
            print(f"Training results saved in: {model.trainer.save_dir}")
            
            # Print training metrics if available
            if hasattr(results, 'results_dict'):
                metrics = results.results_dict
                print("\nFinal Training Metrics:")
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        print(f"  {key}: {value:.4f}")
        else:
            print("Warning: Best model not found. Check training logs.")
            
    except Exception as e:
        print(f"Error during training: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()