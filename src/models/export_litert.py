#!/usr/bin/env python3
"""
YOLOv11 to LiteRT Export Script

This script exports a trained YOLOv11 model to LiteRT format with INT8 quantization
for optimal performance on Android devices.

Usage:
    python src/models/export_litert.py --model models/best.pt --output models/
    python src/models/export_litert.py --model models/best.pt --output models/ --imgsz 640 --quantize
"""

import argparse
import os
import sys
from pathlib import Path
from ultralytics import YOLO


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Export YOLOv11 model to LiteRT format with quantization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model checkpoint (.pt file)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Directory path where to save the exported LiteRT model"
    )
    
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size for export (should match training size)"
    )
    
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Enable INT8 quantization for maximum mobile efficiency"
    )
    
    parser.add_argument(
        "--data",
        type=str,
        default="",
        help="Path to dataset YAML for calibration (required for quantization)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use for export (cpu recommended for compatibility)"
    )
    
    parser.add_argument(
        "--batch",
        type=int,
        default=1,
        help="Batch size for export (1 recommended for mobile)"
    )
    
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Apply additional optimizations for mobile deployment"
    )
    
    parser.add_argument(
        "--simplify",
        action="store_true",
        help="Simplify model for better compatibility"
    )
    
    return parser.parse_args()


def validate_inputs(model_path: str, output_path: str, quantize: bool, data_path: str):
    """Validate input parameters and paths."""
    # Check if model exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Create output directory if it doesn't exist
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check dataset path if quantization is enabled
    if quantize and data_path and not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset YAML file not found: {data_path}")
    
    if quantize and not data_path:
        print("Warning: Quantization enabled but no dataset provided. Using default calibration.")
    
    print(f"Model: {model_path}")
    print(f"Output directory: {output_path}")
    print(f"Quantization: {'Enabled' if quantize else 'Disabled'}")
    if quantize and data_path:
        print(f"Calibration dataset: {data_path}")


def export_to_litert(model: YOLO, output_path: str, imgsz: int, quantize: bool, 
                     data_path: str, device: str, batch: int, optimize: bool, simplify: bool):
    """Export model to LiteRT format with specified options."""
    print(f"\nExporting model to LiteRT format...")
    
    # Prepare export arguments
    export_args = {
        "format": "tflite",
        "imgsz": imgsz,
        "device": device,
        "batch": batch,
        "optimize": optimize,
        "simplify": simplify,
    }
    
    # Add quantization settings
    if quantize:
        export_args["int8"] = True
        if data_path:
            export_args["data"] = data_path
        print("Export configuration:")
    for key, value in export_args.items():
        print(f"  {key}: {value}")
    
    try:
        # Perform export
        export_path = model.export(**export_args)
        
        # Move exported model to desired output location
        if export_path and os.path.exists(export_path):
            model_name = os.path.splitext(os.path.basename(model.ckpt_path or "model"))[0]
            suffix = "_int8" if quantize else ""
            output_filename = f"{model_name}{suffix}.tflite"
            final_output_path = os.path.join(output_path, output_filename)
            
            # Copy to final location
            import shutil
            shutil.copy2(export_path, final_output_path)
            
            return final_output_path, export_path
        else:
            raise RuntimeError("Export failed - no output file generated")
            
    except Exception as e:
        raise RuntimeError(f"Export failed: {str(e)}")


def get_model_info(model_path: str):
    """Get information about the exported model."""
    try:
        import tensorflow as tf
        
        # Load the TFLite model
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        
        # Get input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print("\nModel Information:")
        print(f"  File size: {os.path.getsize(model_path) / (1024*1024):.2f} MB")
        
        print("  Input details:")
        for i, detail in enumerate(input_details):
            print(f"    Input {i}: {detail['name']} - Shape: {detail['shape']} - Type: {detail['dtype']}")
        
        print("  Output details:")
        for i, detail in enumerate(output_details):
            print(f"    Output {i}: {detail['name']} - Shape: {detail['shape']} - Type: {detail['dtype']}")
            
        return True
        
    except ImportError:
        print("Warning: TensorFlow not available for model inspection")
        return False
    except Exception as e:
        print(f"Warning: Could not inspect model: {e}")
        return False


def main():
    """Main export function."""
    args = parse_arguments()
    
    try:
        # Validate inputs
        validate_inputs(args.model, args.output, args.quantize, args.data)
        
        # Load model
        print(f"Loading model: {args.model}")
        model = YOLO(args.model)
        
        # Export to LiteRT
        final_path, temp_path = export_to_litert(
            model, args.output, args.imgsz, args.quantize,
            args.data, args.device, args.batch, args.optimize, args.simplify
        )
        
        print("\nExport completed successfully!")
        print(f"LiteRT model saved to: {final_path}")
        
        # Get model information
        get_model_info(final_path)
        
        # Clean up temporary file if different from final path
        if temp_path != final_path and os.path.exists(temp_path):
            os.remove(temp_path)
        
        # Provide Android integration guidance
        print("\nAndroid Integration:")
        print(f"1. Copy {os.path.basename(final_path)} to your Android app's assets folder")
        print("2. Use TensorFlow Lite Android API to load the model")
        print(f"3. Input image size: {args.imgsz}x{args.imgsz}")
        print(f"4. Model type: {'INT8 Quantized' if args.quantize else 'FP32'}")
        print("5. Expected input format: RGB images normalized to [0,1]")
        print("6. Output format: YOLO detection format (boxes, scores, classes)")
        
    except Exception as e:
        print(f"Error during export: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
