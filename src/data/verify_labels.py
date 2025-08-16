#!/usr/bin/env python3
"""
Image-label verification script for YOLO dataset.

This script loads an image and its corresponding YOLO labels, draws bounding boxes,
and saves the visualization for manual inspection.
"""

import os
import argparse
from PIL import Image, ImageDraw, ImageFont
from typing import List, Tuple


def parse_yolo_labels(label_path: str, img_width: int, img_height: int) -> List[Tuple[int, int, int, int, int]]:
    """
    Parse YOLO format labels and convert to pixel coordinates.
    
    Args:
        label_path: Path to the YOLO label file
        img_width: Image width in pixels
        img_height: Image height in pixels
        
    Returns:
        List of tuples (class_id, x1, y1, x2, y2) in pixel coordinates
    """
    boxes = []
    
    if not os.path.exists(label_path):
        print(f"Warning: Label file not found: {label_path}")
        return boxes
    
    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            parts = line.split()
            if len(parts) != 5:
                print(f"Warning: Invalid label format: {line}")
                continue
            
            try:
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                
                # Convert normalized coordinates to pixel coordinates
                x_center_px = x_center * img_width
                y_center_px = y_center * img_height
                width_px = width * img_width
                height_px = height * img_height
                
                # Calculate bounding box corners
                x1 = int(x_center_px - width_px / 2)
                y1 = int(y_center_px - height_px / 2)
                x2 = int(x_center_px + width_px / 2)
                y2 = int(y_center_px + height_px / 2)
                
                boxes.append((class_id, x1, y1, x2, y2))
                
            except ValueError as e:
                print(f"Warning: Error parsing label: {line}, Error: {e}")
                continue
    
    return boxes


def draw_bounding_boxes(image: Image.Image, boxes: List[Tuple[int, int, int, int, int]], 
                       class_names: dict = None, show_labels: bool = True, 
                       show_names: bool = True, show_indices: bool = True) -> Image.Image:
    """
    Draw bounding boxes on the image.
    
    Args:
        image: PIL Image object
        boxes: List of bounding boxes (class_id, x1, y1, x2, y2)
        class_names: Dictionary mapping class_id to class name
        show_labels: Whether to show any labels at all
        show_names: Whether to show class names (requires show_labels=True)
        show_indices: Whether to show class indices (requires show_labels=True)
        
    Returns:
        Image with bounding boxes drawn
    """
    if not boxes:
        return image
    
    # Create a copy to avoid modifying the original
    img_with_boxes = image.copy()
    draw = ImageDraw.Draw(img_with_boxes)
    
    # Define colors for different classes
    colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'cyan', 'magenta']
    
    # Try to load a font, fall back to default if not available
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except:
        try:
            font = ImageFont.load_default()
        except:
            font = None
    
    for class_id, x1, y1, x2, y2 in boxes:
        # Select color based on class_id
        color = colors[class_id % len(colors)]
        
        # Draw bounding box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        
        # Prepare label text if labels should be shown
        if show_labels:
            label_parts = []
            
            # Add class name if requested and available
            if show_names and class_names and class_id in class_names:
                label_parts.append(class_names[class_id])
            elif show_names:
                label_parts.append(f"Class")
            
            # Add class index if requested
            if show_indices:
                label_parts.append(f"({class_id})")
            
            # Create final label
            if label_parts:
                label = " ".join(label_parts)
                
                # Draw label background and text
                if font:
                    bbox = draw.textbbox((x1, y1), label, font=font)
                    draw.rectangle(bbox, fill=color)
                    draw.text((x1, y1), label, fill='white', font=font)
                else:
                    # Fallback without font
                    draw.text((x1, y1-10), label, fill=color)
    
    return img_with_boxes


def verify_image_labels(split: str, image_filename: str, dataset_path: str = "data/boxed_640x640",
                       show_labels: bool = True, show_names: bool = True, show_indices: bool = True):
    """
    Verify image-label consistency by visualizing bounding boxes.
    
    Args:
        split: Dataset split name (train, val, test)
        image_filename: Name of the image file (with or without extension)
        dataset_path: Path to the dataset directory
    """
    # Ensure image filename has extension
    if not image_filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        image_filename += '.jpg'
    
    # Construct paths
    image_path = os.path.join(dataset_path, 'images', split, image_filename)
    label_filename = os.path.splitext(image_filename)[0] + '.txt'
    label_path = os.path.join(dataset_path, 'labels', split, label_filename)
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Error: Image not found: {image_path}")
        return
    
    print(f"Loading image: {image_path}")
    print(f"Loading labels: {label_path}")
    
    # Load image
    try:
        image = Image.open(image_path)
        img_width, img_height = image.size
        print(f"Image size: {img_width}x{img_height}")
    except Exception as e:
        print(f"Error loading image: {e}")
        return
    
    # Parse labels
    boxes = parse_yolo_labels(label_path, img_width, img_height)
    print(f"Found {len(boxes)} bounding boxes")
    
    # Class names for our dataset
    class_names = {0: 'poo'}
    
    # Draw bounding boxes
    img_with_boxes = draw_bounding_boxes(image, boxes, class_names, show_labels, show_names, show_indices)
    
    # Create output directory
    output_dir = os.path.join("data/evaluation/boxed_640x640", split)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save visualization
    output_filename = f"verified_{image_filename}"
    output_path = os.path.join(output_dir, output_filename)
    
    try:
        img_with_boxes.save(output_path)
        print(f"Visualization saved to: {output_path}")
    except Exception as e:
        print(f"Error saving visualization: {e}")
        return
    
    # Display summary
    if boxes:
        print("\nBounding box details:")
        for i, (class_id, x1, y1, x2, y2) in enumerate(boxes):
            width = x2 - x1
            height = y2 - y1
            class_name = class_names.get(class_id, f"Class {class_id}")
            print(f"  Box {i+1}: {class_name} at ({x1}, {y1}) - ({x2}, {y2}), size: {width}x{height}")
    else:
        print("No bounding boxes found in the label file.")


def main():
    """Main function with command-line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Verify image-label consistency by visualizing bounding boxes"
    )
    parser.add_argument(
        "split", 
        choices=['train', 'val', 'test'],
        help="Dataset split name"
    )
    parser.add_argument(
        "image_filename",
        help="Image filename (with or without extension)"
    )
    parser.add_argument(
        "--dataset-path",
        default="data/boxed_640x640",
        help="Path to the dataset directory (default: data/boxed_640x640)"
    )
    parser.add_argument(
        "--no-labels",
        action="store_true",
        help="Hide all class labels on bounding boxes"
    )
    parser.add_argument(
        "--no-names",
        action="store_true", 
        help="Hide class names (keep indices only)"
    )
    parser.add_argument(
        "--no-indices",
        action="store_true",
        help="Hide class indices (keep names only)"
    )
    
    args = parser.parse_args()
    
    print(f"Verifying image-label consistency...")
    print(f"Split: {args.split}")
    print(f"Image: {args.image_filename}")
    print(f"Dataset path: {args.dataset_path}")
    print("-" * 50)
    
    # Parse label display options
    show_labels = not args.no_labels
    show_names = not args.no_names
    show_indices = not args.no_indices
    
    verify_image_labels(args.split, args.image_filename, args.dataset_path, 
                       show_labels, show_names, show_indices)


if __name__ == "__main__":
    main()
