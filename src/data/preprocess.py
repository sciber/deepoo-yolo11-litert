#!/usr/bin/env python3
"""
Data preprocessing script for converting semantic masks to YOLO-format dataset.

This script processes images and masks from data/semantic_masks/ and generates
640x640 cutouts with YOLO annotations in data/boxed_640x640/.
"""

import os
import cv2
import numpy as np
from typing import List, Dict
import random
from sklearn.model_selection import train_test_split

# Import utility modules
from mask_utils import find_object_centers, merge_close_centers
from cutout_utils import (
    find_valid_cutout,
    filter_cutout_centers,
    generate_random_cutout,
)
from yolo_utils import mask_to_yolo_labels
from dataset_utils import (
    create_dataset_yaml,
    create_dataset_readme,
    create_output_directories,
)

# Constants from SPECS.md
MIN_SEGMENT_DIST = 64  # 10% of the side length of the cutout
MIN_CUTOUT_DIST = 64  # 10% of the side length of the cutout
CUTOUT_SIZE = 640

# Set random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def process_image_mask_pair(
    img_path: str, mask_path: str, output_dir: str, img_filename: str
) -> List[Dict]:
    """
    Process a single image-mask pair and generate cutouts.

    Args:
        img_path: Path to the source image
        mask_path: Path to the source mask
        output_dir: Output directory for processed data
        img_filename: Base filename for outputs

    Returns:
        List of dictionaries containing cutout information
    """
    # Load image and mask
    image = cv2.imread(img_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if image is None or mask is None:
        print(f"Warning: Could not load {img_path} or {mask_path}")
        return []

    img_height, img_width = image.shape[:2]

    # Find object centers
    centers = find_object_centers(mask)

    # Merge close centers
    merged_centers = merge_close_centers(centers, MIN_SEGMENT_DIST)

    # Filter cutout centers
    filtered_centers = filter_cutout_centers(
        merged_centers, MIN_CUTOUT_DIST, img_height, img_width, CUTOUT_SIZE
    )

    cutout_info = []

    # Generate cutouts from filtered centers
    for i, center in enumerate(filtered_centers):
        x1, y1, x2, y2 = find_valid_cutout(center, img_height, img_width, CUTOUT_SIZE)

        # Crop image and mask
        img_crop = image[y1:y2, x1:x2]
        mask_crop = mask[y1:y2, x1:x2]

        # Generate YOLO labels
        yolo_labels = mask_to_yolo_labels(mask_crop, CUTOUT_SIZE)

        cutout_filename = f"{img_filename}_{i:03d}"
        cutout_info.append(
            {"image_crop": img_crop, "labels": yolo_labels, "filename": cutout_filename}
        )

    # If no valid cutouts, generate a random one
    if not cutout_info:
        x1, y1, x2, y2 = generate_random_cutout(img_height, img_width, CUTOUT_SIZE)
        img_crop = image[y1:y2, x1:x2]
        mask_crop = mask[y1:y2, x1:x2]
        yolo_labels = mask_to_yolo_labels(mask_crop, CUTOUT_SIZE)

        cutout_filename = f"{img_filename}_random"
        cutout_info.append(
            {"image_crop": img_crop, "labels": yolo_labels, "filename": cutout_filename}
        )

    return cutout_info


def main():
    """Main preprocessing pipeline."""
    print("Starting data preprocessing...")

    # Define paths
    source_dir = "data/semantic_masks"
    output_dir = "data/boxed_640x640"

    images_dir = os.path.join(source_dir, "images")
    masks_dir = os.path.join(source_dir, "masks")

    # Create output directories
    create_output_directories(output_dir)

    # Collect all image-mask pairs
    all_cutouts = []

    print("Processing image-mask pairs...")
    for img_file in os.listdir(images_dir):
        if not img_file.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        img_path = os.path.join(images_dir, img_file)

        # Find corresponding mask
        mask_file = img_file.replace(".jpg", ".png").replace(".jpeg", ".png")
        mask_path = os.path.join(masks_dir, mask_file)

        if not os.path.exists(mask_path):
            print(f"Warning: No mask found for {img_file}")
            continue

        # Process the pair
        img_basename = os.path.splitext(img_file)[0]
        cutouts = process_image_mask_pair(img_path, mask_path, output_dir, img_basename)
        all_cutouts.extend(cutouts)

        print(f"Processed {img_file}: {len(cutouts)} cutouts generated")

    print(f"Total cutouts generated: {len(all_cutouts)}")

    # Split data into train/val/test (80/10/10)
    if len(all_cutouts) == 0:
        print("Error: No cutouts generated!")
        return

    # First split: 80% train, 20% temp
    train_cutouts, temp_cutouts = train_test_split(
        all_cutouts, test_size=0.2, random_state=RANDOM_SEED
    )

    # Second split: 50% val, 50% test (from the 20% temp)
    val_cutouts, test_cutouts = train_test_split(
        temp_cutouts, test_size=0.5, random_state=RANDOM_SEED
    )

    # Save cutouts to respective directories
    splits = {"train": train_cutouts, "val": val_cutouts, "test": test_cutouts}

    for split_name, cutouts in splits.items():
        print(f"Saving {len(cutouts)} {split_name} samples...")

        for cutout in cutouts:
            # Save image
            img_filename = f"{cutout['filename']}.jpg"
            img_path = os.path.join(output_dir, "images", split_name, img_filename)
            cv2.imwrite(img_path, cutout["image_crop"])

            # Save labels
            label_filename = f"{cutout['filename']}.txt"
            label_path = os.path.join(output_dir, "labels", split_name, label_filename)
            with open(label_path, "w") as f:
                f.write("\n".join(cutout["labels"]))

    # Create dataset configuration files
    create_dataset_yaml(output_dir)
    create_dataset_readme(
        output_dir,
        len(all_cutouts),
        len(train_cutouts),
        len(val_cutouts),
        len(test_cutouts),
        MIN_SEGMENT_DIST,
        MIN_CUTOUT_DIST,
        CUTOUT_SIZE,
        RANDOM_SEED,
    )

    print("Data preprocessing completed successfully!")
    print(f"Dataset saved to: {output_dir}")
    print(
        f"Train: {len(train_cutouts)}, Val: {len(val_cutouts)}, Test: {len(test_cutouts)}"
    )


if __name__ == "__main__":
    main()
