#!/usr/bin/env python3
"""
YOLO label conversion utilities for converting masks to YOLO format annotations.
"""

import cv2
import numpy as np
from typing import List


def mask_to_yolo_labels(mask_crop: np.ndarray, cutout_size: int) -> List[str]:
    """
    Convert a cropped mask to YOLO format labels.

    Args:
        mask_crop: Cropped binary mask
        cutout_size: Size of the cutout (for normalization)

    Returns:
        List of YOLO format label strings
    """
    # Find connected components in the cropped mask
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask_crop, connectivity=8
    )

    yolo_labels = []

    # Skip background (label 0)
    for i in range(1, num_labels):
        # Get bounding box from stats
        x, y, w, h, area = stats[i]

        # Skip very small objects
        if area < 10:
            continue

        # Convert to YOLO format (normalized center coordinates and dimensions)
        x_center = (x + w / 2) / cutout_size
        y_center = (y + h / 2) / cutout_size
        width = w / cutout_size
        height = h / cutout_size

        # Class ID is 0 for 'poo' class
        class_id = 0

        # Format: class_id x_center y_center width height
        yolo_labels.append(
            f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        )

    return yolo_labels
