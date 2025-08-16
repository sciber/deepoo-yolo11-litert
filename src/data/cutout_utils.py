#!/usr/bin/env python3
"""
Cutout generation utilities for creating 640x640 image patches.
"""

import random
import numpy as np
from typing import List, Tuple


def find_valid_cutout(
    center: Tuple[int, int], img_height: int, img_width: int, cutout_size: int
) -> Tuple[int, int, int, int]:
    """
    Find a valid cutout square that contains the center and fits within image bounds.

    Args:
        center: (x, y) coordinates of the desired center
        img_height: Height of the source image
        img_width: Width of the source image
        cutout_size: Size of the square cutout

    Returns:
        (x1, y1, x2, y2) coordinates of the cutout rectangle
    """
    cx, cy = center
    half_size = cutout_size // 2

    # Calculate ideal bounds
    x1 = cx - half_size
    y1 = cy - half_size
    x2 = x1 + cutout_size
    y2 = y1 + cutout_size

    # Adjust to fit within image bounds
    if x1 < 0:
        x1 = 0
        x2 = cutout_size
    elif x2 > img_width:
        x2 = img_width
        x1 = img_width - cutout_size

    if y1 < 0:
        y1 = 0
        y2 = cutout_size
    elif y2 > img_height:
        y2 = img_height
        y1 = img_height - cutout_size

    return x1, y1, x2, y2


def filter_cutout_centers(
    centers: List[Tuple[int, int]],
    min_dist: int,
    img_height: int,
    img_width: int,
    cutout_size: int,
) -> List[Tuple[int, int]]:
    """
    Filter cutout centers that are too close, ensuring valid cutouts.

    Args:
        centers: List of center coordinates
        min_dist: Minimum distance between cutout centers
        img_height: Height of the source image
        img_width: Width of the source image
        cutout_size: Size of the square cutout

    Returns:
        List of filtered center coordinates
    """
    if len(centers) <= 1:
        return centers

    filtered_centers = []

    for center in centers:
        # Check if this center can form a valid cutout
        if (
            center[0] >= cutout_size // 2
            and center[0] <= img_width - cutout_size // 2
            and center[1] >= cutout_size // 2
            and center[1] <= img_height - cutout_size // 2
        ):
            # Check distance from existing centers
            too_close = False
            for existing_center in filtered_centers:
                dist = np.sqrt(
                    (center[0] - existing_center[0]) ** 2
                    + (center[1] - existing_center[1]) ** 2
                )
                if dist < min_dist:
                    too_close = True
                    break

            if not too_close:
                filtered_centers.append(center)

    return filtered_centers


def generate_random_cutout(
    img_height: int, img_width: int, cutout_size: int
) -> Tuple[int, int, int, int]:
    """
    Generate a random valid cutout within image bounds.

    Args:
        img_height: Height of the source image
        img_width: Width of the source image
        cutout_size: Size of the square cutout

    Returns:
        (x1, y1, x2, y2) coordinates of the cutout rectangle
    """
    max_x = img_width - cutout_size
    max_y = img_height - cutout_size

    if max_x <= 0 or max_y <= 0:
        # Image too small, return what we can
        return 0, 0, min(cutout_size, img_width), min(cutout_size, img_height)

    x1 = random.randint(0, max_x)
    y1 = random.randint(0, max_y)
    x2 = x1 + cutout_size
    y2 = y1 + cutout_size

    return x1, y1, x2, y2
