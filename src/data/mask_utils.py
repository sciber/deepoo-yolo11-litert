#!/usr/bin/env python3
"""
Mask processing utilities for semantic mask analysis and object center detection.
"""

import cv2
import numpy as np
from typing import List, Tuple


def find_object_centers(mask: np.ndarray) -> List[Tuple[int, int]]:
    """
    Find centers of all object segments in a binary mask.

    Args:
        mask: Binary mask where objects are white (255) and background is black (0)

    Returns:
        List of (x, y) coordinates representing object centers
    """
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

    centers = []
    # Skip background (label 0)
    for i in range(1, num_labels):
        # Get centroid coordinates
        cx, cy = centroids[i]
        centers.append((int(cx), int(cy)))

    return centers


def merge_close_centers(
    centers: List[Tuple[int, int]], min_dist: int
) -> List[Tuple[int, int]]:
    """
    Merge centers that are closer than min_dist by replacing pairs with their midpoint.

    Args:
        centers: List of (x, y) center coordinates
        min_dist: Minimum distance threshold

    Returns:
        List of merged center coordinates
    """
    if len(centers) <= 1:
        return centers

    merged_centers = centers.copy()

    while True:
        found_merge = False
        new_centers = []
        used_indices = set()

        for i, center1 in enumerate(merged_centers):
            if i in used_indices:
                continue

            merged = False
            for j, center2 in enumerate(merged_centers[i + 1 :], i + 1):
                if j in used_indices:
                    continue

                # Calculate distance
                dist = np.sqrt(
                    (center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2
                )

                if dist < min_dist:
                    # Merge centers to midpoint
                    midpoint = (
                        (center1[0] + center2[0]) // 2,
                        (center1[1] + center2[1]) // 2,
                    )
                    new_centers.append(midpoint)
                    used_indices.add(i)
                    used_indices.add(j)
                    found_merge = True
                    merged = True
                    break

            if not merged:
                new_centers.append(center1)
                used_indices.add(i)

        merged_centers = new_centers

        if not found_merge:
            break

    return merged_centers
