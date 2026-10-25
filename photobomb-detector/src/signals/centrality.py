"""Signal 1: Spatial Centrality

Intended subjects are typically positioned closer to the center of the frame.
Photobombers often appear near the peripheral boundaries or corners.
"""

from __future__ import annotations
import math
from typing import List, Union


def compute_centrality(
    bbox: List[int],
    image_width: int,
    image_height: int,
) -> float:
    """Calculate the spatial centrality score for a bounding box.

    Args:
        bbox: Bounding box coordinates [x1, y1, x2, y2].
        image_width: Total image width in pixels.
        image_height: Total image height in pixels.

    Returns:
        float: Normalized score in [0.0, 1.0], where 1.0 is dead-center.
    """
    if image_width <= 0 or image_height <= 0:
        return 0.5

    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    img_cx = image_width / 2.0
    img_cy = image_height / 2.0

    # Euclidean distance from bbox center to image center
    dist = math.sqrt((cx - img_cx) ** 2 + (cy - img_cy) ** 2)

    # Maximum possible distance is from center to any image corner
    max_dist = math.sqrt(img_cx**2 + img_cy**2)
    if max_dist == 0:
        return 1.0

    norm_dist = min(dist / max_dist, 1.0)
    centrality_score = 1.0 - norm_dist
    return float(round(centrality_score, 4))


if __name__ == "__main__":
    # Quick sanity test
    w, h = 1000, 800
    # Center box
    print("Center score:", compute_centrality([450, 350, 550, 450], w, h))
    # Corner box
    print("Corner score:", compute_centrality([0, 0, 50, 50], w, h))
