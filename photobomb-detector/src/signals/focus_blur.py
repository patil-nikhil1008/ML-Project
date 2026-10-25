"""Signal 3: Focus & Blur (Sharpness)

Lenses naturally create a focal plane where intended subjects are sharpest.
Background bystanders, photobombers, or moving intruders often suffer from
defocus blur or motion blur. We estimate sharpness via the variance of the
Laplacian operator on grayscale person crops.
"""

from __future__ import annotations
from typing import List
import cv2
import numpy as np


def compute_sharpness(image: np.ndarray, bbox: List[int]) -> float:
    """Compute the raw focus/sharpness score of a person crop via Laplacian variance.

    Args:
        image: Full image as BGR numpy array.
        bbox: Bounding box [x1, y1, x2, y2].

    Returns:
        float: Raw variance of the Laplacian.
    """
    if image is None or image.size == 0:
        return 0.0

    h, w = image.shape[:2]
    x1 = max(0, min(bbox[0], w - 1))
    y1 = max(0, min(bbox[1], h - 1))
    x2 = max(x1 + 1, min(bbox[2], w))
    y2 = max(y1 + 1, min(bbox[3], h))

    crop = image[y1:y2, x1:x2]
    if crop.size == 0 or crop.shape[0] < 2 or crop.shape[1] < 2:
        return 0.0

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return laplacian_var


def normalize_sharpness(sharpness_values: List[float]) -> List[float]:
    """Normalize raw sharpness values across all detected persons in an image.

    Args:
        sharpness_values: List of raw Laplacian variance values.

    Returns:
        List[float]: Normalized scores in [0.0, 1.0].
    """
    if not sharpness_values:
        return []

    max_val = max(sharpness_values)
    if max_val <= 1e-6:
        # If all crops have negligible sharpness, treat them equally
        return [1.0] * len(sharpness_values)

    # Normalize relative to the sharpest person in the frame
    # We apply a soft lower bound so slightly softer subjects aren't zeroed out
    return [round(max(0.0, min(val / max_val, 1.0)), 4) for val in sharpness_values]


if __name__ == "__main__":
    dummy_img = np.random.randint(0, 255, (400, 400, 3), dtype=np.uint8)
    s = compute_sharpness(dummy_img, [50, 50, 200, 200])
    print("Sharpness:", s)
    print("Normalized:", normalize_sharpness([s, s * 0.2]))
