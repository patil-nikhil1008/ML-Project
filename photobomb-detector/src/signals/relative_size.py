"""Signal 2: Relative Size

Intended subjects occupy significant visual real estate in the composition.
Photobombers, especially in the background or lurking behind, typically present
a smaller apparent size / bounding box area compared to foreground subjects.
"""

from __future__ import annotations
from typing import List


def compute_relative_sizes(bboxes: List[List[int]]) -> List[float]:
    """Calculate relative size score for each bounding box against the largest person.

    Args:
        bboxes: List of bounding boxes, each [x1, y1, x2, y2].

    Returns:
        List[float]: Normalized size score in [0.0, 1.0] for each person.
                     Returns [1.0] if there is only one person.
    """
    if not bboxes:
        return []

    areas = []
    for b in bboxes:
        w = max(0, b[2] - b[0])
        h = max(0, b[3] - b[1])
        areas.append(float(w * h))

    max_area = max(areas)
    if max_area <= 0:
        return [0.0] * len(bboxes)

    # Relative area ratio: fraction of largest detected person's area
    scores = [round(a / max_area, 4) for a in areas]
    return scores


if __name__ == "__main__":
    boxes = [
        [100, 100, 400, 700],  # Area: 300 * 600 = 180,000
        [600, 200, 700, 350],  # Area: 100 * 150 = 15,000
    ]
    print("Relative sizes:", compute_relative_sizes(boxes))
