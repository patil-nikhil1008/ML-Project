"""Signal 4: Monocular Depth Estimation

Intended subjects are typically positioned in the foreground plane or primary
depth layer. Photobombers often appear deep in the background.
We estimate dense depth/disparity using MiDaS (or a robust geometric fallback).
In MiDaS disparity maps, higher values correspond to objects closer to the camera.
"""

from __future__ import annotations
import os
from typing import List, Optional, Union
import cv2
import numpy as np


class DepthEstimator:
    """Monocular depth estimator with torch.hub MiDaS and geometric fallback."""

    def __init__(self, model_type: str = "MiDaS_small", device: str = "") -> None:
        self.model_type = model_type
        self.device = device
        self.model = None
        self.transforms = None
        self._init_failed = False

    def _load_model(self) -> None:
        if self.model is not None or self._init_failed:
            return

        try:
            import torch

            if not self.device:
                if torch.cuda.is_available():
                    self.device = "cuda"
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    self.device = "mps"
                else:
                    self.device = "cpu"

            # Load MiDaS from torch hub
            self.model = torch.hub.load("intel-isl/MiDaS", self.model_type, trust_repo=True)
            self.model.to(self.device)
            self.model.eval()

            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
            if self.model_type in ("DPT_Large", "DPT_Hybrid"):
                self.transforms = midas_transforms.dpt_transform
            else:
                self.transforms = midas_transforms.small_transform
        except Exception as exc:
            # Mark failed so we don't continually retry loading
            self._init_failed = True
            # We will use geometric vertical ground-contact & area proxy as a graceful fallback

    def predict_depth_map(self, image: np.ndarray) -> np.ndarray:
        """Generate relative depth/disparity map for an image.

        Higher values indicate regions closer to the camera.
        """
        self._load_model()
        if self.model is not None and self.transforms is not None:
            try:
                import torch

                img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                input_batch = self.transforms(img_rgb).to(self.device)

                with torch.no_grad():
                    prediction = self.model(input_batch)
                    prediction = torch.nn.functional.interpolate(
                        prediction.unsqueeze(1),
                        size=image.shape[:2],
                        mode="bicubic",
                        align_corners=False,
                    ).squeeze()

                depth_map = prediction.cpu().numpy()
                # Normalize depth map to [0, 1]
                d_min, d_max = depth_map.min(), depth_map.max()
                if d_max > d_min:
                    depth_map = (depth_map - d_min) / (d_max - d_min)
                else:
                    depth_map = np.ones_like(depth_map)
                return depth_map
            except Exception:
                pass

        # Fallback: Perspective ground-plane geometry heuristic
        # In typical eye-level photos, lower rows (higher Y) and larger scale are closer
        h, w = image.shape[:2]
        y_coords = np.linspace(0.1, 1.0, h, dtype=np.float32).reshape(-1, 1)
        fallback_map = np.repeat(y_coords, w, axis=1)
        return fallback_map


_default_depth_estimator: Optional[DepthEstimator] = None


def estimate_depths(
    image: np.ndarray,
    bboxes: List[List[int]],
) -> List[float]:
    """Estimate depth-based subject likelihood score for each person bbox.

    Higher scores (closer to 1.0) mean the person is in the primary foreground plane.
    Lower scores mean the person is far in the background.

    Args:
        image: BGR numpy image.
        bboxes: List of bboxes [x1, y1, x2, y2].

    Returns:
        List[float]: Normalized scores in [0.0, 1.0].
    """
    if not bboxes:
        return []

    if len(bboxes) == 1:
        return [1.0]

    global _default_depth_estimator
    if _default_depth_estimator is None:
        _default_depth_estimator = DepthEstimator()

    depth_map = _default_depth_estimator.predict_depth_map(image)
    h, w = image.shape[:2]

    median_depths = []
    for b in bboxes:
        x1 = max(0, min(b[0], w - 1))
        y1 = max(0, min(b[1], h - 1))
        x2 = max(x1 + 1, min(b[2], w))
        y2 = max(y1 + 1, min(b[3], h))

        # Sample central 60% of bbox to avoid background border leakage
        bx1 = int(x1 + 0.2 * (x2 - x1))
        bx2 = int(x2 - 0.2 * (x2 - x1))
        by1 = int(y1 + 0.2 * (y2 - y1))
        by2 = int(y2 - 0.2 * (y2 - y1))
        if bx2 <= bx1:
            bx1, bx2 = x1, x2
        if by2 <= by1:
            by1, by2 = y1, y2

        roi_depth = depth_map[by1:by2, bx1:bx2]
        if roi_depth.size > 0:
            median_depths.append(float(np.median(roi_depth)))
        else:
            median_depths.append(0.5)

    max_d = max(median_depths)
    min_d = min(median_depths)

    if max_d - min_d < 1e-4:
        return [1.0] * len(bboxes)

    # Relative foreground position: closer to foreground = higher score
    scores = [round(float((d - min_d) / (max_d - min_d) * 0.7 + 0.3), 4) for d in median_depths]
    return scores


if __name__ == "__main__":
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    boxes = [[200, 100, 350, 450], [50, 150, 100, 250]]
    print("Estimated depth scores:", estimate_depths(img, boxes))
