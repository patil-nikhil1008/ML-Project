"""Signal 5: Pose Confidence & Orientation

Intended photo subjects typically pose towards the camera with high landmark
confidence, clear upper-body visibility, and frontal torso orientation.
Photobombers frequently exhibit occluded keypoints, odd contortions, or are
facing away from the primary camera axis.
"""

from __future__ import annotations
from typing import List, Optional
import cv2
import numpy as np


class PoseAnalyzer:
    """MediaPipe Pose analyzer with heuristic fallback."""

    def __init__(self) -> None:
        self.pose = None
        self._init_failed = False

    def _load_model(self) -> None:
        if self.pose is not None or self._init_failed:
            return

        try:
            import mediapipe as mp

            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                static_image_mode=True,
                model_complexity=1,
                enable_segmentation=False,
                min_detection_confidence=0.3,
            )
        except Exception:
            self._init_failed = True

    def analyze_crop(self, crop_bgr: np.ndarray) -> float:
        """Analyze a cropped person image and return a pose score in [0.0, 1.0]."""
        if crop_bgr is None or crop_bgr.size == 0:
            return 0.1

        h, w = crop_bgr.shape[:2]
        if h < 20 or w < 20:
            return 0.2

        self._load_model()
        if self.pose is not None:
            try:
                crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                results = self.pose.process(crop_rgb)

                if not results.pose_landmarks:
                    # No recognizable human pose detected
                    return 0.25

                landmarks = results.pose_landmarks.landmark

                # Key landmarks of interest:
                # 0: nose, 11: left shoulder, 12: right shoulder,
                # 23: left hip, 24: right hip
                key_indices = [0, 11, 12, 23, 24]
                visibilities = [landmarks[i].visibility for i in key_indices]
                avg_visibility = float(np.mean(visibilities))

                # Torso symmetry / front-facing check:
                # If both shoulders are visible, check width between them
                left_s = landmarks[11]
                right_s = landmarks[12]
                shoulder_dx = abs(left_s.x - right_s.x)

                # A person facing the camera has wide shoulder separation relative to bounding box
                front_facing_bonus = 0.2 if shoulder_dx > 0.15 else 0.0

                score = 0.5 * avg_visibility + front_facing_bonus + 0.3
                return round(float(np.clip(score, 0.0, 1.0)), 4)
            except Exception:
                pass

        # Fallback: Aspect ratio & edge distribution heuristic
        # People facing the camera in portraits tend to have an aspect ratio of 1.5 to 3.5
        aspect = float(h) / max(1.0, float(w))
        if 1.2 <= aspect <= 4.0:
            return 0.7
        return 0.4


_default_pose_analyzer: Optional[PoseAnalyzer] = None


def compute_pose_score(image: np.ndarray, bbox: List[int]) -> float:
    """Compute pose confidence and frontal orientation score for a single person.

    Args:
        image: Full image BGR numpy array.
        bbox: Bounding box [x1, y1, x2, y2].

    Returns:
        float: Pose score in [0.0, 1.0].
    """
    global _default_pose_analyzer
    if _default_pose_analyzer is None:
        _default_pose_analyzer = PoseAnalyzer()

    h, w = image.shape[:2]
    x1 = max(0, min(bbox[0], w - 1))
    y1 = max(0, min(bbox[1], h - 1))
    x2 = max(x1 + 1, min(bbox[2], w))
    y2 = max(y1 + 1, min(bbox[3], h))

    crop = image[y1:y2, x1:x2]
    return _default_pose_analyzer.analyze_crop(crop)


if __name__ == "__main__":
    dummy_img = np.zeros((500, 300, 3), dtype=np.uint8)
    print("Pose score:", compute_pose_score(dummy_img, [50, 50, 250, 450]))
