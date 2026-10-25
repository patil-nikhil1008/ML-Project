"""Person detection module using YOLOv8."""

from __future__ import annotations
import os
from typing import Union, List, Dict, Any
import cv2
import numpy as np


class PersonDetector:
    """Wrapper around YOLOv8 model for detecting persons in images."""

    def __init__(self, model_name: str = "yolov8n.pt", device: str = "") -> None:
        """Initialize the YOLOv8 detector.

        Args:
            model_name: Path or name of YOLOv8 model weights (default: 'yolov8n.pt').
            device: Target device, e.g., 'cpu', 'cuda', 'mps', or empty for auto-detect.
        """
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "ultralytics is required for PersonDetector. Install it with `pip install ultralytics`."
            ) from exc

        self.model = YOLO(model_name)
        self.device = device

    def detect_persons(
        self,
        image_input: Union[str, np.ndarray],
        conf_threshold: float = 0.25,
    ) -> List[Dict[str, Any]]:
        """Detect all persons in an image and assign sequential instance IDs.

        Args:
            image_input: File path to image or pre-loaded BGR numpy array.
            conf_threshold: Minimum detection confidence score.

        Returns:
            List of dictionaries containing:
                - instance_id (int): Unique person ID (0, 1, 2, ...)
                - bounding_box (List[int]): [x1, y1, x2, y2] pixel coordinates
                - detection_confidence (float): YOLO detection confidence score
        """
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Image file not found: {image_input}")
            img = cv2.imread(image_input)
            if img is None:
                raise ValueError(f"Could not decode image at path: {image_input}")
        else:
            img = image_input

        # Run inference. Class 0 corresponds to 'person' in the COCO dataset
        predict_args: Dict[str, Any] = {
            "source": img,
            "classes": [0],
            "conf": conf_threshold,
            "verbose": False,
        }
        if self.device:
            predict_args["device"] = self.device

        results = self.model.predict(**predict_args)
        detections: List[Dict[str, Any]] = []

        if not results:
            return detections

        first_res = results[0]
        boxes = first_res.boxes
        if boxes is None or len(boxes) == 0:
            return detections

        h, w = img.shape[:2]

        for idx, box in enumerate(boxes):
            coords = box.xyxy[0].cpu().numpy().tolist()
            x1 = max(0, min(int(round(coords[0])), w - 1))
            y1 = max(0, min(int(round(coords[1])), h - 1))
            x2 = max(x1 + 1, min(int(round(coords[2])), w))
            y2 = max(y1 + 1, min(int(round(coords[3])), h))
            conf = float(box.conf[0].cpu().numpy())

            detections.append(
                {
                    "instance_id": idx,
                    "bounding_box": [x1, y1, x2, y2],
                    "detection_confidence": round(conf, 4),
                }
            )

        return detections


_default_detector: Union[PersonDetector, None] = None


def detect_persons(
    image_input: Union[str, np.ndarray],
    conf_threshold: float = 0.25,
    model_name: str = "yolov8n.pt",
) -> List[Dict[str, Any]]:
    """Convenience helper using a cached default PersonDetector instance."""
    global _default_detector
    if _default_detector is None:
        _default_detector = PersonDetector(model_name=model_name)
    return _default_detector.detect_persons(image_input, conf_threshold=conf_threshold)


if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage: python -m src.detector <image_path>")
        sys.exit(1)

    target_image = sys.argv[1]
    res = detect_persons(target_image)
    print(f"Found {len(res)} person(s):")
    print(json.dumps(res, indent=2))
