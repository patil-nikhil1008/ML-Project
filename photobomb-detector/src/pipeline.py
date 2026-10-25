"""Main pipeline orchestrator for Photobomb Detection and Subject Classification."""

from __future__ import annotations
import os
import json
from typing import Union, Dict, Any, List, Optional
import cv2
import numpy as np

from .detector import detect_persons
from .signals.centrality import compute_centrality
from .signals.relative_size import compute_relative_sizes
from .signals.focus_blur import compute_sharpness, normalize_sharpness
from .signals.depth import estimate_depths
from .signals.pose import compute_pose_score
from .classifier import classify_persons, DEFAULT_SIGNAL_WEIGHTS


class PhotobombPipeline:
    """End-to-end photobomb detection and subject classification pipeline."""

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        threshold: float = 0.62,
        conf_threshold: float = 0.25,
    ) -> None:
        self.weights = weights or DEFAULT_SIGNAL_WEIGHTS
        self.threshold = threshold
        self.conf_threshold = conf_threshold

    def process_image(
        self,
        image_input: Union[str, np.ndarray],
    ) -> Dict[str, Any]:
        """Run full detection, signal extraction, and classification on an image.

        Args:
            image_input: Path to image file or BGR numpy array.

        Returns:
            Dict containing metadata and structured person classification results.
        """
        image_path = image_input if isinstance(image_input, str) else "in_memory_image.jpg"
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"File not found: {image_input}")
            img = cv2.imread(image_input)
            if img is None:
                raise ValueError(f"Could not load image: {image_input}")
        else:
            img = image_input

        h, w = img.shape[:2]

        # Step 1: Detect persons
        detections = detect_persons(img, conf_threshold=self.conf_threshold)

        if not detections:
            return {
                "image_path": image_path,
                "image_dimensions": {"width": w, "height": h},
                "num_persons_detected": 0,
                "persons": [],
            }

        bboxes = [d["bounding_box"] for d in detections]

        # Step 2: Extract spatial and visual signals
        # Signal 1: Centrality
        centrality_scores = [compute_centrality(b, w, h) for b in bboxes]

        # Signal 2: Relative Size
        size_scores = compute_relative_sizes(bboxes)

        # Signal 3: Focus & Blur
        raw_sharpness = [compute_sharpness(img, b) for b in bboxes]
        focus_scores = normalize_sharpness(raw_sharpness)

        # Signal 4: Depth
        depth_scores = estimate_depths(img, bboxes)

        # Signal 5: Pose
        pose_scores = [compute_pose_score(img, b) for b in bboxes]

        signal_scores = {
            "centrality": centrality_scores,
            "relative_size": size_scores,
            "focus": focus_scores,
            "depth": depth_scores,
            "pose": pose_scores,
        }

        # Step 3: Classify persons
        classified_persons = classify_persons(
            detections=detections,
            signal_scores=signal_scores,
            weights=self.weights,
            threshold=self.threshold,
        )

        return {
            "image_path": image_path,
            "image_dimensions": {"width": w, "height": h},
            "num_persons_detected": len(classified_persons),
            "persons": classified_persons,
        }

    def visualize(
        self,
        image_input: Union[str, np.ndarray],
        pipeline_output: Dict[str, Any],
        output_path: Optional[str] = None,
    ) -> np.ndarray:
        """Render labeled bounding boxes and signal breakdown onto the image.

        Args:
            image_input: Image path or BGR numpy array.
            pipeline_output: Output dictionary from `process_image`.
            output_path: Optional file path to save the annotated image.

        Returns:
            np.ndarray: Annotated BGR image.
        """
        if isinstance(image_input, str):
            annotated = cv2.imread(image_input)
        else:
            annotated = image_input.copy()

        for p in pipeline_output.get("persons", []):
            x1, y1, x2, y2 = p["bounding_box"]
            cls = p["classification"]
            conf = p["confidence_score"]
            inst_id = p["instance_id"]

            # Green for KEEP (intended subject), Red for REMOVE (photobomber)
            color = (0, 200, 0) if cls == "keep" else (0, 0, 230)
            thickness = 3

            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

            # Label banner
            label = f"ID {inst_id}: {cls.upper()} ({conf*100:.1f}%)"
            (text_w, text_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )

            # Background rectangle for text
            banner_y1 = max(0, y1 - text_h - 10)
            banner_y2 = y1
            cv2.rectangle(
                annotated,
                (x1, banner_y1),
                (x1 + text_w + 10, banner_y2),
                color,
                -1,
            )
            cv2.putText(
                annotated,
                label,
                (x1 + 5, banner_y2 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        if output_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            cv2.imwrite(output_path, annotated)

        return annotated


def run_pipeline(
    image_path: str,
    output_dir: Optional[str] = None,
    weights: Optional[Dict[str, float]] = None,
    threshold: float = 0.62,
) -> Dict[str, Any]:
    """Single-function convenience entrypoint for the photobomb pipeline."""
    pipeline = PhotobombPipeline(weights=weights, threshold=threshold)
    result = pipeline.process_image(image_path)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(image_path))[0]

        # Save JSON output
        json_path = os.path.join(output_dir, f"{base_name}_result.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        # Save annotated image
        vis_path = os.path.join(output_dir, f"{base_name}_annotated.jpg")
        pipeline.visualize(image_path, result, output_path=vis_path)

    return result
