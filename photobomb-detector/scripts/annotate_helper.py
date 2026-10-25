"""Helper script to generate or update ground truth annotations for test images."""

from __future__ import annotations
import os
import json
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detector import detect_persons


def generate_baseline_ground_truth(data_dir: Path, output_file: Path) -> None:
    """Scan data directory and build baseline ground truth annotations."""
    images_list = []

    for category in ["photobomber", "ambiguous", "clean"]:
        cat_folder = data_dir / category
        if not cat_folder.exists():
            continue

        for img_path in sorted(cat_folder.glob("*.jpg")):
            rel_path = f"{category}/{img_path.name}"

            try:
                detections = detect_persons(str(img_path))
            except Exception:
                detections = []

            annotations = []
            if category == "clean":
                # In clean photos, every detected person is an intended subject ("keep")
                for d in detections:
                    annotations.append({"instance_id": d["instance_id"], "label": "keep"})
            elif category == "photobomber":
                # In photobomber photos, typically the smaller/peripheral persons are photobombers
                if detections:
                    # Largest person is keep
                    areas = [
                        (d["bounding_box"][2] - d["bounding_box"][0])
                        * (d["bounding_box"][3] - d["bounding_box"][1])
                        for d in detections
                    ]
                    max_idx = int(areas.index(max(areas)))
                    for idx, d in enumerate(detections):
                        lbl = "keep" if idx == max_idx else "remove"
                        annotations.append({"instance_id": d["instance_id"], "label": lbl})
            else:  # ambiguous
                # In ambiguous/crowded photos, foreground prominent persons are kept, edge/small are remove
                if detections:
                    areas = [
                        (d["bounding_box"][2] - d["bounding_box"][0])
                        * (d["bounding_box"][3] - d["bounding_box"][1])
                        for d in detections
                    ]
                    max_area = max(areas) if areas else 1
                    for idx, d in enumerate(detections):
                        ratio = areas[idx] / max_area
                        lbl = "keep" if ratio > 0.4 else "remove"
                        annotations.append({"instance_id": d["instance_id"], "label": lbl})

            images_list.append(
                {
                    "filename": rel_path,
                    "category": category,
                    "annotations": annotations,
                }
            )

    output_data = {"images": images_list}
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    print(f"Generated ground truth with {len(images_list)} images -> {output_file}")


if __name__ == "__main__":
    base = Path(__file__).resolve().parent.parent
    data_dir = base / "data" / "test_images"
    gt_path = base / "data" / "ground_truth.json"
    generate_baseline_ground_truth(data_dir, gt_path)
