#!/usr/bin/env python3
"""CLI entrypoint for running the Photobomb Detection & Classification Pipeline."""

from __future__ import annotations
import os
import sys
import argparse
import json
from pathlib import Path
from typing import List

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False

from src.pipeline import PhotobombPipeline


def format_summary_table(pipeline_output: dict) -> str:
    """Format classification results into an ASCII table."""
    persons = pipeline_output.get("persons", [])
    if not persons:
        return "No persons detected in the image."

    headers = [
        "ID",
        "BBox [x1,y1,x2,y2]",
        "Class",
        "Confidence",
        "Centrality",
        "Rel Size",
        "Focus",
        "Depth",
        "Pose",
    ]
    rows = []
    for p in persons:
        sigs = p.get("signals_used", {})
        cls_str = p["classification"].upper()
        if cls_str == "REMOVE":
            cls_str = f"\033[91m{cls_str}\033[0m"  # Red
        else:
            cls_str = f"\033[92m{cls_str}\033[0m"  # Green

        rows.append([
            p["instance_id"],
            str(p["bounding_box"]),
            cls_str,
            f"{p['confidence_score']:.2f}",
            f"{sigs.get('centrality', 0.0):.2f}",
            f"{sigs.get('relative_size', 0.0):.2f}",
            f"{sigs.get('focus', 0.0):.2f}",
            f"{sigs.get('depth', 0.0):.2f}",
            f"{sigs.get('pose', 0.0):.2f}",
        ])

    if HAS_TABULATE:
        return tabulate(rows, headers=headers, tablefmt="fancy_grid")
    else:
        header_line = " | ".join(headers)
        sep_line = "-" * len(header_line)
        body = "\n".join([" | ".join(map(str, r)) for r in rows])
        return f"{header_line}\n{sep_line}\n{body}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Photobomb Detection & Subject Classification Pipeline"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--input", "-i", type=str, help="Path to a single image file"
    )
    group.add_argument(
        "--input-dir", "-d", type=str, help="Path to directory of images to process"
    )
    parser.add_argument(
        "--output", "-o", type=str, default="output", help="Directory to save JSON & visualized images"
    )
    parser.add_argument(
        "--threshold", "-t", type=float, default=0.62, help="Classification decision threshold (default: 0.62)"
    )
    parser.add_argument(
        "--conf", "-c", type=float, default=0.25, help="YOLO person detection confidence threshold (default: 0.25)"
    )

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    pipeline = PhotobombPipeline(threshold=args.threshold, conf_threshold=args.conf)

    # Collect images
    image_paths: List[Path] = []
    if args.input:
        image_paths = [Path(args.input)]
    else:
        in_dir = Path(args.input_dir)
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
            image_paths.extend(in_dir.rglob(ext))

    if not image_paths:
        print(f"No images found to process.")
        sys.exit(1)

    print(f"\n=======================================================")
    print(f" Processing {len(image_paths)} image(s)...")
    print(f" Output folder: {os.path.abspath(args.output)}")
    print(f"=======================================================\n")

    for idx, img_path in enumerate(image_paths, 1):
        print(f"[{idx}/{len(image_paths)}] Processing: {img_path.name}")
        try:
            result = pipeline.process_image(str(img_path))

            # Save JSON
            base_name = img_path.stem
            json_file = Path(args.output) / f"{base_name}_result.json"
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)

            # Save annotated image
            vis_file = Path(args.output) / f"{base_name}_annotated.jpg"
            pipeline.visualize(str(img_path), result, output_path=str(vis_file))

            # Display table
            print(format_summary_table(result))
            print(f"  -> Saved JSON:      {json_file.name}")
            print(f"  -> Saved Annotated: {vis_file.name}\n")
        except Exception as e:
            print(f"  \033[91mError processing {img_path.name}: {e}\033[0m\n")

    print("Pipeline execution complete!\n")


if __name__ == "__main__":
    main()
