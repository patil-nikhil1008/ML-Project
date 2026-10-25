"""Evaluation module to compute Precision, Recall, and F1 for the 'remove' class."""

from __future__ import annotations
import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False


def compute_metrics(tp: int, fp: int, fn: int, tn: int) -> Dict[str, float]:
    """Compute precision, recall, F1, and accuracy from confusion matrix counts."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        (2.0 * precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total if total > 0 else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(accuracy, 4),
    }


def evaluate(
    predictions_dir: str,
    ground_truth_path: str,
) -> Dict[str, Any]:
    """Evaluate pipeline predictions against ground truth annotations.

    Args:
        predictions_dir: Directory containing `*_result.json` files.
        ground_truth_path: Path to ground_truth.json.

    Returns:
        Dict with overall and per-category evaluation metrics.
    """
    if not os.path.exists(ground_truth_path):
        raise FileNotFoundError(f"Ground truth file not found: {ground_truth_path}")

    with open(ground_truth_path, "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    # Collect prediction files
    pred_dir = Path(predictions_dir)
    pred_files = list(pred_dir.glob("*_result.json"))
    pred_map: Dict[str, Dict[str, Any]] = {}
    for pf in pred_files:
        with open(pf, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Map by image basename (without suffix)
            img_name = Path(data.get("image_path", pf.stem.replace("_result", ""))).stem
            pred_map[img_name] = data
            # Also map by file stem
            pred_map[pf.stem.replace("_result", "")] = data

    categories = ["photobomber", "ambiguous", "clean"]
    category_counts = {c: {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "images": 0} for c in categories}
    overall_counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "images": 0}

    detailed_results = []

    for item in gt_data.get("images", []):
        filename = item["filename"]
        base_name = Path(filename).stem
        cat = item.get("category", "photobomber").lower()
        if cat not in category_counts:
            category_counts[cat] = {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "images": 0}

        pred_entry = pred_map.get(base_name)
        if pred_entry is None:
            continue

        category_counts[cat]["images"] += 1
        overall_counts["images"] += 1

        # Build map of predictions by instance_id
        pred_persons = {p["instance_id"]: p["classification"] for p in pred_entry.get("persons", [])}

        for ann in item.get("annotations", []):
            inst_id = ann["instance_id"]
            true_label = ann["label"].lower()  # 'keep' or 'remove'
            pred_label = pred_persons.get(inst_id, "keep").lower()

            # Evaluating for the target positive class: "remove" (photobomber)
            is_tp = (pred_label == "remove" and true_label == "remove")
            is_fp = (pred_label == "remove" and true_label == "keep")
            is_fn = (pred_label == "keep" and true_label == "remove")
            is_tn = (pred_label == "keep" and true_label == "keep")

            if is_tp:
                category_counts[cat]["tp"] += 1
                overall_counts["tp"] += 1
            elif is_fp:
                category_counts[cat]["fp"] += 1
                overall_counts["fp"] += 1
            elif is_fn:
                category_counts[cat]["fn"] += 1
                overall_counts["fn"] += 1
            elif is_tn:
                category_counts[cat]["tn"] += 1
                overall_counts["tn"] += 1

            detailed_results.append({
                "image": filename,
                "category": cat,
                "instance_id": inst_id,
                "ground_truth": true_label,
                "prediction": pred_label,
                "correct": true_label == pred_label,
            })

    # Compute metrics for overall and each category
    per_cat_metrics = {}
    for cat, counts in category_counts.items():
        if counts["images"] > 0:
            m = compute_metrics(counts["tp"], counts["fp"], counts["fn"], counts["tn"])
            m["images"] = counts["images"]
            per_cat_metrics[cat] = m

    overall_metrics = compute_metrics(
        overall_counts["tp"], overall_counts["fp"], overall_counts["fn"], overall_counts["tn"]
    )
    overall_metrics["images"] = overall_counts["images"]

    return {
        "overall": overall_metrics,
        "by_category": per_cat_metrics,
        "detailed": detailed_results,
    }


def print_evaluation_report(results: Dict[str, Any]) -> None:
    """Print a clean metrics table and confusion matrix to the terminal."""
    print("\n" + "=" * 75)
    print("        PHOTOBOMB EVALUATION REPORT — CLASS: 'REMOVE' (PHOTOBOMBER)       ")
    print("=" * 75)

    headers = [
        "Category",
        "Images",
        "TP",
        "FP",
        "FN",
        "TN",
        "Precision",
        "Recall",
        "F1-Score",
        "Accuracy",
    ]
    rows = []

    for cat, m in results.get("by_category", {}).items():
        rows.append([
            cat.capitalize(),
            m["images"],
            m["tp"],
            m["fp"],
            m["fn"],
            m["tn"],
            f"{m['precision']:.4f}",
            f"{m['recall']:.4f}",
            f"{m['f1']:.4f}",
            f"{m['accuracy']:.4f}",
        ])

    ov = results.get("overall", {})
    rows.append([
        "OVERALL",
        ov["images"],
        ov["tp"],
        ov["fp"],
        ov["fn"],
        ov["tn"],
        f"{ov['precision']:.4f}",
        f"{ov['recall']:.4f}",
        f"{ov['f1']:.4f}",
        f"{ov['accuracy']:.4f}",
    ])

    if HAS_TABULATE:
        print(tabulate(rows, headers=headers, tablefmt="fancy_grid"))
    else:
        print(" | ".join(headers))
        print("-" * 75)
        for r in rows:
            print(" | ".join(map(str, r)))

    print("\nConfusion Matrix Summary (Overall):")
    print(f"  True Positives  (Photobomber correctly flagged): {ov['tp']}")
    print(f"  False Positives (Subject falsely flagged):        {ov['fp']}")
    print(f"  False Negatives (Photobomber missed):             {ov['fn']}")
    print(f"  True Negatives  (Subject correctly kept):         {ov['tn']}")
    print("=" * 75 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Photobomb Detection Predictions against Ground Truth"
    )
    parser.add_argument(
        "--predictions",
        "-p",
        type=str,
        default="output",
        help="Path to directory containing prediction JSON files",
    )
    parser.add_argument(
        "--ground-truth",
        "-g",
        type=str,
        default="data/ground_truth.json",
        help="Path to data/ground_truth.json",
    )
    args = parser.parse_args()

    results = evaluate(args.predictions, args.ground_truth)
    print_evaluation_report(results)


if __name__ == "__main__":
    main()
