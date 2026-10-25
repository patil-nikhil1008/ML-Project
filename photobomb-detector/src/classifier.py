"""Subject vs Photobomber Classifier.

Fuses multiple spatial, geometric, depth, and pose signals into a composite
score and outputs a structured classification ('keep' vs 'remove') with
confidence metrics.
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional

DEFAULT_SIGNAL_WEIGHTS: Dict[str, float] = {
    "centrality": 0.25,
    "relative_size": 0.25,
    "focus": 0.20,
    "depth": 0.15,
    "pose": 0.15,
}

DEFAULT_DECISION_THRESHOLD = 0.62


def classify_persons(
    detections: List[Dict[str, Any]],
    signal_scores: Dict[str, List[float]],
    weights: Optional[Dict[str, float]] = None,
    threshold: float = DEFAULT_DECISION_THRESHOLD,
) -> List[Dict[str, Any]]:
    """Classify detected persons as 'keep' (intended subject) or 'remove' (photobomber).

    Args:
        detections: List of detection dicts (each having 'instance_id' and 'bounding_box').
        signal_scores: Dictionary mapping signal names ('centrality', 'relative_size', etc.)
                       to lists of float scores (one per detection).
        weights: Dictionary of signal weights. Defaults to DEFAULT_SIGNAL_WEIGHTS.
        threshold: Classification decision boundary. Scores >= threshold are marked 'keep'.

    Returns:
        List of structured result dictionaries with:
            - instance_id (int)
            - bounding_box (List[int])
            - classification ('keep' | 'remove')
            - confidence_score (float in [0.0, 1.0])
            - signals_used (Dict[str, float])
    """
    if not detections:
        return []

    if weights is None:
        weights = DEFAULT_SIGNAL_WEIGHTS.copy()

    # Normalize weights so they sum to 1.0
    total_w = sum(weights.values())
    norm_weights = {k: v / total_w for k, v in weights.items()}

    num_persons = len(detections)
    results: List[Dict[str, Any]] = []

    # Edge case: If only 1 person in the entire frame, they are the subject
    if num_persons == 1:
        d = detections[0]
        person_signals = {sig: signal_scores.get(sig, [1.0])[0] for sig in norm_weights}
        results.append(
            {
                "instance_id": d["instance_id"],
                "bounding_box": d["bounding_box"],
                "classification": "keep",
                "confidence_score": 0.98,
                "signals_used": person_signals,
            }
        )
        return results

    # Determine highest composite score among detections to enforce relative ranking
    composite_scores: List[float] = []
    per_person_signals: List[Dict[str, float]] = []

    for i in range(num_persons):
        signals_dict: Dict[str, float] = {}
        weighted_sum = 0.0

        for sig, w in norm_weights.items():
            sig_list = signal_scores.get(sig, [])
            val = sig_list[i] if i < len(sig_list) else 0.5
            signals_dict[sig] = round(float(val), 4)
            weighted_sum += w * val

        # Group Cohesion Analysis (Assignment Requirement #2):
        # In multi-person group portraits (>= 2), persons with substantial scale (size >= 0.45)
        # and upright frontal orientation (pose >= 0.55) are legitimate posing members of the cohort.
        # They must not be penalized simply for standing on the periphery of a wide group.
        if num_persons >= 2:
            rel_size = signals_dict.get("relative_size", 0.0)
            pose_val = signals_dict.get("pose", 0.0)
            if rel_size >= 0.45 and pose_val >= 0.55:
                weighted_sum = min(1.0, weighted_sum + 0.09)

        composite_scores.append(weighted_sum)
        per_person_signals.append(signals_dict)

    # In any photo, at least the top-scoring person must be considered a 'keep' subject
    max_score = max(composite_scores)
    adaptive_threshold = min(threshold, max_score - 1e-4) if max_score < threshold else threshold

    for i, d in enumerate(detections):
        comp = composite_scores[i]
        signals_dict = per_person_signals[i]

        is_keep = comp >= adaptive_threshold
        label = "keep" if is_keep else "remove"

        # Calculate decision confidence: distance from boundary mapped to [0.5, 0.99]
        margin = abs(comp - adaptive_threshold)
        # Normalize margin
        max_possible_margin = max(adaptive_threshold, 1.0 - adaptive_threshold)
        raw_conf = 0.5 + 0.49 * (margin / max(max_possible_margin, 1e-3))
        confidence = round(float(min(0.99, max(0.50, raw_conf))), 4)

        results.append(
            {
                "instance_id": d["instance_id"],
                "bounding_box": d["bounding_box"],
                "classification": label,
                "confidence_score": confidence,
                "signals_used": signals_dict,
            }
        )

    return results


if __name__ == "__main__":
    dets = [
        {"instance_id": 0, "bounding_box": [200, 100, 500, 700]},
        {"instance_id": 1, "bounding_box": [50, 50, 120, 180]},
    ]
    sigs = {
        "centrality": [0.85, 0.15],
        "relative_size": [1.0, 0.08],
        "focus": [0.95, 0.20],
        "depth": [0.90, 0.35],
        "pose": [0.88, 0.30],
    }
    res = classify_persons(dets, sigs)
    import json

    print(json.dumps(res, indent=2))
