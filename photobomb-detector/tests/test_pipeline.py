"""Unit tests for the Photobomb Detection & Subject Classification Pipeline."""

import pytest
import numpy as np

from src.signals.centrality import compute_centrality
from src.signals.relative_size import compute_relative_sizes
from src.signals.focus_blur import compute_sharpness, normalize_sharpness
from src.signals.depth import estimate_depths
from src.signals.pose import compute_pose_score
from src.classifier import classify_persons
from src.evaluator import compute_metrics


def test_centrality_center():
    # Box dead-center in 1000x800 image
    bbox = [400, 300, 600, 500]  # Center is (500, 400)
    score = compute_centrality(bbox, 1000, 800)
    assert score == 1.0


def test_centrality_corner():
    # Box in extreme top-left corner
    bbox = [0, 0, 10, 10]
    score = compute_centrality(bbox, 1000, 800)
    assert 0.0 <= score <= 0.15


def test_relative_size_single():
    bboxes = [[100, 100, 300, 400]]
    scores = compute_relative_sizes(bboxes)
    assert scores == [1.0]


def test_relative_size_multiple():
    # Large foreground person (area: 200 * 300 = 60,000)
    # Small background photobomber (area: 50 * 60 = 3,000) -> 0.05
    bboxes = [
        [100, 100, 300, 400],
        [500, 200, 550, 260],
    ]
    scores = compute_relative_sizes(bboxes)
    assert len(scores) == 2
    assert scores[0] == 1.0
    assert 0.04 <= scores[1] <= 0.06


def test_sharpness_normalization():
    raw_vals = [1200.0, 300.0, 120.0]
    norm = normalize_sharpness(raw_vals)
    assert len(norm) == 3
    assert norm[0] == 1.0
    assert norm[1] == 0.25
    assert norm[2] == 0.10


def test_classifier_keep_and_remove():
    detections = [
        {"instance_id": 0, "bounding_box": [200, 100, 500, 700]},
        {"instance_id": 1, "bounding_box": [50, 50, 120, 180]},
    ]
    signals = {
        "centrality": [0.92, 0.10],
        "relative_size": [1.00, 0.05],
        "focus": [0.95, 0.15],
        "depth": [0.90, 0.25],
        "pose": [0.85, 0.20],
    }
    results = classify_persons(detections, signals)
    assert len(results) == 2

    # Person 0 should be "keep"
    assert results[0]["instance_id"] == 0
    assert results[0]["classification"] == "keep"
    assert results[0]["confidence_score"] >= 0.70

    # Person 1 should be "remove"
    assert results[1]["instance_id"] == 1
    assert results[1]["classification"] == "remove"
    assert results[1]["confidence_score"] >= 0.70


def test_json_output_structure():
    detections = [{"instance_id": 0, "bounding_box": [10, 10, 50, 80]}]
    signals = {
        "centrality": [0.8],
        "relative_size": [1.0],
        "focus": [0.9],
        "depth": [0.85],
        "pose": [0.75],
    }
    results = classify_persons(detections, signals)
    p = results[0]

    # Required fields from prompt requirement #3:
    # instance_id, bounding_box, classification, confidence_score, and a signals_used dictionary
    assert "instance_id" in p
    assert "bounding_box" in p
    assert "classification" in p
    assert "confidence_score" in p
    assert "signals_used" in p
    assert isinstance(p["signals_used"], dict)
    assert "centrality" in p["signals_used"]
    assert "relative_size" in p["signals_used"]
    assert "focus" in p["signals_used"]
    assert "depth" in p["signals_used"]
    assert "pose" in p["signals_used"]


def test_empty_detections():
    results = classify_persons([], {})
    assert results == []


def test_metric_calculations():
    # 8 TP, 2 FP, 1 FN, 10 TN
    m = compute_metrics(tp=8, fp=2, fn=1, tn=10)
    assert m["tp"] == 8
    assert m["precision"] == round(8 / 10, 4)  # 0.8
    assert m["recall"] == round(8 / 9, 4)      # 0.8889
    assert m["f1"] > 0.8
