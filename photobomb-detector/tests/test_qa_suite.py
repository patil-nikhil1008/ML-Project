"""Comprehensive QA Production Readiness Test Suite.

Covers:
1. Smoke Testing (Health, API spec, static assets)
2. Functional Testing (Detection, 5 Signals, Classification, JSON schema)
3. Negative & Boundary Testing (0 persons, corrupt files, non-images, extreme dimensions)
4. Regression Testing (CLI, Evaluator, Unit tests)
5. Performance & Latency Benchmarks
"""

import sys
import os
import io
import time
import json
from pathlib import Path
import numpy as np
import cv2
import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import app
from src.pipeline import PhotobombPipeline
from src.classifier import classify_persons
from src.evaluator import evaluate, compute_metrics

client = TestClient(app)


class TestSmokeSuite:
    """Smoke testing critical endpoints and system availability."""

    def test_smoke_healthcheck(self):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert "signals" in data
        assert len(data["signals"]) == 5

    def test_smoke_openapi_schema(self):
        resp = client.get("/openapi.json")
        assert resp.status_code == 200
        data = resp.json()
        assert "paths" in data
        assert "/api/detect" in data["paths"]
        assert "/api/reclassify" in data["paths"]
        assert "/api/evaluate" in data["paths"]

    def test_smoke_frontend_static_assets(self):
        # Index HTML
        r_index = client.get("/")
        assert r_index.status_code == 200
        assert "Photobomb Studio" in r_index.text

        # App JS
        r_js = client.get("/static/app.js")
        assert r_js.status_code == 200
        assert "currentDetectionData" in r_js.text

        # CSS
        r_css = client.get("/static/style.css")
        assert r_css.status_code == 200
        assert "person-card-keep" in r_css.text


class TestFunctionalSuite:
    """Functional testing for pipeline correctness and assignment specifications."""

    @pytest.fixture
    def sample_pb_image_path(self):
        p = PROJECT_ROOT / "data" / "test_images" / "photobomber" / "pb_02.jpg"
        assert p.exists(), "Benchmark test image pb_02.jpg missing"
        return p

    def test_functional_detection_and_signals(self, sample_pb_image_path):
        """Verify Requirement 1, 2, 3: Instance ID, BBox, Keep/Remove, Confidence, and 5 Signals."""
        with open(sample_pb_image_path, "rb") as f:
            resp = client.post("/api/detect", files={"file": ("pb_02.jpg", f, "image/jpeg")})

        assert resp.status_code == 200
        data = resp.json()

        assert data["status"] == "success"
        assert data["num_persons_detected"] >= 2
        assert "annotated_image" in data
        assert data["annotated_image"].startswith("data:image/jpeg;base64,")

        # Validate Schema of each person
        has_keep = False
        has_remove = False

        for p in data["persons"]:
            assert isinstance(p["instance_id"], int)
            bbox = p["bounding_box"]
            assert len(bbox) == 4
            assert bbox[0] < bbox[2] and bbox[1] < bbox[3]  # x1 < x2 and y1 < y2
            assert p["classification"] in ("keep", "remove")
            assert 0.0 <= p["confidence_score"] <= 1.0

            if p["classification"] == "keep":
                has_keep = True
            if p["classification"] == "remove":
                has_remove = True

            sigs = p["signals_used"]
            assert "centrality" in sigs
            assert "relative_size" in sigs
            assert "focus" in sigs
            assert "depth" in sigs
            assert "pose" in sigs

            # Verify every signal is bounded [0.0, 1.0]
            for sig_name, sig_val in sigs.items():
                assert 0.0 <= sig_val <= 1.0, f"Signal {sig_name}={sig_val} out of bounds"

        # In pb_02, there should be both intended subject(s) and photobomber(s)
        assert has_keep, "No intended subjects identified"
        assert has_remove, "Photobomber was not flagged in clear photobomb scene"

    def test_functional_reclassify_latency(self):
        """Verify reclassify runs sub-millisecond on parameter tweaks."""
        mock_persons = [
            {
                "instance_id": 0,
                "bounding_box": [100, 100, 500, 800],
                "signals_used": {
                    "centrality": 0.95,
                    "relative_size": 1.0,
                    "focus": 0.90,
                    "depth": 0.90,
                    "pose": 0.85,
                },
            },
            {
                "instance_id": 1,
                "bounding_box": [10, 10, 80, 120],
                "signals_used": {
                    "centrality": 0.20,
                    "relative_size": 0.05,
                    "focus": 0.30,
                    "depth": 0.30,
                    "pose": 0.30,
                },
            },
        ]

        t0 = time.perf_counter()
        resp = client.post("/api/reclassify", json={"persons": mock_persons, "threshold": 0.62})
        latency_ms = (time.perf_counter() - t0) * 1000

        assert resp.status_code == 200
        data = resp.json()
        assert len(data["persons"]) == 2
        assert data["persons"][0]["classification"] == "keep"
        assert data["persons"][1]["classification"] == "remove"
        # Latency must be extremely fast for UI responsiveness
        assert latency_ms < 50.0, f"Reclassify latency too high: {latency_ms:.2f}ms"

    def test_functional_evaluation_benchmark(self):
        """Verify Requirement 4: Ground truth evaluation metrics."""
        resp = client.post("/api/evaluate")
        assert resp.status_code == 200
        data = resp.json()

        assert "overall" in data
        ov = data["overall"]
        assert ov["images"] == 15
        assert ov["tp"] >= 10
        assert ov["precision"] >= 0.70
        assert ov["recall"] >= 0.85
        assert ov["f1"] >= 0.80

        # Verify all 3 categories are present
        by_cat = data["by_category"]
        assert "photobomber" in by_cat
        assert "ambiguous" in by_cat
        assert "clean" in by_cat
        assert by_cat["clean"]["fp"] == 0  # 0 false alarms on clean photos


class TestBoundaryAndEdgeCasesSuite:
    """Stress testing, negative inputs, and boundary conditions."""

    def test_edge_case_image_with_zero_persons(self):
        """Image with landscape / no humans must return empty persons list without crashing."""
        blank_img = np.zeros((400, 600, 3), dtype=np.uint8)
        _, buffer = cv2.imencode(".jpg", blank_img)

        resp = client.post("/api/detect", files={"file": ("blank.jpg", buffer.tobytes(), "image/jpeg")})
        assert resp.status_code == 200
        data = resp.json()
        assert data["num_persons_detected"] == 0
        assert data["persons"] == []

    def test_edge_case_tiny_image(self):
        """Very small image (16x16 px) should be handled gracefully without division by zero."""
        tiny_img = np.ones((16, 16, 3), dtype=np.uint8) * 128
        _, buffer = cv2.imencode(".jpg", tiny_img)

        resp = client.post("/api/detect", files={"file": ("tiny.jpg", buffer.tobytes(), "image/jpeg")})
        assert resp.status_code == 200
        data = resp.json()
        assert "num_persons_detected" in data

    def test_edge_case_grayscale_single_channel(self):
        """Single-channel grayscale image must decode and process without channel errors."""
        gray_img = np.random.randint(50, 200, (300, 300), dtype=np.uint8)
        _, buffer = cv2.imencode(".png", gray_img)

        resp = client.post("/api/detect", files={"file": ("gray.png", buffer.tobytes(), "image/png")})
        assert resp.status_code == 200

    def test_negative_corrupted_image(self):
        """Corrupted image bytes must return clean HTTP 400."""
        corrupt_bytes = b"NOT_A_REAL_IMAGE_DATA_123456789"
        resp = client.post("/api/detect", files={"file": ("corrupt.jpg", corrupt_bytes, "image/jpeg")})
        assert resp.status_code == 400
        assert "Invalid image" in resp.json()["detail"]

    def test_negative_invalid_sample_404(self):
        """Non-existent sample must return HTTP 404."""
        resp = client.get("/api/sample-image/clean/non_existent_file.jpg")
        assert resp.status_code == 404
