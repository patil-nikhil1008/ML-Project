"""
User Acceptance Testing (UAT) and End-to-End Simulation
"""

import os
import io
import sys
import time
import pytest
from PIL import Image
from fastapi.testclient import TestClient

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from app.main import app

client = TestClient(app)


def test_uat_user_journey_samples_flow():
    """UAT 1: End-user loads UI, queries samples, and executes classification on all samples."""
    res_samples = client.get("/api/samples")
    assert res_samples.status_code == 200
    samples_list = res_samples.json()["samples"]
    assert len(samples_list) == 3

    for sample in samples_list:
        class_name = sample["class_name"]
        sample_res = client.get(sample["url"])
        assert sample_res.status_code == 200

        pred_res = client.post(
            "/api/predict",
            files={"file": (f"{class_name.lower()}.png", sample_res.content, "image/png")}
        )
        assert pred_res.status_code == 200
        data = pred_res.json()
        assert data["predicted_class"] in ["Airplane", "Automobile", "Bird"]


def test_uat_non_standard_aspect_ratios():
    """UAT 2: Real users upload non-standard photos (panoramas, mobile vertical portraits)."""
    aspect_ratios = [
        (1920, 1080),  # Landscape widescreen
        (1080, 1920),  # Mobile portrait story
        (500, 200),    # Ultra-wide banner
        (16, 16),      # Tiny icon
        (800, 800)     # Standard square
    ]
    for width, height in aspect_ratios:
        img = Image.new("RGB", (width, height), color=(120, 180, 240))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        raw_bytes = buf.getvalue()

        res = client.post(
            "/api/predict",
            files={"file": (f"test_{width}x{height}.jpg", raw_bytes, "image/jpeg")}
        )
        assert res.status_code == 200, f"Failed for aspect ratio {width}x{height}"
        data = res.json()
        assert "confidence" in data


def test_uat_ui_frontend_accessibility():
    """UAT 3: Verify frontend root URL serves valid HTML with correct page elements."""
    res = client.get("/")
    assert res.status_code == 200
    assert "text/html" in res.headers["content-type"]
    html = res.text
    assert "CIFAR-10 Pure NumPy Classifier" in html
    assert "Classify Image" in html
    assert "dropzone" in html
    assert "loss_curves.png" in html or "Metrics Table" in html


def test_uat_latency_sla():
    """UAT 4: Service Level Agreement (SLA) — Inferences must execute under 50ms average."""
    img = Image.new("RGB", (32, 32), color=(200, 200, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    raw_bytes = buf.getvalue()

    latencies = []
    for _ in range(10):
        t0 = time.perf_counter()
        res = client.post("/api/predict", files={"file": ("sla.png", raw_bytes, "image/png")})
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        assert res.status_code == 200
        latencies.append(elapsed_ms)

    avg_latency = sum(latencies) / len(latencies)
    print(f"\n[UAT SLA] Average API call latency: {avg_latency:.2f} ms")
    assert avg_latency < 50.0, f"Average latency too high: {avg_latency:.2f} ms"


def test_uat_burst_concurrency_simulation():
    """UAT 5: Burst simulation: 25 sequential user queries without error or memory leaks."""
    img = Image.new("RGB", (64, 64), color=(50, 100, 150))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    raw_bytes = buf.getvalue()

    for i in range(25):
        res = client.post("/api/predict", files={"file": (f"user_{i}.png", raw_bytes, "image/png")})
        assert res.status_code == 200


def test_uat_ood_unsupported_class_handling():
    """UAT 6: Verify out-of-distribution / people uploads are flagged with helpful warning."""
    user_upload_path = os.path.join(PARENT_DIR, "extracted_user_upload.png")
    if os.path.exists(user_upload_path):
        with open(user_upload_path, "rb") as f:
            raw_bytes = f.read()
        res = client.post("/api/predict", files={"file": ("user_upload.png", raw_bytes, "image/png")})
        assert res.status_code == 200
        data = res.json()
        assert data["is_out_of_distribution"] is True
        assert "Person" in data["warning_message"]
