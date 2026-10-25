"""
Integration Tests for FastAPI Endpoints
"""

import os
import io
import sys
import pytest
from PIL import Image
from fastapi.testclient import TestClient

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from app.main import app

client = TestClient(app)


def create_dummy_image_bytes(format="PNG", size=(32, 32), color=(100, 150, 200)) -> bytes:
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format=format)
    return buf.getvalue()


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["classes_count"] == 3


def test_model_info_endpoint():
    response = client.get("/api/model-info")
    assert response.status_code == 200
    data = response.json()
    assert data["framework"] == "Pure NumPy (Zero PyTorch/Keras dependencies)"
    assert data["flattened_features"] == 3072
    assert "Airplane" in data["classes"]
    assert "Automobile" in data["classes"]
    assert "Bird" in data["classes"]
    assert data["trainable_parameters"] > 1_000_000


def test_predict_endpoint_valid_image():
    img_bytes = create_dummy_image_bytes()
    response = client.post(
        "/api/predict",
        files={"file": ("test.png", img_bytes, "image/png")}
    )
    assert response.status_code == 200
    data = response.json()
    assert "predicted_class" in data
    assert data["predicted_class"] in ["Airplane", "Automobile", "Bird"]
    assert "confidence" in data
    assert 0.0 <= data["confidence"] <= 1.0
    assert "probabilities" in data
    assert len(data["probabilities"]) == 3
    # Check probabilities sum to 1.0
    total_prob = sum(data["probabilities"].values())
    assert abs(total_prob - 1.0) < 1e-4
    assert "latency_ms" in data


def test_predict_endpoint_invalid_file():
    response = client.post(
        "/api/predict",
        files={"file": ("document.pdf", b"%PDF-1.4 dummy content", "application/pdf")}
    )
    assert response.status_code == 400
    assert "not a valid image" in response.json()["detail"]


def test_predict_endpoint_empty_file():
    response = client.post(
        "/api/predict",
        files={"file": ("empty.png", b"", "image/png")}
    )
    assert response.status_code == 400


def test_predict_batch_endpoint():
    img1 = create_dummy_image_bytes(color=(255, 0, 0))
    img2 = create_dummy_image_bytes(color=(0, 255, 0))
    response = client.post(
        "/api/predict-batch",
        files=[
            ("files", ("img1.png", img1, "image/png")),
            ("files", ("img2.png", img2, "image/png"))
        ]
    )
    assert response.status_code == 200
    data = response.json()
    assert data["batch_size"] == 2
    assert len(data["results"]) == 2
    assert data["results"][0]["predicted_class"] in ["Airplane", "Automobile", "Bird"]


def test_metrics_endpoint():
    response = client.get("/api/metrics")
    assert response.status_code == 200
    data = response.json()
    assert data["test_accuracy"] >= 0.60
    assert data["target_met"] is True
    assert "classification_report" in data
    assert "confusion_matrix" in data


def test_samples_endpoint():
    response = client.get("/api/samples")
    assert response.status_code == 200
    data = response.json()
    assert len(data["samples"]) == 3

    # Test downloading each sample image
    for sample in data["samples"]:
        sample_url = sample["url"]
        res = client.get(sample_url)
        assert res.status_code == 200
        assert res.headers["content-type"].startswith("image/")
