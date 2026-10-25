"""Integration tests for FastAPI backend endpoints."""

import pytest
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


def test_healthcheck():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "YOLOv8" in data["model"]
    assert len(data["signals"]) == 5


def test_get_samples():
    response = client.get("/api/samples")
    assert response.status_code == 200
    data = response.json()
    assert "samples" in data
    assert len(data["samples"]) > 0


def test_evaluate_endpoint():
    response = client.post("/api/evaluate")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "overall" in data
    assert data["overall"]["f1"] > 0.70


def test_reclassify_fast_endpoint():
    sample_persons = [
        {
            "instance_id": 0,
            "bounding_box": [100, 100, 400, 500],
            "signals_used": {
                "centrality": 0.95,
                "relative_size": 1.0,
                "focus": 0.9,
                "depth": 0.9,
                "pose": 0.8,
            },
        },
        {
            "instance_id": 1,
            "bounding_box": [10, 10, 50, 80],
            "signals_used": {
                "centrality": 0.15,
                "relative_size": 0.05,
                "focus": 0.2,
                "depth": 0.3,
                "pose": 0.3,
            },
        },
    ]

    # At threshold 0.62: ID 0 is keep, ID 1 is remove
    res = client.post(
        "/api/reclassify",
        json={"persons": sample_persons, "threshold": 0.62},
    )
    assert res.status_code == 200
    data = res.json()
    assert len(data["persons"]) == 2
    assert data["persons"][0]["classification"] == "keep"
    assert data["persons"][1]["classification"] == "remove"


def test_frontend_index_served():
    response = client.get("/")
    assert response.status_code == 200
    assert "Photobomb" in response.text
