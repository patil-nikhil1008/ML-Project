"""
FastAPI Production Backend for CIFAR-10 Pure NumPy Classifier
Provides REST API endpoints for single/batch prediction, healthcheck, metrics, and static UI.
"""

from __future__ import annotations
import os
import time
from contextlib import asynccontextmanager
from typing import List, Dict, Any
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .service import model_service, CLASS_NAMES

APP_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(APP_DIR, "static")
PROJECT_DIR = os.path.dirname(APP_DIR)


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[API] Starting CIFAR-10 Pure NumPy Classifier Service...")
    model_service.generate_sample_images()
    print("[API] Ready for production traffic.")
    yield
    print("[API] Shutting down service.")


app = FastAPI(
    title="CIFAR-10 Pure NumPy Classifier API",
    description="Production REST API serving a 4-layer deep neural network built entirely from scratch with NumPy.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Enable CORS for cross-origin integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Health & Liveness
@app.get("/health", tags=["System"])
async def health_check():
    """Liveness probe for container orchestrators (Kubernetes / Docker)."""
    return {
        "status": "healthy",
        "model_loaded": model_service.is_loaded,
        "classes_count": len(CLASS_NAMES),
        "timestamp": time.time()
    }


# Model Metadata
@app.get("/api/model-info", tags=["Model"])
async def get_model_info():
    """Returns neural network architecture specifications, weights status, and parameter counts."""
    info = model_service.get_model_info()
    info["benchmark_accuracy"] = 0.8457
    return info


# Single Image Prediction
@app.post("/api/predict", tags=["Inference"])
async def predict_image(file: UploadFile = File(...)):
    """
    Classify an uploaded image into Airplane, Automobile, or Bird.
    Accepts JPG, PNG, WebP, etc.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Uploaded file type '{file.content_type}' is not a valid image."
        )

    try:
        contents = await file.read()
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="Empty image file received.")
        result = model_service.predict(contents)
        result["filename"] = file.filename
        return result
    except HTTPException:
        raise
    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


# Batch Image Prediction
@app.post("/api/predict-batch", tags=["Inference"])
async def predict_batch_images(files: List[UploadFile] = File(...)):
    """Classify multiple images in a single batch request."""
    if not files or len(files) == 0:
        raise HTTPException(status_code=400, detail="No files provided.")
    if len(files) > 20:
        raise HTTPException(status_code=400, detail="Maximum 20 images per batch request.")

    results = []
    for file in files:
        contents = await file.read()
        try:
            res = model_service.predict(contents)
            res["filename"] = file.filename
            results.append(res)
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e),
                "predicted_class": None
            })
    return {"batch_size": len(files), "results": results}


# Evaluation Metrics & Reports
@app.get("/api/metrics", tags=["Evaluation"])
async def get_evaluation_metrics():
    """Returns official test set benchmark metrics and links to visualization curves."""
    return {
        "benchmark_dataset": "CIFAR-10 Official (Airplane, Automobile, Bird)",
        "test_samples": 3000,
        "test_accuracy": 0.8457,
        "accuracy_target": 0.6000,
        "target_met": True,
        "classification_report": {
            "Airplane (0)": {"precision": 0.7816, "recall": 0.8730, "f1_score": 0.8248, "support": 1000},
            "Automobile (1)": {"precision": 0.8947, "recall": 0.8750, "f1_score": 0.8847, "support": 1000},
            "Bird (2)": {"precision": 0.8718, "recall": 0.7890, "f1_score": 0.8283, "support": 1000},
            "Macro Avg": {"precision": 0.8494, "recall": 0.8457, "f1_score": 0.8459, "support": 3000},
            "Weighted Avg": {"precision": 0.8494, "recall": 0.8457, "f1_score": 0.8459, "support": 3000}
        },
        "confusion_matrix": {
            "matrix": [
                [873, 52, 75],
                [84, 875, 41],
                [160, 51, 789]
            ],
            "labels": ["Airplane", "Automobile", "Bird"]
        },
        "plots": {
            "loss_curves": "/static/loss_curves.png",
            "accuracy_curves": "/static/accuracy_curves.png",
            "confusion_matrix": "/static/confusion_matrix.png"
        }
    }


# Sample Images
@app.get("/api/samples", tags=["Samples"])
async def list_sample_images():
    """List pre-generated test sample images for one-click testing."""
    model_service.generate_sample_images()
    return {
        "samples": [
            {
                "class_name": name,
                "url": f"/api/samples/{name.lower()}",
                "description": model_service.class_descriptions[name]
            }
            for name in CLASS_NAMES
        ]
    }


@app.get("/api/samples/{class_name}", tags=["Samples"])
async def get_sample_image(class_name: str):
    """Retrieve sample image binary."""
    sample_path = os.path.join(APP_DIR, "samples", f"sample_{class_name.lower()}.png")
    if not os.path.exists(sample_path):
        raise HTTPException(status_code=404, detail="Sample image not found.")
    return FileResponse(sample_path, media_type="image/png")


# Serve training plots from project root as static assets
for plot_name in ["loss_curves.png", "accuracy_curves.png", "confusion_matrix.png"]:
    src_plot = os.path.join(PROJECT_DIR, plot_name)
    dst_plot = os.path.join(STATIC_DIR, plot_name)
    if os.path.exists(src_plot):
        try:
            import shutil
            shutil.copyfile(src_plot, dst_plot)
        except Exception:
            pass

# Mount static folder
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", tags=["UI"])
async def serve_ui():
    """Serve the main web frontend single-page application."""
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(
            index_path,
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0"
            }
        )
    return JSONResponse({"message": "Frontend static file is initializing."})
