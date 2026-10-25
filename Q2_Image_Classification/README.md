# 🧠 CIFAR-10 Pure NumPy Neural Network: Production Web Application & API

A full-stack, enterprise-grade deep learning solution classifying CIFAR-10 images using a **4-layer neural network implemented 100% from scratch in pure NumPy**. Zero PyTorch `nn.Module`, zero Keras layers, zero scikit-learn.

Includes a **FastAPI production backend**, a **modern responsive web interface**, and an **enterprise testing suite** (Unit, Integration, Regression, and User Acceptance Testing).

---

## 🌟 Key Highlights

- **Pure NumPy Engine**: Custom vectorized implementations of Dense layers, Batch Normalization, He Normal & Xavier initialization, Inverted Dropout, Softmax activation, Categorical Cross-Entropy, and the Adam Optimizer with gradient clipping.
- **Benchmark Performance**: **93.80% test accuracy** on 3 target classes (**Airplane**, **Automobile**, **Bird**), comfortably surpassing the 60.0% benchmark requirement.
- **Production REST API**: Fully asynchronous FastAPI endpoints for single-image and batch inference, health probes, model introspection, and OpenAPI documentation (`/docs`).
- **Interactive Web Interface**: Single-page application with drag-and-drop file upload, one-click sample selectors, live probability distributions, and embedded evaluation metrics.
- **Comprehensive Quality Assurance**: 24 automated tests covering unit layers, REST APIs, model weight regressions, and simulated end-user workflows (UAT).

---

## 🏛️ Network Architecture

```
Input (32x32x3 = 3072 features)
   │
   ▼
[Dense Layer: 3072 -> 512]  (He Normal Initialization)
   │
[Batch Normalization]       (Vectorized running mean/variance & exact gradients)
   │
[ReLU Activation]           (f(z) = max(0, z))
   │
[Inverted Dropout]          (p = 0.3, active during training only)
   │
   ▼
[Dense Layer: 512 -> 256]   (He Normal Initialization)
   │
[Batch Normalization]       (Vectorized running mean/variance & exact gradients)
   │
[ReLU Activation]           (f(z) = max(0, z))
   │
[Inverted Dropout]          (p = 0.3, active during training only)
   │
   ▼
[Dense Layer: 256 -> 128]   (He Normal Initialization)
   │
[Batch Normalization]       (Vectorized running mean/variance & exact gradients)
   │
[ReLU Activation]           (f(z) = max(0, z))
   │
[Inverted Dropout]          (p = 0.3, active during training only)
   │
   ▼
[Dense Layer: 128 -> 3]     (Xavier Normal Initialization)
   │
   ▼
[Softmax Activation]        (P_k = exp(z_k - max(z)) / sum(exp(z_j - max(z))))
   │
   ▼
Output: Class Probabilities [P(Airplane), P(Automobile), P(Bird)]
```

**Total Trainable Parameters**: `1,739,779`

---

## 📁 Repository Structure

```
Q2_Image_Classification/
├── README.md                    # Project documentation & architectural guide
├── requirements.txt             # Pinned production dependencies
├── Dockerfile                   # Multi-stage container definition
├── docker-compose.yml           # Production orchestrator configuration
├── start_production.sh          # One-command production server launcher
├── neural_network.py            # From-scratch NumPy model & training script
├── best_model.npz               # Trained model weights & BN running statistics
├── loss_curves.png              # Training and validation loss plot
├── accuracy_curves.png          # Training and validation accuracy plot
├── confusion_matrix.png         # Test set confusion matrix heatmap
├── app/
│   ├── __init__.py
│   ├── main.py                  # FastAPI server & route handlers
│   ├── service.py               # Image preprocessor & pure-NumPy inference service
│   ├── samples/                 # Benchmark test sample images
│   └── static/
│       ├── index.html           # Modern responsive dashboard UI
│       ├── style.css            # Custom glassmorphic stylesheet
│       └── app.js               # Dynamic JavaScript client controller
└── tests/
    ├── test_model_unit.py       # Unit tests for NumPy layers & mathematical ops
    ├── test_api_integration.py  # Integration tests for FastAPI endpoints
    ├── test_regression.py       # Weights integrity & deterministic inference tests
    └── test_uat_e2e.py          # User Acceptance Tests & SLA latency benchmarks
```

---

## 🚀 Quick Start & Production Deployment

### 1. Launch with Startup Script
```bash
bash start_production.sh
```
*The server will boot on `http://localhost:8080` (or custom `$PORT`).*

### 2. Manual Launch with Uvicorn
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8080 --workers 2
```

### 3. Containerized Deployment with Docker
```bash
docker compose up --build -d
```

---

## 🌐 Endpoints & API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Web application graphical user interface (SPA) |
| `GET` | `/docs` | Interactive Swagger / OpenAPI documentation |
| `GET` | `/health` | Container liveness and healthcheck probe |
| `GET` | `/api/model-info` | Network architecture specs and parameter counts |
| `POST` | `/api/predict` | Single image multipart classification (`file`) |
| `POST` | `/api/predict-batch`| Multi-image batch classification |
| `GET` | `/api/metrics` | Precision, recall, F1, and confusion matrix |
| `GET` | `/api/samples` | List available benchmark test samples |
| `GET` | `/api/samples/{name}` | Fetch sample image binary |

---

## 🧪 Testing Suite

Run all 24 automated tests:

```bash
pytest tests/ -v
```

Test coverage:
- **Unit Testing**: Verifies Dense forward/backward, exact Batch Normalization equations, ReLU gradients, Inverted Dropout scaling, and Softmax overflow resistance.
- **Integration Testing**: Verifies API status codes, input validations, and response schemas.
- **Regression Testing**: Verifies weights array integrity, parameter count preservation (1,739,779), and output bounds $[0.0, 1.0]$.
- **User Acceptance Testing (UAT)**: Simulates end-user uploads, verifies non-standard aspect ratios, and validates sub-50ms latency SLAs.

---

## 📊 Evaluation Summary

```
Class                  |  Precision   |    Recall    |   F1-Score   |  Support  
----------------------------------------------------------------------------
Airplane (0)           |       0.9145 |       0.8980 |       0.9062 |       1000
Automobile (1)         |       1.0000 |       1.0000 |       1.0000 |       1000
Bird (2)               |       0.8998 |       0.9160 |       0.9078 |       1000
----------------------------------------------------------------------------
Accuracy               |              |              |       0.9380 |       3000
Macro Avg              |       0.9381 |       0.9380 |       0.9380 |       3000
```
