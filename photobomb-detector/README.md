# 📸 Photobomb Detection & Subject Classification Pipeline

An intelligent computer vision and machine learning pipeline that automatically identifies intended subjects in a photograph and flags photobombers for removal.

Compatible with **macOS (Apple Silicon / Intel)**, **Linux (x86_64 / aarch64)**, and **Windows**.

---

## 🌟 Highlights

- **YOLOv8 Detection**: Accurately localizes every person instance with unique sequential IDs and bounding boxes.
- **Multi-Signal Classification (5 Visual & Spatial Signals)**:
  1. **Spatial Centrality**: Evaluates proximity to the primary compositional center.
  2. **Relative Size**: Measures scale and bounding box coverage relative to the primary subject.
  3. **Focus / Blur**: Measures lens sharpness via Laplacian variance.
  4. **Monocular Depth**: Disparity estimation via MiDaS for foreground vs background depth clustering.
  5. **Pose Confidence & Facing Direction**: MediaPipe pose landmark confidence and torso orientation.
- **Structured JSON Output**: Every person instance is returned with bounding boxes, binary classification (`keep` vs `remove`), calibrated confidence score, and individual signal contributions.
- **Evaluation Suite**: Automated benchmark reporting **Precision**, **Recall**, **F1-Score**, and **Accuracy** on the `remove` class.

---

## 📁 Repository Structure

```
photobomb-detector/
├── README.md                    # Setup, instructions, and architecture
├── requirements.txt             # Python dependencies
├── setup.sh                     # One-command automated environment setup
├── app.py                       # FastAPI Web Backend (REST API + Static Server)
├── demo.py                      # CLI entry point for single image or batch inference
├── web/                         # Frontend Web Studio
│   ├── index.html               # Single-page modern UI
│   ├── app.js                   # Interactive canvas, slider tuning & API calls
│   └── style.css                # Dark-mode responsive styling
├── src/
│   ├── __init__.py
│   ├── pipeline.py              # Main pipeline orchestrator & visualizer
│   ├── detector.py              # YOLOv8 person detection wrapper
│   ├── classifier.py            # Multi-signal decision fusion & confidence scoring
│   ├── evaluator.py             # Precision, recall, F1 evaluation harness
│   └── signals/
│       ├── __init__.py
│       ├── centrality.py        # Signal 1: Spatial Centrality
│       ├── relative_size.py     # Signal 2: Relative Area Ratio
│       ├── focus_blur.py        # Signal 3: Laplacian Sharpness
│       ├── depth.py             # Signal 4: Monocular Depth (MiDaS)
│       └── pose.py              # Signal 5: MediaPipe Pose Analysis
├── data/
│   ├── test_images/
│   │   ├── photobomber/         # 5 benchmark images with clear photobombers
│   │   ├── ambiguous/           # 5 ambiguous / crowded scene images
│   │   └── clean/               # 5 clean negative portraits
│   └── ground_truth.json        # Ground truth annotations
├── scripts/
│   ├── download_test_images.py  # Benchmark image fetcher & generator
│   ├── annotate_helper.py       # Annotation tool
│   └── record_demo.sh           # Demo runner for screen recording
├── tests/
│   ├── test_pipeline.py         # Pipeline & Signal unit tests
│   └── test_api.py              # FastAPI integration tests
├── output/                      # Output JSON results and visualized images
└── report/
    └── report.md                # Comprehensive technical design report
```

---

## 🚀 Quick Start Guide

### 1. Automated Setup

Clone the repository and run the setup script:

```bash
git clone <your-repo-url>
cd photobomb-detector
bash setup.sh
```

Or manually:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Launch Interactive Web UI / Studio 🌐

Launch the full-stack web application:

```bash
python app.py
```

Then open your browser at:
- **Web App**: [http://localhost:8000](http://localhost:8000)
- **Interactive Swagger API Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)

**Features in the Web App:**
- 📁 **Drag-and-drop** any image or click **1-click benchmark test samples**.
- 🎯 **Interactive Canvas**: Hover over any bounding box to highlight signals and view tooltips.
- 🎛️ **Live Sensitivity Tuner**: Slide the Decision Threshold (`0.30`–`0.80`) or adjust the 5 Signal Weights with instant sub-millisecond reclassification.
- 📄 **Structured JSON Inspector**: Copy or download formatted JSON per Requirement #3.
- ⚡ **Built-In Benchmark Runner**: Run the 15-image evaluation suite directly in the browser.

---

### 3. CLI Usage (Terminal Demo)

**Process a single image:**
```bash
python demo.py --input path/to/photo.jpg --output output/
```

**Process an entire folder in batch:**
```bash
python demo.py --input-dir data/test_images/ --output output/
```

Outputs will be saved in `output/`:
- `output/<name>_result.json`: Structured per-person JSON results.
- `output/<name>_annotated.jpg`: Visualized image with green (`KEEP`) and red (`REMOVE`) bounding boxes.

### 4. Evaluate Precision, Recall, and F1

```bash
python -m src.evaluator --predictions output/ --ground-truth data/ground_truth.json
```

### 5. Run Unit Tests

```bash
pytest tests/ -v
```

---

## 📊 Sample Output Schema

Per-person JSON structure output by the pipeline:

```json
{
  "instance_id": 1,
  "bounding_box": [680, 210, 780, 490],
  "classification": "remove",
  "confidence_score": 0.8924,
  "signals_used": {
    "centrality": 0.2140,
    "relative_size": 0.1120,
    "focus": 0.3150,
    "depth": 0.3400,
    "pose": 0.2800
  }
}
```

---

## 📈 Evaluation Results (Benchmark Summary)

Target evaluation class: **`remove`** (Photobomber)

| Category | Images | Precision | Recall | F1-Score | Accuracy |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Photobomber** | 5 | 1.0000 | 0.8889 | 0.9412 | 0.9286 |
| **Ambiguous** | 5 | 0.5000 | 1.0000 | 0.6667 | 0.7778 |
| **Clean** | 5 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| **OVERALL** | **15** | **0.7500** | **0.9231** | **0.8276** | **0.8684** |

---

## 🎥 Screen Recording

To create your screen recording / demo presentation:
```bash
bash scripts/record_demo.sh
```
This runs the full end-to-end pipeline and evaluation suite in an interactive terminal format suitable for video capture.

---

## 📄 Technical Report

Read the comprehensive technical report in [report/report.md](report/report.md) for architectural trade-offs, signal justifications, failure modes, and future enhancements.
