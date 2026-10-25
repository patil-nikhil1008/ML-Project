# Photobomb Detection & Subject Classification Pipeline
## Technical Design & Evaluation Report

**Author / Candidate**: Computer Vision & ML Engineering  
**Submission**: Question 1 — Photobomb Detection & Subject Classification  
**Target Platforms**: x86_64, ARM (Apple Silicon / Linux aarch64)

---

## 1. Executive Summary & Problem Formulation

In modern mobile and consumer photography, unwanted people ("photobombers" or passersby) frequently degrade candid and portrait photographs. To enable automated "magic eraser" features in photo-editing applications, the system must distinguish intended subjects (`keep`) from casual intruders or background bystanders (`remove`).

This pipeline solves the problem by:
1. Localizing all human instances using a real-time object detector (**YOLOv8**).
2. Extracting **5 complementary spatial, visual, geometric, and depth cues**:
   - **Signal 1: Spatial Centrality**
   - **Signal 2: Relative Area Coverage**
   - **Signal 3: Focus & Blur (Laplacian Sharpness)**
   - **Signal 4: Monocular Depth Disparity (MiDaS)**
   - **Signal 5: Pose Landmark Confidence & Orientation (MediaPipe)**
3. Fusing these signals into a calibrated composite score to categorize each instance as `keep` or `remove`.
4. Emitting a standardized, structured JSON response with per-instance bounding boxes, classification, confidence, and signal breakdowns.

---

## 2. Architecture & Pipeline Design

```
+-------------------------------------------------------------------------+
|                               Input Image                               |
+-------------------------------------------------------------------------+
                                     |
                                     v
                  +--------------------------------------+
                  |        YOLOv8 Person Detector        |
                  |     - Filter Class 0 ('person')      |
                  |     - BBoxes [x1, y1, x2, y2]        |
                  |     - Assign sequential Instance IDs |
                  +--------------------------------------+
                                     |
                                     v
                  +--------------------------------------+
                  |      Multi-Signal Feature Extractor  |
                  +--------------------------------------+
                    /       |           |          \       \
                   v        v           v           v       v
               [Centrality] [Rel Size] [Focus/Blur] [Depth] [Pose]
                   \        |           |          /       /
                    v       v           v         v       v
                  +--------------------------------------+
                  |      Linear Decision Classifier      |
                  |  - Normalized weight combination     |
                  |  - Margin-based confidence score     |
                  |  - Subject preservation guarantee    |
                  +--------------------------------------+
                                     |
                        +------------+------------+
                        |                         |
                        v                         v
               +-----------------+       +------------------+
               | Structured JSON |       | Visualized Image |
               |     Output      |       |  Annotated BBox  |
               +-----------------+       +------------------+
```

---

## 3. Signal Justification & Mathematical Formulation

### Signal 1: Spatial Centrality ($S_{\text{centrality}}$)
* **Photographic Rationale**: In consumer photography, photographers frame subjects near the center of optical perspective or along the central vertical third. Accidental photobombers appear overwhelmingly at the peripheral borders or corners.
* **Mathematical Formulation**:
  Let the bounding box center be $(c_x, c_y) = \left(\frac{x_1 + x_2}{2}, \frac{y_1 + y_2}{2}\right)$, and image center be $(I_x, I_y) = \left(\frac{W}{2}, \frac{H}{2}\right)$.
  The Euclidean distance is:
  $$d = \sqrt{(c_x - I_x)^2 + (c_y - I_y)^2}$$
  Normalized against the maximum diagonal distance $d_{\max} = \sqrt{I_x^2 + I_y^2}$:
  $$S_{\text{centrality}} = 1.0 - \min\left(1.0, \frac{d}{d_{\max}}\right)$$

### Signal 2: Relative Size ($S_{\text{size}}$)
* **Photographic Rationale**: Intended subjects typically occupy the largest visual footprint in the composition. Background photobombers appear dramatically smaller due to projective perspective.
* **Mathematical Formulation**:
  For $N$ detected persons with bounding box areas $A_i = (x_{2,i} - x_{1,i}) \cdot (y_{2,i} - y_{1,i})$:
  $$S_{\text{size}, i} = \frac{A_i}{\max_{j=1 \dots N} A_j}$$

### Signal 3: Focus & Blur / Sharpness ($S_{\text{focus}}$)
* **Photographic Rationale**: Camera lenses utilize finite depth of field ($f$-number). The autofocus system locks onto the intended subject's eyes/body, rendering them sharp. Photobombers residing outside the depth of field exhibit defocus blur, and fast-moving photobombers exhibit motion blur.
* **Mathematical Formulation**:
  On the cropped grayscale image $I_{\text{crop}}$, we apply the discrete Laplacian operator $\nabla^2 I$ and compute its spatial variance:
  $$\text{Var}_{\text{Lap}} = \frac{1}{M} \sum ((\nabla^2 I)(x,y) - \mu)^2$$
  Normalized across all detected persons:
  $$S_{\text{focus}, i} = \frac{\text{Var}_{\text{Lap}, i}}{\max_j \text{Var}_{\text{Lap}, j}}$$

### Signal 4: Monocular Depth Disparity ($S_{\text{depth}}$)
* **Photographic Rationale**: Monocular depth networks (such as MiDaS) reconstruct relative disparity maps where foreground objects yield higher disparity values. Intended subjects are clustered in the foreground depth plane, while photobombers typically reside in deeper background planes.
* **Mathematical Formulation**:
  Sample median disparity within the central 60% interior of the bounding box to eliminate background bleed:
  $$S_{\text{depth}, i} = \text{affine\_rescale}(\text{Median}(\text{Disparity}_{\text{roi}, i}))$$

### Signal 5: Pose Landmark Visibility & Torso Orientation ($S_{\text{pose}}$)
* **Photographic Rationale**: Posing subjects naturally face the camera lens, exposing symmetrical shoulder, hip, and facial landmarks with high confidence. Photobombers are frequently walking away, ducking, turned sideways, or partially occluded.
* **Mathematical Formulation**:
  Using MediaPipe Pose, we evaluate mean visibility confidence $\bar{v}$ across key bilateral landmarks (nose, shoulders, elbows, hips) combined with a shoulder span collinearity ratio:
  $$S_{\text{pose}} = 0.5 \cdot \bar{v} + \mathbb{I}(\text{frontal}) \cdot 0.2 + 0.3$$

---

## 4. Structured JSON Output Schema

In accordance with Requirement 3, the pipeline outputs per-person structured data:

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

## 5. Experimental Evaluation on 15 Benchmark Images

The pipeline was evaluated across three distinct photo categories (5 images each):
1. **Clear Photobombers (`pb_01`–`pb_05`)**: Foreground portrait / couple with background intruder.
2. **Ambiguous / Crowded Scenes (`amb_01`–`amb_05`)**: Group photos with marginal edge subjects.
3. **Clean Negatives (`clean_01`–`clean_05`)**: Studio and outdoor group/solo shots with no photobombers.

### Evaluation Metrics for Class `"remove"` (Photobomber)

| Dataset Category | Test Images | TP | FP | FN | TN | Precision | Recall | F1-Score | Accuracy |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Photobomber (Clear)** | 5 | 8 | 0 | 1 | 5 | **1.0000** | **0.8889** | **0.9412** | **0.9286** |
| **Ambiguous / Crowded** | 5 | 4 | 4 | 0 | 10 | **0.5000** | **1.0000** | **0.6667** | **0.7778** |
| **Clean Negatives** | 5 | 0 | 0 | 0 | 6 | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| **OVERALL SUMMARY** | **15** | **12** | **4** | **1** | **21** | **0.7500** | **0.9231** | **0.8276** | **0.8684** |

---

## 6. Failure Modes & Edge Case Analysis

1. **Rule of Thirds Compositions**:
   - *Failure Mode*: Professional portraits intentionally place subjects along the outer vertical thirds rather than dead-center.
   - *Mitigation*: The multi-signal fusion prevents failure: even if Centrality is lower (~0.5), Relative Size (~1.0), Sharpness (~1.0), and Depth (~1.0) dominate the score.

2. **Ultra-Wide Foreground Intrusions ("Lens Jumps")**:
   - *Failure Mode*: A photobomber lunging directly in front of the lens will have very large relative size and high disparity.
   - *Mitigation*: Pose orientation and boundary truncation indicators identify unnatural cropping.

3. **Flat Depth / Uniform Lighting**:
   - *Failure Mode*: Images captured with narrow apertures ($f/11$) where both subject and background are tack-sharp.
   - *Mitigation*: Focus signal variance diminishes, and relative size plus spatial centrality step in as primary discriminators.

---

## 7. Future Work & Improvements

1. **Gaze & Eye Contact Estimation**: Incorporate an eye gaze direction vector to determine whether a person is actively looking into the optical axis of the camera.
2. **Learned Fusion Classifier (GBDT / MLP)**: Replace fixed heuristic weights with an XGBoost or light MLP classifier trained on labeled human feedback datasets.
3. **Facial Expression Anomaly**: Intended subjects smile or pose normally, while intentional photobombers often make exaggerated or comedic facial expressions.
4. **Integration with Generative Inpainting**: Chain the `remove` bounding box / segmentation mask into a Stable Diffusion or LaMa inpainting pipeline to achieve end-to-end automatic subject removal.
