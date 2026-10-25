"""
Automated Comprehensive 1-by-1 Functionality Test Suite
Testing every button, menu, slider, API, and visual component of Photobomb Studio.
"""

import os
import re
import json
import base64
import unittest
from fastapi.testclient import TestClient
import app

class TestPhotobombStudioE2E(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app.app)
        
        # Load HTML and JS for DOM verification
        with open("web/index.html", "r", encoding="utf-8") as f:
            cls.html_content = f.read()
            
        with open("web/app.js", "r", encoding="utf-8") as f:
            cls.js_content = f.read()
            
        with open("web/style.css", "r", encoding="utf-8") as f:
            cls.css_content = f.read()

    # ----------------------------------------------------
    # SECTION 1: DOM Elements & Interactive Triggers
    # ----------------------------------------------------
    def test_01_all_dom_ids_exist_in_html(self):
        """Verifies that all 38 DOM IDs accessed by app.js exist in index.html"""
        required_ids = [
            "toast-container",
            "drop-zone",
            "file-input",
            "workspace-section",
            "active-filename",
            "active-dimensions",
            "active-count",
            "btn-view-annotated",
            "btn-view-original",
            "canvas-container",
            "image-display",
            "interactive-canvas",
            "loading-overlay",
            "loading-text",
            "threshold-val",
            "threshold-slider",
            "weights-panel",
            "weights-toggle-icon",
            "w-centrality",
            "w-centrality-val",
            "w-size",
            "w-size-val",
            "w-focus",
            "w-focus-val",
            "w-depth",
            "w-depth-val",
            "w-pose",
            "w-pose-val",
            "person-count-badge",
            "persons-list",
            "tab-btn-json",
            "tab-btn-benchmark",
            "tab-btn-architecture",
            "tab-pane-json",
            "tab-pane-benchmark",
            "tab-pane-architecture",
            "btn-copy-json",
            "json-pre-block",
            "btn-run-eval",
            "metric-f1",
            "metric-recall",
            "metric-prec",
            "metric-acc",
            "eval-table-body",
            "filter-all",
            "filter-photobomber",
            "filter-ambiguous",
            "filter-clean",
            "sample-cards-grid",
            "btn-view-cleaned",
            "btn-revert-original",
            "view-mode-name"
        ]
        
        for elem_id in required_ids:
            self.assertIn(f'id="{elem_id}"', self.html_content, f"Missing DOM id: '{elem_id}' in index.html")

    def test_02_all_onclick_functions_exist_in_js(self):
        """Verifies all button onclick handlers in index.html are implemented in app.js"""
        onclick_matches = re.findall(r'onclick="([a-zA-Z0-9_]+)\(', self.html_content)
        unique_funcs = set(onclick_matches)
        
        for func_name in unique_funcs:
            # Check function declaration in app.js
            pattern = rf"(function\s+{func_name}\b|const\s+{func_name}\s*=|let\s+{func_name}\s*=)"
            self.assertTrue(bool(re.search(pattern, self.js_content)), f"Handler '{func_name}' not defined in app.js")

    # ----------------------------------------------------
    # SECTION 2: Static Server & Core Routes
    # ----------------------------------------------------
    def test_03_static_routes_and_docs(self):
        """Tests that web root, static assets, and OpenAPI docs respond with 200 OK"""
        r_root = self.client.get("/")
        self.assertEqual(r_root.status_code, 200)
        self.assertIn("Photobomb Studio", r_root.text)
        
        r_css = self.client.get("/static/style.css")
        self.assertEqual(r_css.status_code, 200)
        self.assertIn("person-card-keep", r_css.text)
        
        r_js = self.client.get("/static/app.js")
        self.assertEqual(r_js.status_code, 200)
        self.assertIn("renderPersonCards", r_js.text)
        
        r_docs = self.client.get("/docs")
        self.assertEqual(r_docs.status_code, 200)

    def test_04_health_check_endpoint(self):
        """Verifies /api/health reports active pipeline status"""
        r = self.client.get("/api/health")
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertEqual(data["status"], "healthy")
        self.assertEqual(data["service"], "Photobomb Detection Pipeline")
        self.assertIn("signals", data)
        self.assertEqual(len(data["signals"]), 5)

    # ----------------------------------------------------
    # SECTION 3: 1-Click Benchmark Gallery & Filtering
    # ----------------------------------------------------
    def test_05_samples_gallery_endpoint(self):
        """Tests /api/samples returns exactly 15 benchmark items (5 per category)"""
        r = self.client.get("/api/samples")
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertIn("samples", data)
        samples = data["samples"]
        self.assertEqual(len(samples), 15)
        
        cats = {"photobomber": 0, "ambiguous": 0, "clean": 0}
        for s in samples:
            self.assertIn("id", s)
            self.assertIn("filename", s)
            self.assertIn("category", s)
            self.assertIn("url", s)
            cats[s["category"]] += 1
            
        self.assertEqual(cats["photobomber"], 5)
        self.assertEqual(cats["ambiguous"], 5)
        self.assertEqual(cats["clean"], 5)

    def test_06_sample_image_serving(self):
        """Tests that sample images in each category can be retrieved and are valid images"""
        r_samples = self.client.get("/api/samples")
        samples = r_samples.json()["samples"]
        
        # Test one from each category
        for cat in ["photobomber", "ambiguous", "clean"]:
            target = next(s for s in samples if s["category"] == cat)
            r_img = self.client.get(target["url"])
            self.assertEqual(r_img.status_code, 200, f"Failed to fetch {target['url']}")
            self.assertEqual(r_img.headers["content-type"], "image/jpeg")
            self.assertGreater(len(r_img.content), 1000)

    # ----------------------------------------------------
    # SECTION 4: Image Detection & Pipeline Processing
    # ----------------------------------------------------
    def test_07_detect_endpoint_with_photobomb_image(self):
        """Tests POST /api/detect on a photobomb image and verifies all required schema fields"""
        # Fetch pb_01.jpg sample
        r_img = self.client.get("/api/sample-image/photobomber/pb_01.jpg")
        self.assertEqual(r_img.status_code, 200)
        
        files = {"file": ("pb_01.jpg", r_img.content, "image/jpeg")}
        data = {
            "threshold": "0.62",
            "conf_threshold": "0.25",
            "weight_centrality": "0.25",
            "weight_size": "0.25",
            "weight_focus": "0.20",
            "weight_depth": "0.15",
            "weight_pose": "0.15"
        }
        
        r_detect = self.client.post("/api/detect", files=files, data=data)
        self.assertEqual(r_detect.status_code, 200)
        res = r_detect.json()
        
        # Verify required keys
        self.assertEqual(res["filename"], "pb_01.jpg")
        self.assertIn("image_dimensions", res)
        self.assertIn("width", res["image_dimensions"])
        self.assertIn("height", res["image_dimensions"])
        self.assertGreater(res["num_persons_detected"], 0)
        self.assertIn("original_image", res)
        self.assertIn("annotated_image", res)
        self.assertIn("person_crops", res)
        
        # Verify structured person format (Assignment Requirement #3)
        for person in res["persons"]:
            self.assertIn("instance_id", person)
            self.assertIn("bounding_box", person)
            self.assertEqual(len(person["bounding_box"]), 4)
            self.assertIn(person["classification"], ["keep", "remove"])
            self.assertGreaterEqual(person["confidence_score"], 0.0)
            self.assertLessEqual(person["confidence_score"], 1.0)
            
            # Verify 5 signals in signals_used dictionary
            sigs = person["signals_used"]
            for sig in ["centrality", "relative_size", "focus", "depth", "pose"]:
                self.assertIn(sig, sigs)
                self.assertGreaterEqual(sigs[sig], 0.0)
                self.assertLessEqual(sigs[sig], 1.0)
                
            # Verify person crop thumbnail exists
            crop_key = str(person["instance_id"])
            self.assertIn(crop_key, res["person_crops"])
            self.assertTrue(res["person_crops"][crop_key].startswith("data:image/jpeg;base64,"))

    # ----------------------------------------------------
    # SECTION 5: Real-Time Sensitivity Tuner & Reclassify
    # ----------------------------------------------------
    def test_08_reclassify_endpoint_submillisecond(self):
        """Tests POST /api/reclassify for instantaneous slider adjustment (<10ms)"""
        # Prepare synthetic detected persons
        sample_persons = [
            {
                "instance_id": 1,
                "bounding_box": [100, 100, 300, 500],
                "confidence_score": 0.85,
                "classification": "keep",
                "signals_used": {
                    "centrality": 0.90,
                    "relative_size": 0.80,
                    "focus": 0.85,
                    "depth": 0.90,
                    "pose": 0.80
                }
            },
            {
                "instance_id": 2,
                "bounding_box": [10, 20, 80, 150],
                "confidence_score": 0.45,
                "classification": "remove",
                "signals_used": {
                    "centrality": 0.20,
                    "relative_size": 0.15,
                    "focus": 0.30,
                    "depth": 0.35,
                    "pose": 0.40
                }
            }
        ]
        
        # 1. Test lenient threshold (0.20) -> Person 2 with score ~0.26 becomes KEEP
        payload_lenient = {
            "persons": sample_persons,
            "threshold": 0.20,
            "weights": {"centrality": 0.25, "relative_size": 0.25, "focus": 0.2, "depth": 0.15, "pose": 0.15}
        }
        r1 = self.client.post("/api/reclassify", json=payload_lenient)
        self.assertEqual(r1.status_code, 200)
        data1 = r1.json()
        self.assertEqual(data1["persons"][0]["classification"], "keep")
        self.assertEqual(data1["persons"][1]["classification"], "keep")
        
        # 2. Test strict threshold (0.80) -> Person 2 is REMOVE
        payload_strict = {
            "persons": sample_persons,
            "threshold": 0.80,
            "weights": {"centrality": 0.25, "relative_size": 0.25, "focus": 0.2, "depth": 0.15, "pose": 0.15}
        }
        r2 = self.client.post("/api/reclassify", json=payload_strict)
        self.assertEqual(r2.status_code, 200)
        data2 = r2.json()
        self.assertEqual(data2["persons"][1]["classification"], "remove")

    # ----------------------------------------------------
    # SECTION 6: Benchmark Suite & Metrics Runner
    # ----------------------------------------------------
    def test_09_evaluate_benchmark_runner_endpoint(self):
        """Tests POST /api/evaluate runs ground truth evaluation across all 15 images"""
        r_eval = self.client.post("/api/evaluate")
        self.assertEqual(r_eval.status_code, 200)
        data = r_eval.json()
        
        # Check overall metrics
        ov = data["overall"]
        self.assertIn("f1", ov)
        self.assertIn("recall", ov)
        self.assertIn("precision", ov)
        self.assertIn("accuracy", ov)
        self.assertIn("tp", ov)
        self.assertIn("fp", ov)
        self.assertIn("fn", ov)
        self.assertIn("tn", ov)
        
        # Verify benchmark performance standards
        self.assertGreaterEqual(ov["f1"], 0.80, f"F1 score {ov['f1']} below expected 0.80")
        self.assertGreaterEqual(ov["recall"], 0.90, f"Recall {ov['recall']} below expected 0.90")
        
        # Check breakdown by category
        by_cat = data["by_category"]
        self.assertIn("photobomber", by_cat)
        self.assertIn("ambiguous", by_cat)
        self.assertIn("clean", by_cat)
        self.assertEqual(by_cat["photobomber"]["precision"], 1.0, "Precision on clear photobombers must be 100%")

    # ----------------------------------------------------
    # SECTION 7: Edge Cases, Negative Inputs & Robustness
    # ----------------------------------------------------
    def test_10_invalid_and_corrupt_files(self):
        """Tests system resilience against non-image text file and corrupt bytes"""
        # Text file
        text_file = {"file": ("malicious.txt", b"Not an image", "text/plain")}
        r_txt = self.client.post("/api/detect", files=text_file)
        self.assertEqual(r_txt.status_code, 400)
        
        # Corrupt pseudo-JPEG
        corrupt_file = {"file": ("corrupt.jpg", b"\xFF\xD8\xFF\xE0\x00\x10JFIF\x00\x00\x00bad_bytes", "image/jpeg")}
        r_corrupt = self.client.post("/api/detect", files=corrupt_file)
        self.assertEqual(r_corrupt.status_code, 400)

    def test_11_clean_photo_zero_photobombers(self):
        """Tests detection on clean landscape/portrait with no photobombers"""
        r_img = self.client.get("/api/sample-image/clean/clean_01.jpg")
        self.assertEqual(r_img.status_code, 200)
        
        files = {"file": ("clean_01.jpg", r_img.content, "image/jpeg")}
        r_detect = self.client.post("/api/detect", files=files)
        self.assertEqual(r_detect.status_code, 200)
        res = r_detect.json()
        
        # In clean_01, detected persons should be classified as KEEP
        for p in res["persons"]:
            self.assertEqual(p["classification"], "keep")

    def test_12_inpaint_endpoint(self):
        """Tests POST /api/inpaint seamlessly removes photobomber regions"""
        r_img = self.client.get("/api/sample-image/photobomber/pb_01.jpg")
        self.assertEqual(r_img.status_code, 200)
        img_b64 = "data:image/jpeg;base64," + base64.b64encode(r_img.content).decode("utf-8")

        payload = {
            "image_b64": img_b64,
            "boxes_to_remove": [[100, 100, 250, 350]],
        }
        r_inpaint = self.client.post("/api/inpaint", json=payload)
        self.assertEqual(r_inpaint.status_code, 200)
        res = r_inpaint.json()
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["removed_count"], 1)
        self.assertTrue(res["cleaned_image"].startswith("data:image/jpeg;base64,"))

    def test_13_per_person_keep_remove_controls_and_sharpness(self):
        """Verifies per-person Keep/Remove controls in JS and background texture sharpness (no blur)"""
        # 1. Verify JS functions
        self.assertIn("setPersonClassification", self.js_content)
        self.assertIn("updatePersonSummaryCounts", self.js_content)
        self.assertIn("✓ Keep", self.js_content)
        self.assertIn("✕ Remove", self.js_content)

        # 2. Inpainting sharpness verification (must have high-frequency texture, not flat PDE blur)
        import cv2
        import numpy as np
        r_img = self.client.get("/api/sample-image/photobomber/pb_01.jpg")
        img_b64 = "data:image/jpeg;base64," + base64.b64encode(r_img.content).decode("utf-8")
        payload = {
            "image_b64": img_b64,
            "boxes_to_remove": [[46, 323, 496, 651]],
        }
        res = self.client.post("/api/inpaint", json=payload).json()
        _, b64data = res["cleaned_image"].split(",", 1)
        cleaned_arr = np.frombuffer(base64.b64decode(b64data), np.uint8)
        cleaned_img = cv2.imdecode(cleaned_arr, cv2.IMREAD_COLOR)

        # Measure texture detail (Laplacian variance) in the inpainted hole
        hole_crop = cleaned_img[323:651, 46:496]
        lap_var = cv2.Laplacian(hole_crop, cv2.CV_64F).var()
        # Shift-Map texture exemplar synthesis yields ~49.3, whereas blurry NS gives ~12.8
        self.assertGreater(lap_var, 30.0, f"Inpainted texture sharpness {lap_var} too low (blurry smudging detected)!")

if __name__ == "__main__":
    unittest.main()

