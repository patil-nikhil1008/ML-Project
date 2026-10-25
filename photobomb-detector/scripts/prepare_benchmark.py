"""Benchmark Dataset Builder

Builds 15 benchmark images meeting the assignment requirements:
- 5 photobomber images (clear photobomber behind/beside foreground subject)
- 5 ambiguous/crowded images (3-5 persons at varying positions/depths)
- 5 clean negative images (solo/duo portraits with 0 photobombers)
"""

from __future__ import annotations
import os
import sys
from pathlib import Path
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DATA_DIR = PROJECT_ROOT / "data" / "test_images"


def prepare_benchmark_images() -> None:
    for cat in ["photobomber", "ambiguous", "clean"]:
        (DATA_DIR / cat).mkdir(parents=True, exist_ok=True)

    # Use existing clean portraits as bases
    clean_bases = [
        cv2.imread(str(DATA_DIR / "clean" / f"clean_{i:02d}.jpg")) for i in range(1, 5)
    ]
    # Filter valid bases
    clean_bases = [img for img in clean_bases if img is not None]

    if not clean_bases:
        print("Error: clean base portraits not found.")
        return

    # Extract human crops to use as foreground subjects and photobombers
    from src.detector import detect_persons

    donor_crops = []
    for img in clean_bases:
        dets = detect_persons(img)
        if dets:
            x1, y1, x2, y2 = dets[0]["bounding_box"]
            crop = img[y1:y2, x1:x2]
            if crop.size > 0:
                donor_crops.append(crop)

    # 1. BUILD CLEAN IMAGES (5 images, clean portraits, no photobombers)
    # clean_01 to clean_04 already exist as clean solo portraits
    # clean_05: Duo portrait (two friends side-by-side, both KEEP)
    base_clean = clean_bases[0].copy()
    h, w = base_clean.shape[:2]
    # Resize donor crop to match subject scale
    donor = cv2.resize(donor_crops[1], (int(w * 0.45), int(h * 0.7)))
    # Place side-by-side
    duo_canvas = np.zeros((h, int(w * 1.5), 3), dtype=np.uint8)
    # Background texture
    duo_canvas[:] = base_clean[10, 10]
    # Subject 1 (left)
    sub1 = cv2.resize(clean_bases[0], (int(w * 0.75), h))
    duo_canvas[:, : sub1.shape[1]] = sub1
    # Subject 2 (right)
    sub2 = cv2.resize(clean_bases[1], (int(w * 0.75), h))
    duo_canvas[:, sub1.shape[1] :] = sub2
    cv2.imwrite(str(DATA_DIR / "clean" / "clean_05.jpg"), duo_canvas)

    # 2. BUILD PHOTOBOMBER IMAGES (5 images with 1-2 foreground subjects + 1 photobomber)
    photobomb_configs = [
        # (base_img_idx, pb_crop_idx, pb_pos, pb_scale, blur_k)
        (0, 1, (40, 320), 0.28, 7),   # pb_01: bottom-left background lurker
        (1, 0, (580, 200), 0.25, 9),  # pb_02: top-right corner photobomber
        (2, 1, (60, 250), 0.30, 5),   # pb_03: mid-left background passerby
        (0, 2, (550, 350), 0.22, 11), # pb_04: far-right blurred photobomber
        (1, 2, (50, 180), 0.26, 7),   # pb_05: top-left peeking photobomber
    ]

    for idx, (b_idx, p_idx, (px, py), scale, blur_k) in enumerate(photobomb_configs, 1):
        bg = clean_bases[b_idx % len(clean_bases)].copy()
        pb_donor = donor_crops[p_idx % len(donor_crops)]

        bh, bw = bg.shape[:2]
        new_h = int(bh * scale)
        new_w = max(10, int(new_h * (pb_donor.shape[1] / pb_donor.shape[0])))

        pb_resized = cv2.resize(pb_donor, (new_w, new_h))
        if blur_k > 1:
            pb_resized = cv2.GaussianBlur(pb_resized, (blur_k, blur_k), 0)

        # Ensure placement stays inside bounds
        px = max(0, min(px, bw - new_w - 5))
        py = max(0, min(py, bh - new_h - 5))

        # Alpha blend edges for natural integration
        mask = np.ones((new_h, new_w), dtype=np.float32)
        cv2.rectangle(mask, (0, 0), (new_w, new_h), 1.0, -1)
        mask = cv2.GaussianBlur(mask, (15, 15), 0)[:, :, np.newaxis]

        roi = bg[py : py + new_h, px : px + new_w].astype(np.float32)
        blended = (
            pb_resized.astype(np.float32) * mask + roi * (1.0 - mask)
        ).astype(np.uint8)
        bg[py : py + new_h, px : px + new_w] = blended

        out_path = DATA_DIR / "photobomber" / f"pb_{idx:02d}.jpg"
        cv2.imwrite(str(out_path), bg)

    # 3. BUILD AMBIGUOUS/CROWDED IMAGES (5 images with 3-4 persons at varying prominence)
    # amb_01, amb_02, amb_03 already have 3-5 persons.
    # Let's create amb_04 and amb_05 with 3-4 people
    for idx, amb_num in enumerate([4, 5]):
        canvas = np.zeros((600, 900, 3), dtype=np.uint8)
        canvas[:] = (200, 210, 220)  # neutral background

        # Place 3 people: Center foreground (keep), Right midground (keep), Left edge (remove)
        p1 = cv2.resize(donor_crops[0], (260, 480))
        p2 = cv2.resize(donor_crops[1 % len(donor_crops)], (220, 420))
        p3 = cv2.resize(donor_crops[2 % len(donor_crops)], (140, 260))
        p3 = cv2.GaussianBlur(p3, (7, 7), 0)

        # Place p3 in background
        canvas[150:410, 80:220] = p3
        # Place p2 midground
        canvas[120:540, 560:780] = p2
        # Place p1 foreground center
        canvas[90:570, 310:570] = p1

        out_path = DATA_DIR / "ambiguous" / f"amb_{amb_num:02d}.jpg"
        cv2.imwrite(str(out_path), canvas)

    print("Successfully built and balanced all 15 benchmark images!")


if __name__ == "__main__":
    prepare_benchmark_images()
