"""Script to download or generate the 15 benchmark test images.

Categories:
1. photobomber (5 images): Foreground subjects with clear background/peripheral photobomber
2. ambiguous   (5 images): Crowded social group scenes, party, or street scenes
3. clean       (5 images): Clean single/group portraits with no photobombers
"""

from __future__ import annotations
import os
import urllib.request
from pathlib import Path
import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "test_images"

# High-resolution, royalty-free curated Unsplash direct URLs (with CDN downscaling for fast inference)
TEST_IMAGE_URLS = {
    "photobomber": [
        # Photo 1: Beach couple with photobomber in background
        ("pb_01.jpg", "https://images.unsplash.com/photo-1517841905240-472988babdf9?w=800&q=80"),
        # Photo 2: Friends smiling with bystander peeking
        ("pb_02.jpg", "https://images.unsplash.com/photo-1529156069898-49953e39b3ac?w=800&q=80"),
        # Photo 3: Outdoor portrait with distant passerby
        ("pb_03.jpg", "https://images.unsplash.com/photo-1492562080023-ab3db95bfbce?w=800&q=80"),
        # Photo 4: Event photo with background attendee
        ("pb_04.jpg", "https://images.unsplash.com/photo-1511632765486-a01980e01a18?w=800&q=80"),
        # Photo 5: Vacation selfie with background photobomber
        ("pb_05.jpg", "https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=800&q=80"),
    ],
    "ambiguous": [
        # Ambiguous 1: Crowded dining table
        ("amb_01.jpg", "https://images.unsplash.com/photo-1543807535-eceef0bc6599?w=800&q=80"),
        # Ambiguous 2: Street festival crowd
        ("amb_02.jpg", "https://images.unsplash.com/photo-1514525253161-7a46d19cd819?w=800&q=80"),
        # Ambiguous 3: Team celebration
        ("amb_03.jpg", "https://images.unsplash.com/photo-1522071820081-009f0129c71c?w=800&q=80"),
        # Ambiguous 4: Audience at event
        ("amb_04.jpg", "https://images.unsplash.com/photo-1492684223066-81342ee5ff30?w=800&q=80"),
        # Ambiguous 5: Sports fans in stands
        ("amb_05.jpg", "https://images.unsplash.com/photo-1471295253337-3ceaaedca402?w=800&q=80"),
    ],
    "clean": [
        # Clean 1: Studio portrait
        ("clean_01.jpg", "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=800&q=80"),
        # Clean 2: Couple portrait
        ("clean_02.jpg", "https://images.unsplash.com/photo-1522529599102-193c0d76b5b6?w=800&q=80"),
        # Clean 3: Solo outdoor portrait
        ("clean_03.jpg", "https://images.unsplash.com/photo-1500648767791-00dcc994a43e?w=800&q=80"),
        # Clean 4: Business duo portrait
        ("clean_04.jpg", "https://images.unsplash.com/photo-1573496359142-b8d87734a5a2?w=800&q=80"),
        # Clean 5: Family portrait with clear focus
        ("clean_05.jpg", "https://images.unsplash.com/photo-1511895426328-dc8714191300?w=800&q=80"),
    ],
}


def create_synthetic_image(category: str, filename: str, target_path: Path) -> None:
    """Generate a realistic test image with visual silhouettes to enable offline testing."""
    h, w = 600, 800
    img = np.zeros((h, w, 3), dtype=np.uint8)

    # Gradient background
    for y in range(h):
        r = int(120 + 80 * (y / h))
        g = int(140 + 70 * (y / h))
        b = int(160 + 60 * (y / h))
        img[y, :] = [b, g, r]

    def draw_person_silhouette(
        canvas, center_x, ground_y, height, color=(40, 40, 40), blur_amount=0
    ):
        head_radius = int(height * 0.12)
        head_y = ground_y - height + head_radius
        body_top_y = head_y + head_radius
        body_w = int(height * 0.28)

        # Head
        cv2.circle(canvas, (center_x, head_y), head_radius, color, -1)
        # Torso & Legs
        pts = np.array(
            [
                [center_x - body_w // 2, body_top_y],
                [center_x + body_w // 2, body_top_y],
                [center_x + int(body_w * 0.6), ground_y],
                [center_x - int(body_w * 0.6), ground_y],
            ],
            np.int32,
        )
        cv2.fillPoly(canvas, [pts], color)

        if blur_amount > 1:
            x1 = max(0, center_x - body_w)
            x2 = min(canvas.shape[1], center_x + body_w)
            y1 = max(0, head_y - head_radius)
            y2 = min(canvas.shape[0], ground_y)
            sub = canvas[y1:y2, x1:x2]
            k = blur_amount if blur_amount % 2 == 1 else blur_amount + 1
            canvas[y1:y2, x1:x2] = cv2.GaussianBlur(sub, (k, k), 0)

    if category == "photobomber":
        # 1-2 Prominent centered subjects
        draw_person_silhouette(img, center_x=380, ground_y=550, height=380, color=(60, 40, 30))
        # 1 Photobomber in peripheral background (smaller, offset, blurred)
        draw_person_silhouette(
            img, center_x=710, ground_y=380, height=130, color=(100, 90, 80), blur_amount=9
        )
    elif category == "ambiguous":
        # Crowded scene with 4 people of varying prominence
        draw_person_silhouette(img, center_x=220, ground_y=520, height=320, color=(50, 40, 30))
        draw_person_silhouette(img, center_x=370, ground_y=530, height=330, color=(40, 50, 40))
        draw_person_silhouette(img, center_x=510, ground_y=510, height=300, color=(30, 40, 60))
        draw_person_silhouette(img, center_x=660, ground_y=450, height=220, color=(70, 70, 70), blur_amount=5)
    else:  # clean
        # Clear focused subject(s) centered in frame
        draw_person_silhouette(img, center_x=340, ground_y=550, height=400, color=(40, 40, 70))
        draw_person_silhouette(img, center_x=470, ground_y=540, height=380, color=(60, 30, 40))

    cv2.imwrite(str(target_path), img)


def download_or_generate_dataset() -> None:
    """Download images from Unsplash, or generate synthetic test images if offline."""
    print("Preparing 15 benchmark test images...")
    headers = {"User-Agent": "Mozilla/5.0"}

    for cat, items in TEST_IMAGE_URLS.items():
        cat_dir = DATA_DIR / cat
        cat_dir.mkdir(parents=True, exist_ok=True)

        for filename, url in items:
            out_file = cat_dir / filename
            if out_file.exists():
                continue

            success = False
            try:
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=5) as response, open(out_file, "wb") as f:
                    f.write(response.read())
                # Verify cv2 can read it
                test_read = cv2.imread(str(out_file))
                if test_read is not None and test_read.size > 0:
                    success = True
                    print(f"  [Downloaded] {cat}/{filename}")
            except Exception:
                success = False

            if not success:
                # Generate synthetic image for offline testing
                create_synthetic_image(cat, filename, out_file)
                print(f"  [Synthesized] {cat}/{filename} (offline fallback)")

    print("All 15 benchmark test images ready.\n")


if __name__ == "__main__":
    download_or_generate_dataset()
