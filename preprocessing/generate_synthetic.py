"""
Generate synthetic prescription text images using TRDG (TextRecognitionDataGenerator).

Produces handwriting-style images of drug names, dosages, and frequencies
for data augmentation during CRNN training.

Usage:
    python preprocessing/generate_synthetic.py                 # default 5000 samples
    python preprocessing/generate_synthetic.py --count 10000   # custom count

Output:
    {PROCESSED_DIR}/synthetic/        — image files
    {PROCESSED_DIR}/synthetic.csv     — manifest CSV
"""
import os
import sys
import csv
import random
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import PROCESSED_DIR, SYNTHETIC_COUNT, IMG_HEIGHT

# Drug names and medical terms for generating realistic prescription text
from postprocessing.lexicon import DRUG_NAMES, UNITS, FREQUENCIES


def _generate_prescription_text() -> str:
    """Generate a random realistic prescription text line."""
    templates = [
        # Drug + dosage
        lambda: f"{random.choice(DRUG_NAMES)} {random.choice(['100', '200', '250', '500', '650', '1000'])} {random.choice(['mg', 'ml', 'mcg'])}",
        # Drug + dosage + frequency
        lambda: f"{random.choice(DRUG_NAMES)} {random.choice(['250', '500'])} mg {random.choice(['OD', 'BD', 'TDS'])}",
        # Drug + schedule
        lambda: f"{random.choice(DRUG_NAMES)} {random.choice(['1-0-1', '1-1-1', '0-0-1', '1-0-0'])}",
        # Drug + dosage + duration
        lambda: f"{random.choice(DRUG_NAMES)} {random.choice(['250', '500'])} mg for {random.randint(3, 14)} days",
        # Drug only
        lambda: random.choice(DRUG_NAMES),
        # Dosage + unit
        lambda: f"{random.choice(['50', '100', '250', '500', '1000'])} {random.choice(['mg', 'ml', 'tabs', 'cap'])}",
        # Frequency terms
        lambda: random.choice([f for f in FREQUENCIES if len(f) > 2]),
        # Drug + timing
        lambda: f"{random.choice(DRUG_NAMES)} {random.choice(['before meals', 'after meals', 'at bedtime', 'morning', 'evening'])}",
    ]
    return random.choice(templates)()


def generate_synthetic_data(count: int = SYNTHETIC_COUNT, output_dir: str = None):
    """
    Generate synthetic prescription images using TRDG.

    Args:
        count: Number of synthetic samples to generate.
        output_dir: Directory to save images. Defaults to {PROCESSED_DIR}/synthetic/
    """
    try:
        from trdg.generators import GeneratorFromStrings
    except ImportError:
        print("  ✗ trdg not installed. Run: pip install trdg")
        print("    Falling back to OpenCV-based synthetic generation...")
        return _generate_with_opencv(count, output_dir)

    if output_dir is None:
        output_dir = os.path.join(PROCESSED_DIR, "synthetic")
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(PROCESSED_DIR, "synthetic.csv")

    # Generate text strings
    texts = [_generate_prescription_text() for _ in range(count)]

    print(f"  Generating {count} synthetic prescription images...")
    print(f"  Output: {output_dir}")

    # TRDG generator with handwriting-like settings
    generator = GeneratorFromStrings(
        strings=texts,
        count=count,
        fonts=[],            # use default fonts (includes handwriting)
        language="en",
        size=IMG_HEIGHT,
        skewing_angle=5,     # slight rotation
        random_skew=True,
        blur=1,
        random_blur=True,
        distorsion_type=2,   # sine wave distortion
        distorsion_orientation=0,
        background_type=1,   # gaussian noise background
        width=-1,            # auto width
        is_handwritten=True, # use handwriting synthesis
        fit=True,
    )

    rows = []
    for i, (img, lbl) in enumerate(generator):
        if img is None:
            continue

        img_filename = f"synth_{i:06d}.png"
        img_path = os.path.join(output_dir, img_filename)

        # Convert PIL to grayscale and save
        img_gray = img.convert("L")
        img_gray.save(img_path)

        rows.append({
            "image_path": img_path,
            "text_label": lbl,
            "source_dataset": "synthetic_trdg",
            "granularity": "line",
            "is_noisy_label": "False",
        })

        if (i + 1) % 500 == 0:
            print(f"    Generated {i + 1}/{count}...")

    # Write CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "image_path", "text_label", "source_dataset",
            "granularity", "is_noisy_label"
        ])
        writer.writeheader()
        writer.writerows(rows)

    print(f"  ✓ Generated {len(rows)} synthetic images")
    print(f"  ✓ Manifest saved to {csv_path}")
    return csv_path


def _generate_with_opencv(count: int, output_dir: str = None):
    """
    Fallback synthetic generator using OpenCV when TRDG is not available.
    Creates simple text-on-paper images with noise and slight transformations.
    """
    import cv2
    import numpy as np

    if output_dir is None:
        output_dir = os.path.join(PROCESSED_DIR, "synthetic")
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(PROCESSED_DIR, "synthetic.csv")

    # Available OpenCV fonts (Hershey fonts — not beautiful but functional)
    fonts = [
        cv2.FONT_HERSHEY_SIMPLEX,
        cv2.FONT_HERSHEY_COMPLEX,
        cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
        cv2.FONT_HERSHEY_SCRIPT_COMPLEX,
        cv2.FONT_ITALIC,
    ]

    rows = []
    for i in range(count):
        text = _generate_prescription_text()

        # Create image with slight random variations
        h = IMG_HEIGHT
        font = random.choice(fonts)
        font_scale = random.uniform(0.5, 1.0)
        thickness = random.choice([1, 2])

        # Calculate text size
        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        w = max(tw + 40, 200)

        # White background with slight noise
        img = np.ones((h, w), dtype=np.uint8) * random.randint(230, 255)
        noise = np.random.normal(0, random.uniform(3, 10), (h, w)).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Draw text with slight random offset
        x = random.randint(5, 20)
        y = h // 2 + th // 2 + random.randint(-5, 5)
        color = random.randint(0, 60)  # dark text
        cv2.putText(img, text, (x, y), font, font_scale, color, thickness)

        # Random slight rotation
        angle = random.uniform(-3, 3)
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h), borderValue=255)

        img_filename = f"synth_{i:06d}.png"
        img_path = os.path.join(output_dir, img_filename)
        cv2.imwrite(img_path, img)

        rows.append({
            "image_path": img_path,
            "text_label": text,
            "source_dataset": "synthetic_opencv",
            "granularity": "line",
            "is_noisy_label": "False",
        })

        if (i + 1) % 500 == 0:
            print(f"    Generated {i + 1}/{count}...")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "image_path", "text_label", "source_dataset",
            "granularity", "is_noisy_label"
        ])
        writer.writeheader()
        writer.writerows(rows)

    print(f"  ✓ Generated {len(rows)} synthetic images (OpenCV fallback)")
    print(f"  ✓ Manifest saved to {csv_path}")
    return csv_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic prescription images")
    parser.add_argument("--count", type=int, default=SYNTHETIC_COUNT,
                        help=f"Number of images to generate (default: {SYNTHETIC_COUNT})")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: {PROCESSED_DIR}/synthetic/)")
    args = parser.parse_args()

    generate_synthetic_data(count=args.count, output_dir=args.output_dir)
