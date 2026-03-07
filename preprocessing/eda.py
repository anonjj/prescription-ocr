"""
Exploratory Data Analysis — visualize dataset samples and distributions.
"""
import os
import sys
import argparse
import csv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import PROCESSED_DIR

try:
    import matplotlib.pyplot as plt
    import cv2
    import numpy as np
    from preprocessing.transforms import preprocess_image
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False


def show_samples(manifest_path: str, n: int = 5):
    """Show raw vs preprocessed image comparisons."""
    if not HAS_DEPS:
        print("Install matplotlib and opencv-python for EDA visualizations.")
        return

    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    import random
    random.shuffle(rows)
    samples = rows[:n]

    fig, axes = plt.subplots(n, 2, figsize=(12, 3 * n))
    if n == 1:
        axes = [axes]

    for i, sample in enumerate(samples):
        img_path = sample["image_path"]
        label = sample["text_label"]

        if not os.path.exists(img_path):
            continue

        # Raw image
        raw = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if raw is None:
            continue

        axes[i][0].imshow(raw, cmap="gray")
        axes[i][0].set_title(f"Raw: '{label}'", fontsize=10)
        axes[i][0].axis("off")

        # Preprocessed
        try:
            processed = preprocess_image(img_path)
            axes[i][1].imshow(processed, cmap="gray")
            axes[i][1].set_title(f"Processed: '{label}'", fontsize=10)
            axes[i][1].axis("off")
        except Exception as e:
            axes[i][1].set_title(f"Error: {e}", fontsize=8)
            axes[i][1].axis("off")

    plt.tight_layout()
    save_path = os.path.join(PROCESSED_DIR, "eda_samples.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Samples saved → {save_path}")


def show_label_distribution(manifest_path: str):
    """Plot label length distribution."""
    if not HAS_DEPS:
        return

    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        lengths = [len(row["text_label"]) for row in reader]

    plt.figure(figsize=(10, 4))
    plt.hist(lengths, bins=50, color="steelblue", edgecolor="white")
    plt.xlabel("Label Length (characters)")
    plt.ylabel("Count")
    plt.title("Label Length Distribution")
    plt.tight_layout()

    save_path = os.path.join(PROCESSED_DIR, "eda_label_dist.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Label distribution saved → {save_path}")


def main():
    parser = argparse.ArgumentParser(description="EDA visualizations")
    parser.add_argument("--sample", type=int, default=5, help="Number of samples to show")
    args = parser.parse_args()

    manifest_path = os.path.join(PROCESSED_DIR, "manifest.csv")
    if not os.path.exists(manifest_path):
        print(f"  Manifest not found: {manifest_path}")
        print("  Run 'python data/create_unified_manifest.py' first.")
        return

    show_samples(manifest_path, n=args.sample)
    show_label_distribution(manifest_path)


if __name__ == "__main__":
    main()
