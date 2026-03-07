"""
HuggingFace baseline benchmark using a pretrained OCR model.
Compares against our CRNN model on the test set.
"""
import os
import sys
import argparse
import csv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import PROCESSED_DIR
from model.utils import compute_cer, compute_wer


def run_baseline(max_samples: int = 100):
    """Run a pretrained HuggingFace model as baseline."""
    try:
        from transformers import pipeline
        from PIL import Image
    except ImportError:
        print("  Install transformers: pip install transformers")
        return

    test_csv = os.path.join(PROCESSED_DIR, "test.csv")
    if not os.path.exists(test_csv):
        print(f"  Test data not found: {test_csv}")
        return

    # Load baseline model
    print("  Loading HuggingFace OCR model...")
    pipe = pipeline("image-to-text", model="microsoft/trocr-small-handwritten")

    # Read test data
    with open(test_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)[:max_samples]

    total_cer = 0.0
    total_wer = 0.0
    n = 0

    print(f"\n  Running baseline on {len(rows)} samples...")

    for row in rows:
        img_path = row["image_path"]
        target = row["text_label"]

        if not os.path.exists(img_path):
            continue

        try:
            img = Image.open(img_path).convert("RGB")
            result = pipe(img)
            predicted = result[0]["generated_text"] if result else ""

            cer = compute_cer(predicted, target)
            wer = compute_wer(predicted, target)
            total_cer += cer
            total_wer += wer
            n += 1

            if n <= 5:
                print(f"    '{target}' → '{predicted}' (CER: {cer:.4f})")

        except Exception as e:
            print(f"    Error on {img_path}: {e}")
            continue

    if n > 0:
        print(f"\n  Baseline Results ({n} samples):")
        print(f"    CER: {total_cer/n:.4f}")
        print(f"    WER: {total_wer/n:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run HuggingFace baseline")
    parser.add_argument("--max-samples", type=int, default=100)
    args = parser.parse_args()
    run_baseline(max_samples=args.max_samples)
