"""
Create a unified manifest CSV from all downloaded datasets.
Schema: image_path, text_label, source_dataset, granularity, is_noisy_label
"""
import os
import sys
import csv
import re
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import RAW_DIR, PROCESSED_DIR

# Global class mapping for datasets with numeric IDs
CLASS_MAPPING = {}
mapping_file = os.path.join(RAW_DIR, "class_mapping.json")
if os.path.exists(mapping_file):
    try:
        with open(mapping_file, "r") as f:
            CLASS_MAPPING = json.load(f)
    except Exception:
        pass


def normalize_label(text: str) -> str:
    """Clean up and normalize a text label."""
    if not text:
        return ""
    # Strip whitespace
    text = text.strip()
    # Remove multiple spaces
    text = re.sub(r"\s+", " ", text)
    # Remove non-printable characters
    text = "".join(ch for ch in text if ch.isprintable())
    return text


def process_csv_dataset(dataset_dir: str, dataset_name: str,
                        granularity: str = "word",
                        is_noisy: bool = False) -> list:
    """Process a dataset that has labels.csv files."""
    rows = []
    csv_paths = []

    # Find all CSV files recursively
    for root, dirs, files in os.walk(dataset_dir):
        for f in files:
            if f.endswith(".csv"):
                csv_paths.append(os.path.join(root, f))

    for csv_path in csv_paths:
        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Try to get image path
                    img_path = None
                    for col in ["image_path", "filename", "image", "IMAGE", "file", "image_name", "image_file"]:
                        if col in row and row[col]:
                            img_path = row[col]
                            break

                    # Try to get label
                    label = None
                    for col in ["text_label", "text", "label", "MEDICINE_NAME", "transcription", "word", "ground_truth"]:
                        if col in row and row[col]:
                            label = row[col]
                            break

                    # Translate numeric labels if we have a mapping
                    if label and label.isdigit() and str(int(label)) in CLASS_MAPPING:
                        label = CLASS_MAPPING[str(int(label))]

                    if img_path and label:
                        # Resolve full path by searching both relative to CSV and recursively
                        full_img_path = None
                        
                        # Check relative to CSV file
                        p1 = os.path.join(os.path.dirname(csv_path), img_path)
                        # Check relative to dataset root
                        p2 = os.path.join(dataset_dir, img_path)
                        
                        if os.path.exists(p1):
                            full_img_path = p1
                        elif os.path.exists(p2):
                            full_img_path = p2
                        else:
                            # Search the whole dataset dir as fallback lookup
                            base_img = os.path.basename(img_path)
                            for r2, d2, f2 in os.walk(dataset_dir):
                                if base_img in f2:
                                    full_img_path = os.path.join(r2, base_img)
                                    break

                        if full_img_path and os.path.exists(full_img_path):
                            clean_label = normalize_label(label)
                            if clean_label:
                                rows.append({
                                    "image_path": full_img_path,
                                    "text_label": clean_label,
                                    "source_dataset": dataset_name,
                                    "granularity": granularity,
                                    "is_noisy_label": is_noisy,
                                })
        except Exception as e:
            print(f"    Warning: Could not process {csv_path}: {e}")

    return rows


def process_folder_dataset(dataset_dir: str, dataset_name: str,
                           granularity: str = "word") -> list:
    """Process a dataset organized as folders (folder name = label)."""
    rows = []

    for folder_name in sorted(os.listdir(dataset_dir)):
        folder_path = os.path.join(dataset_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue

        label = normalize_label(folder_name)
        if not label:
            continue

        for img_file in sorted(os.listdir(folder_path)):
            ext = os.path.splitext(img_file)[1].lower()
            if ext in (".png", ".jpg", ".jpeg", ".bmp", ".tiff"):
                rows.append({
                    "image_path": os.path.join(folder_path, img_file),
                    "text_label": label,
                    "source_dataset": dataset_name,
                    "granularity": granularity,
                    "is_noisy_label": False,
                })

    return rows


def create_manifest():
    """Build the unified manifest from all datasets."""
    print("=" * 60)
    print("  Creating Unified Manifest")
    print("=" * 60)

    all_rows = []

    if not os.path.exists(RAW_DIR):
        print(f"\n  RAW_DIR not found: {RAW_DIR}")
        return

    for dataset_name in sorted(os.listdir(RAW_DIR)):
        dataset_dir = os.path.join(RAW_DIR, dataset_name)
        if not os.path.isdir(dataset_dir):
            continue

        print(f"\n  Processing: {dataset_name}")

        # Determine dataset type and settings
        is_noisy = "ocr_processed" in dataset_name.lower()
        granularity = "line" if "iam_line" in dataset_name.lower() else "word"

        # Try CSV-based first
        rows = process_csv_dataset(dataset_dir, dataset_name, granularity, is_noisy)

        # If no CSV, try folder-based
        if not rows:
            rows = process_folder_dataset(dataset_dir, dataset_name, granularity)

        print(f"             → {len(rows)} samples")
        all_rows.extend(rows)

    # Save manifest
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    manifest_path = os.path.join(PROCESSED_DIR, "manifest.csv")

    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "text_label", "source_dataset",
                                                "granularity", "is_noisy_label"])
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\n  Total samples: {len(all_rows)}")
    print(f"  Manifest saved → {manifest_path}")
    print("=" * 60)


if __name__ == "__main__":
    create_manifest()
