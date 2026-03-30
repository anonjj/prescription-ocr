"""
Download datasets from HuggingFace using the datasets library.
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import RAW_DIR, HUGGINGFACE_DATASETS  # type: ignore

# Dataset-specific column overrides
DATASET_COLUMN_MAP = {
    "medical_prescription": {"label_cols": ["label", "text", "word"]},
    "iam_word":             {"label_cols": ["text", "label"]},
    "iam_line":             {"label_cols": ["text", "label"]},
}


def download_hf_dataset(name: str, identifier: str, target_dir: str, dry_run: bool = False):
    """Download a HuggingFace dataset and save images + labels."""
    dest = os.path.join(target_dir, name)

    print(f"[HuggingFace] {'DRY-RUN ' if dry_run else ''}Downloading: {identifier}")
    print(f"              → {dest}")

    if dry_run:
        return

    os.makedirs(dest, exist_ok=True)
    images_dir = os.path.join(dest, "images")
    os.makedirs(images_dir, exist_ok=True)

    try:
        from datasets import load_dataset  # type: ignore
        import csv

        ds = load_dataset(identifier, trust_remote_code=True)

        labels_path = os.path.join(dest, "labels.csv")
        total = 0

        with open(labels_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["image_path", "text_label", "split"])

            for split_name in ds:
                split = ds[split_name]
                print(f"              Extracting split: {split_name} ({len(split)} samples)")
                
                # Import tqdm for progress tracking
                from tqdm import tqdm  # type: ignore
                
                for idx, sample in enumerate(tqdm(split, desc=f"              {split_name}")):
                    # Get column overrides if available
                    overrides = DATASET_COLUMN_MAP.get(name, {})
                    
                    # Try common column names for image
                    img = None
                    for col in ["image", "img", "pixel_values"]:
                        if col in sample:
                            img = sample[col]
                            break

                    # Try common column names for label
                    label = None
                    # Use overrides first
                    if "label_cols" in overrides:
                        for col in overrides["label_cols"]:
                            if col in sample:
                                label = sample[col]
                                break
                    
                    if label is None:
                        for col in ["text", "label", "transcription", "ground_truth", "word"]:
                            if col in sample:
                                label = sample[col]
                                break

                    if img is None or label is None:
                        continue

                    # Save image
                    img_filename = f"{split_name}_{idx:06d}.png"
                    img_path = os.path.join(images_dir, img_filename)

                    if hasattr(img, "save"):
                        img.save(img_path)  # type: ignore
                    elif isinstance(img, dict) and "bytes" in img:
                        from PIL import Image
                        import io
                        Image.open(io.BytesIO(img["bytes"])).convert("L").save(img_path)
                    else:
                        from PIL import Image  # type: ignore
                        Image.fromarray(img).save(img_path)

                    writer.writerow([os.path.join("images", img_filename), str(label), split_name])
                    total += 1

        print(f"              ✓ Done ({total} samples saved)")

    except Exception as e:
        import traceback
        print(f"              ✗ Error downloading {identifier}:")
        traceback.print_exc()
        # Not raising — allows other datasets to continue


def download_all_huggingface(dry_run: bool = False):
    """Download all configured HuggingFace datasets."""
    for name, identifier in HUGGINGFACE_DATASETS.items():
        download_hf_dataset(name, identifier, RAW_DIR, dry_run)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download HuggingFace datasets")
    parser.add_argument("--dry-run", action="store_true", help="Preview without downloading")
    args = parser.parse_args()
    download_all_huggingface(dry_run=args.dry_run)
