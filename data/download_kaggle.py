"""
Download datasets from Kaggle using the Kaggle API.
Requires: ~/.kaggle/kaggle.json
"""
import os
import sys
import zipfile
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import RAW_DIR, KAGGLE_DATASETS


def download_kaggle_dataset(identifier: str, target_dir: str, dry_run: bool = False):
    """Download and extract a Kaggle dataset."""
    name = identifier.split("/")[-1]
    dest = os.path.join(target_dir, name)

    print(f"[Kaggle] {'DRY-RUN ' if dry_run else ''}Downloading: {identifier}")
    print(f"         → {dest}")

    if dry_run:
        return

    os.makedirs(dest, exist_ok=True)

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(identifier, path=dest, unzip=False)

        # Unzip
        for f in os.listdir(dest):
            if f.endswith(".zip"):
                zip_path = os.path.join(dest, f)
                print(f"         Extracting {f}...")
                with zipfile.ZipFile(zip_path, "r") as zf:
                    zf.extractall(dest)
                os.remove(zip_path)

        print(f"         ✓ Done ({len(os.listdir(dest))} items)")

    except Exception as e:
        print(f"         ✗ Error: {e}")
        print("         Make sure ~/.kaggle/kaggle.json exists and is valid.")
        raise


def download_all_kaggle(dry_run: bool = False):
    """Download all configured Kaggle datasets."""
    for key, identifier in KAGGLE_DATASETS.items():
        download_kaggle_dataset(identifier, RAW_DIR, dry_run)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Kaggle datasets")
    parser.add_argument("--dry-run", action="store_true", help="Preview without downloading")
    args = parser.parse_args()
    download_all_kaggle(dry_run=args.dry_run)
