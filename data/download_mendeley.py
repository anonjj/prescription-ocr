"""
Helper script to download and extract the RxHandBD Mendeley Data dataset.
DOI: 10.17632/dsb5r6vskg.2

This dataset contains 5,578 cropped handwritten prescription words.
Because Mendeley Data uses dynamic JS-based download links, this script 
provides instructions to download the ZIP file and then handles the 
extraction into the correct RAW_DIR structure.
"""
import os
import sys
import zipfile
import shutil
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import RAW_DIR  # type: ignore

def verify_and_extract(zip_path: str):
    """Extracts RxHandBD to RAW_DIR."""
    dataset_name = "mendeley_rxhandbd"
    target_dir = os.path.join(RAW_DIR, dataset_name)
    
    if not os.path.exists(zip_path):
        print(f"Error: {zip_path} not found.")
        print("\n=== HOW TO DOWNLOAD ===")
        print("1. Go to: https://data.mendeley.com/datasets/dsb5r6vskg/2")
        print("2. Click 'Download All' (top right corner)")
        print("3. Wait for the ZIP file to finish downloading")
        print(f"4. Move the downloaded zip file to: {zip_path}")
        print("5. Run this script again.")
        return

    print(f"Extracting {zip_path} to {target_dir}...")
    os.makedirs(target_dir, exist_ok=True)
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(target_dir)
            
        print(f"\n✓ Successfully extracted RxHandBD dataset to {target_dir}")
        print("✓ The `create_unified_manifest.py` script will automatically pick up")
        print("  the `train_labels.csv` and `test_labels.csv` from this folder!")
        
    except zipfile.BadZipFile:
        print("Error: The file provided is not a valid ZIP file.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract Mendeley RxHandBD dataset")
    parser.add_argument("--zip-path", type=str, default="/tmp/RxHandBD.zip",
                        help="Path to the downloaded Mendeley zip file")
    
    args = parser.parse_args()
    verify_and_extract(args.zip_path)
