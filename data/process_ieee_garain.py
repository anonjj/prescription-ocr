"""
Helper script to extract the IEEE DataPort Garain et al. Dataset.
URL: https://ieee-dataport.org/documents/dataset-classification-handwritten-and-printed-text-doctors-prescription

Because IEEE DataPort strictly requires a user account and authentication 
to download datasets, this script cannot pull the files automatically.
Instead, it provides instructions to manually download the ZIP and then
handles extracting and formatting it so that `create_unified_manifest.py` 
can pick it up natively.
"""
import os
import sys
import zipfile
import shutil
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import RAW_DIR  # type: ignore

def verify_and_extract(zip_path: str):
    """Extracts IEEE Garain et al. dataset to RAW_DIR."""
    dataset_name = "ieee_garain_prescription"
    target_dir = os.path.join(RAW_DIR, dataset_name)
    
    if not os.path.exists(zip_path):
        print(f"Error: {zip_path} not found.")
        print("\n=== HOW TO DOWNLOAD ===")
        print("1. Create an account or log in at IEEE DataPort:")
        print("   https://ieee-dataport.org/documents/dataset-classification-handwritten-and-printed-text-doctors-prescription")
        print("2. Navigate to the dataset files tab.")
        print("3. Download the zipped dataset archive.")
        print(f"4. Move the downloaded zip file to: {zip_path}")
        print("5. Run this script again.")
        return

    print(f"Extracting {zip_path} to {target_dir}...")
    os.makedirs(target_dir, exist_ok=True)
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(target_dir)
            
        print(f"\n✓ Successfully extracted IEEE dataset to {target_dir}")
        print("✓ The `create_unified_manifest.py` script will automatically pick up")
        print("  the files in this folder based on their layout.")
        print("  Note: Make sure the extracted folders match the structure needed (folder=label)")
        print("  or that label CSVs are generated appropriately if missing.")
        
    except zipfile.BadZipFile:
        print("Error: The file provided is not a valid ZIP file.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract IEEE Garain et al. Dataset")
    parser.add_argument("--zip-path", type=str, default="/tmp/IEEE_Prescription.zip",
                        help="Path to the downloaded IEEE zip file")
    
    args = parser.parse_args()
    verify_and_extract(args.zip_path)
