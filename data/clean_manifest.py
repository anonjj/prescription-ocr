"""
Clean the unified manifest by removing problematic samples.
"""
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import PROCESSED_DIR, CHARS

def clean_label(text, charset):
    """Strip characters that are not in the allowed charset."""
    return "".join(ch for ch in str(text) if ch in charset)

def main():
    charset = set(CHARS)
    manifest_path = os.path.join(PROCESSED_DIR, 'manifest.csv')
    
    if not os.path.exists(manifest_path):
        print(f"✗ manifest.csv not found at {manifest_path}. Run create_unified_manifest.py first.")
        return

    manifest = pd.read_csv(manifest_path)
    original_len = len(manifest)
    print(f"Original dataset: {original_len} samples")

    # Fix 1: Remove the garbage ocr-processed dataset entirely
    manifest = manifest[manifest['source_dataset'] != 'ocr-processed-handwritten-prescriptions']
    print(f"Removed ocr-processed: {original_len} → {len(manifest)}")

    # Fix 2: Strip bad-charset characters instead of dropping
    before = len(manifest)
    manifest['text_label'] = manifest['text_label'].apply(lambda x: clean_label(x, charset))
    # Drop rows that are now empty after stripping
    manifest = manifest[manifest['text_label'].str.len() >= 1]
    print(f"Stripped bad-charset chars: {before} → {len(manifest)} valid samples")

    # Fix 3: Remove empty labels (keep single-char labels as they are valid OCR samples)
    before = len(manifest)
    manifest = manifest[manifest['text_label'].astype(str).str.len() >= 1]
    print(f"Removed empty labels: {before} → {len(manifest)}")

    # Save cleaned manifest
    clean_path = os.path.join(PROCESSED_DIR, 'manifest_clean.csv')
    manifest.to_csv(clean_path, index=False)

    print(f"\nCleaned dataset statistics:")
    for src, grp in manifest.groupby('source_dataset'):
        median_len = grp['text_label'].astype(str).str.len().median()
        print(f"  {src}: {len(grp)} samples, median label len: {median_len:.0f}")

    print(f"\nTotal: {len(manifest)} samples")
    print(f"Saved → {clean_path}")

if __name__ == "__main__":
    main()
