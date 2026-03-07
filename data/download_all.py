"""
Master download script — orchestrates Kaggle and HuggingFace downloads.
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def main():
    parser = argparse.ArgumentParser(description="Download all datasets")
    parser.add_argument("--dry-run", action="store_true", help="Preview without downloading")
    parser.add_argument("--hf-only", action="store_true", help="Download HuggingFace only")
    parser.add_argument("--kaggle-only", action="store_true", help="Download Kaggle only")
    args = parser.parse_args()

    print("=" * 60)
    print("  Doctor Handwriting OCR — Dataset Downloader")
    print("=" * 60)

    if not args.hf_only:
        print("\n── Kaggle Datasets ──")
        from data.download_kaggle import download_all_kaggle
        download_all_kaggle(dry_run=args.dry_run)

    if not args.kaggle_only:
        print("\n── HuggingFace Datasets ──")
        from data.download_huggingface import download_all_huggingface
        download_all_huggingface(dry_run=args.dry_run)

    print("\n" + "=" * 60)
    print("  All downloads complete!" if not args.dry_run else "  Dry run complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
