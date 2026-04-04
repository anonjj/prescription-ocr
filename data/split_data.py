"""
Split the unified manifest into train / val / test CSVs.
Stratified by source_dataset to avoid data leakage.
"""
import os
import sys
import csv
import random
import argparse
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import PROCESSED_DIR, TRAIN_RATIO, VAL_RATIO

FINETUNE_SOURCES = [
    "rxhandbd",
    "prescription_bd",
]


def _load_manifest(manifest_name: str = "manifest_clean.csv"):
    """Load the cleaned manifest and return fieldnames + rows."""
    manifest_path = os.path.join(PROCESSED_DIR, manifest_name)

    if not os.path.exists(manifest_path):
        print(f"  Manifest not found: {manifest_path}")
        print("  Run 'python data/create_unified_manifest.py' first.")
        return None, None

    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames, list(reader)


def _write_split_csvs(fieldnames, split_rows, prefix: str = ""):
    """Write split CSVs using an optional filename prefix."""
    for name, rows in split_rows.items():
        filename = f"{prefix}_{name}.csv" if prefix else f"{name}.csv"
        path = os.path.join(PROCESSED_DIR, filename)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def _split_rows(all_rows, seed: int):
    """Split rows into train/val/test, stratified by source dataset."""
    by_source = defaultdict(list)
    for row in all_rows:
        by_source[row["source_dataset"]].append(row)

    train_rows, val_rows, test_rows = [], [], []
    random.seed(seed)

    for source, rows in sorted(by_source.items()):
        random.shuffle(rows)
        n = len(rows)
        n_train = int(n * TRAIN_RATIO)
        n_val = int(n * VAL_RATIO)

        train_rows.extend(rows[:n_train])
        val_rows.extend(rows[n_train:n_train + n_val])
        test_rows.extend(rows[n_train + n_val:])

        print(f"\n  {source}: {n} total → {n_train} train / {n_val} val / {n - n_train - n_val} test")

    random.shuffle(train_rows)
    random.shuffle(val_rows)
    random.shuffle(test_rows)

    return {
        "train": train_rows,
        "val": val_rows,
        "test": test_rows,
    }


def split_data(seed: int = 42):
    """Split manifest.csv into train.csv, val.csv, test.csv."""
    fieldnames, all_rows = _load_manifest("manifest_clean.csv")
    if all_rows is None:
        return

    print("=" * 60)
    print("  Splitting Data")
    print("=" * 60)
    split_rows = _split_rows(all_rows, seed=seed)
    _write_split_csvs(fieldnames, split_rows, prefix="")

    print(f"\n  Total: {len(split_rows['train'])} train / {len(split_rows['val'])} val / {len(split_rows['test'])} test")
    print(f"  Files saved → {PROCESSED_DIR}")
    print("=" * 60)


def split_finetune(seed: int = 42, sources=None):
    """Create prescription-only train/val/test CSVs for fine-tuning."""
    fieldnames, all_rows = _load_manifest("manifest_clean.csv")
    if all_rows is None:
        return

    sources = sources or FINETUNE_SOURCES
    filtered_rows = [row for row in all_rows if row["source_dataset"] in set(sources)]

    print("=" * 60)
    print("  Splitting Fine-Tune Data")
    print("=" * 60)
    print(f"  Sources: {', '.join(sources)}")
    print(f"  Fine-tune dataset: {len(filtered_rows)} samples")

    if not filtered_rows:
        print("  No matching prescription samples found.")
        return

    split_rows = _split_rows(filtered_rows, seed=seed)
    _write_split_csvs(fieldnames, split_rows, prefix="finetune")

    print(f"\n  Total: {len(split_rows['train'])} train / {len(split_rows['val'])} val / {len(split_rows['test'])} test")
    print(f"  Files saved → {PROCESSED_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create train/val/test splits")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")
    parser.add_argument("--finetune", action="store_true",
                        help="Create prescription-only fine-tune splits")
    args = parser.parse_args()

    manifest_path = os.path.join(PROCESSED_DIR, "manifest_clean.csv")
    if not os.path.exists(manifest_path):
        print(f"  Manifest not found: {manifest_path}")
        print("  Run 'python data/create_unified_manifest.py' first.")
        raise SystemExit(0)

    split_data(seed=args.seed)
    if args.finetune:
        split_finetune(seed=args.seed)
