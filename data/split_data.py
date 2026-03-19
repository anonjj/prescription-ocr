"""
Split the unified manifest into train / val / test CSVs.
Stratified by source_dataset to avoid data leakage.
"""
import os
import sys
import csv
import random
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import PROCESSED_DIR, TRAIN_RATIO, VAL_RATIO


def split_data(seed: int = 42):
    """Split manifest.csv into train.csv, val.csv, test.csv."""
    manifest_path = os.path.join(PROCESSED_DIR, "manifest_clean.csv")

    if not os.path.exists(manifest_path):
        print(f"  Manifest not found: {manifest_path}")
        print("  Run 'python data/create_unified_manifest.py' first.")
        return

    print("=" * 60)
    print("  Splitting Data")
    print("=" * 60)

    # Read all rows
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        all_rows = list(reader)

    # Group by source dataset for stratified split
    by_source = defaultdict(list)
    for row in all_rows:
        by_source[row["source_dataset"]].append(row)

    train_rows, val_rows, test_rows = [], [], []
    random.seed(seed)

    for source, rows in by_source.items():
        random.shuffle(rows)
        n = len(rows)
        n_train = int(n * TRAIN_RATIO)
        n_val = int(n * VAL_RATIO)

        train_rows.extend(rows[:n_train])
        val_rows.extend(rows[n_train:n_train + n_val])
        test_rows.extend(rows[n_train + n_val:])

        print(f"\n  {source}: {n} total → {n_train} train / {n_val} val / {n - n_train - n_val} test")

    # Shuffle within splits
    random.shuffle(train_rows)
    random.shuffle(val_rows)
    random.shuffle(test_rows)

    # Write splits
    for name, rows in [("train", train_rows), ("val", val_rows), ("test", test_rows)]:
        path = os.path.join(PROCESSED_DIR, f"{name}.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    print(f"\n  Total: {len(train_rows)} train / {len(val_rows)} val / {len(test_rows)} test")
    print(f"  Files saved → {PROCESSED_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    split_data()
