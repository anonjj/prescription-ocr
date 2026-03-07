"""
Audit downloaded datasets — report sample counts, label types, image stats.
"""
import os
import sys
import csv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import RAW_DIR

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


def audit_directory(name: str, path: str) -> dict:
    """Audit a single dataset directory."""
    info = {
        "name": name,
        "path": path,
        "exists": os.path.exists(path),
        "num_images": 0,
        "num_labels": 0,
        "image_formats": set(),
        "avg_width": 0,
        "avg_height": 0,
        "has_csv": False,
        "label_source": "unknown",
    }

    if not info["exists"]:
        return info

    # Count images and check formats
    widths, heights = [], []
    for root, dirs, files in os.walk(path):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in (".png", ".jpg", ".jpeg", ".bmp", ".tiff"):
                info["num_images"] += 1
                info["image_formats"].add(ext)

                if HAS_PIL and len(widths) < 200:  # sample up to 200
                    try:
                        img = Image.open(os.path.join(root, f))
                        widths.append(img.width)
                        heights.append(img.height)
                    except Exception:
                        pass

            elif ext == ".csv":
                info["has_csv"] = True
                info["label_source"] = "csv"
                # Count label rows
                try:
                    with open(os.path.join(root, f), "r", encoding="utf-8") as csvfile:
                        reader = csv.reader(csvfile)
                        info["num_labels"] = max(0, sum(1 for _ in reader) - 1)
                except Exception:
                    pass

    if widths:
        info["avg_width"] = sum(widths) / len(widths)
        info["avg_height"] = sum(heights) / len(heights)

    info["image_formats"] = ", ".join(sorted(info["image_formats"])) if info["image_formats"] else "none"

    return info


def run_audit():
    """Audit all datasets in RAW_DIR."""
    print("=" * 70)
    print("  Dataset Audit Report")
    print("=" * 70)

    if not os.path.exists(RAW_DIR):
        print(f"\n  RAW_DIR not found: {RAW_DIR}")
        print("  Run 'python data/download_all.py' first.")
        return []

    results = []
    for entry in sorted(os.listdir(RAW_DIR)):
        full_path = os.path.join(RAW_DIR, entry)
        if os.path.isdir(full_path):
            info = audit_directory(entry, full_path)
            results.append(info)

            print(f"\n  📂 {info['name']}")
            print(f"     Images:      {info['num_images']}")
            print(f"     Labels:      {info['num_labels']}")
            print(f"     Formats:     {info['image_formats']}")
            print(f"     Avg size:    {info['avg_width']:.0f} × {info['avg_height']:.0f}")
            print(f"     Label src:   {info['label_source']}")

    if not results:
        print("\n  No datasets found in RAW_DIR.")
        print("  Run 'python data/download_all.py' first.")

    # Save audit report CSV
    report_path = os.path.join(os.path.dirname(RAW_DIR), "audit_report.csv")
    with open(report_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "num_images", "num_labels",
                                                "image_formats", "avg_width", "avg_height",
                                                "label_source"])
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in writer.fieldnames})

    print(f"\n  Report saved → {report_path}")
    print("=" * 70)

    return results


if __name__ == "__main__":
    run_audit()
