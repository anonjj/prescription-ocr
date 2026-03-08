"""
Evaluate a trained CRNN model on the test set.
Reports CER, WER, exact-match accuracy, and per-sample predictions.
"""
import os
import sys
import csv
import argparse
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import PROCESSED_DIR, CHECKPOINTS_DIR, BATCH_SIZE, NUM_WORKERS, BLANK_LABEL
from model.crnn import CRNN
from model.dataset import HandwritingDataset, collate_fn
from model.utils import decode_prediction, compute_cer, compute_wer


def evaluate(split: str = "test", checkpoint: str = None,
             save_predictions: bool = False, max_samples: int = None):
    """Evaluate model on a data split."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    csv_path = os.path.join(PROCESSED_DIR, f"{split}.csv")
    if not os.path.exists(csv_path):
        print(f"  Data not found: {csv_path}")
        return

    dataset = HandwritingDataset(csv_path, full_pipeline=False)
    if max_samples:
        dataset.samples = dataset.samples[:max_samples]

    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn, num_workers=NUM_WORKERS
    )

    # Load model
    ckpt_path = checkpoint or os.path.join(CHECKPOINTS_DIR, "best_model.pt")
    if not os.path.exists(ckpt_path):
        print(f"  Checkpoint not found: {ckpt_path}")
        return

    model = CRNN().to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"\n{'='*60}")
    print(f"  Evaluating on '{split}' set ({len(dataset)} samples)")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"{'='*60}")

    predictions = []
    total_cer = 0.0
    total_wer = 0.0
    exact_matches = 0
    n = 0

    with torch.no_grad():
        for images, labels, label_lengths, texts in loader:
            images = images.to(device)
            outputs = model(images)

            _, preds = outputs.max(2)
            preds = preds.permute(1, 0)

            for i in range(images.size(0)):
                pred_text = decode_prediction(preds[i].cpu().tolist())
                target_text = texts[i]

                cer = compute_cer(pred_text, target_text)
                wer = compute_wer(pred_text, target_text)

                total_cer += cer
                total_wer += wer
                if pred_text.strip().lower() == target_text.strip().lower():
                    exact_matches += 1
                n += 1

                predictions.append({
                    "target": target_text,
                    "predicted": pred_text,
                    "cer": f"{cer:.4f}",
                    "wer": f"{wer:.4f}",
                    "exact_match": pred_text.strip().lower() == target_text.strip().lower(),
                })

    avg_cer = total_cer / max(n, 1)
    avg_wer = total_wer / max(n, 1)
    accuracy = exact_matches / max(n, 1)

    print(f"\n  Results:")
    print(f"    CER:           {avg_cer:.4f}")
    print(f"    WER:           {avg_wer:.4f}")
    print(f"    Exact Match:   {accuracy:.2%} ({exact_matches}/{n})")

    # Show some examples
    print(f"\n  Sample Predictions:")
    for p in predictions[:10]:
        marker = "✓" if p["exact_match"] else "✗"
        print(f"    {marker} '{p['target']}' → '{p['predicted']}' (CER: {p['cer']})")

    # Save predictions
    if save_predictions:
        pred_path = os.path.join(PROCESSED_DIR, f"predictions_{split}.csv")
        with open(pred_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["target", "predicted", "cer", "wer", "exact_match"])
            writer.writeheader()
            writer.writerows(predictions)
        print(f"\n  Predictions saved → {pred_path}")

    print(f"{'='*60}")
    return {"cer": avg_cer, "wer": avg_wer, "accuracy": accuracy}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate CRNN model")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    evaluate(split=args.split, checkpoint=args.checkpoint,
             save_predictions=args.save_predictions, max_samples=args.max_samples)
