"""
Confidence scoring from CTC output probabilities.
Flags low-confidence predictions for human review.
"""
import torch
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import CONFIDENCE_THRESHOLD, BLANK_LABEL


def compute_confidence(log_probs: torch.Tensor) -> float:
    """
    Compute confidence score from CTC log-probabilities.

    Uses the average of the max log-probability at each timestep,
    excluding blank tokens, converted to probability space.

    Args:
        log_probs: (seq_len, num_classes) — CTC log-probabilities for one sample.

    Returns:
        Confidence score between 0 and 1.
    """
    if log_probs is None or log_probs.numel() == 0:
        return 0.0

    probs = torch.exp(log_probs)  # Convert to probabilities
    max_probs, max_indices = probs.max(dim=1)

    # Filter out blank predictions
    non_blank_mask = max_indices != BLANK_LABEL
    if non_blank_mask.sum() == 0:
        return 0.0

    non_blank_probs = max_probs[non_blank_mask]
    return float(non_blank_probs.mean())


def needs_review(confidence: float, threshold: float = CONFIDENCE_THRESHOLD) -> bool:
    """Check if a prediction needs human review."""
    return confidence < threshold


def confidence_report(texts: list, confidences: list,
                      threshold: float = CONFIDENCE_THRESHOLD) -> dict:
    """
    Generate a confidence report for a batch of predictions.

    Returns:
        Dict with: predictions (list), avg_confidence, num_flagged, num_ok.
    """
    predictions = []
    for text, conf in zip(texts, confidences):
        predictions.append({
            "text": text,
            "confidence": round(conf, 4),
            "needs_review": needs_review(conf, threshold),
        })

    flagged = sum(1 for p in predictions if p["needs_review"])
    return {
        "predictions": predictions,
        "avg_confidence": round(np.mean(confidences), 4) if confidences else 0.0,
        "num_flagged": flagged,
        "num_ok": len(predictions) - flagged,
        "threshold": threshold,
    }


if __name__ == "__main__":
    # Quick test
    fake_log_probs = torch.randn(32, 94)  # 32 timesteps, 94 classes
    fake_log_probs = torch.nn.functional.log_softmax(fake_log_probs, dim=1)

    conf = compute_confidence(fake_log_probs)
    review = needs_review(conf)
    print(f"  Confidence: {conf:.4f}")
    print(f"  Needs review: {review}")
    print(f"  Threshold: {CONFIDENCE_THRESHOLD}")
