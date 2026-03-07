"""
Utility functions for label encoding/decoding and metric computation.
"""
import editdistance
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import CHAR_TO_IDX, IDX_TO_CHAR, BLANK_LABEL


def encode_label(text: str) -> list:
    """
    Convert text string to list of integer indices.
    Unknown characters are skipped.
    """
    return [CHAR_TO_IDX[ch] for ch in text if ch in CHAR_TO_IDX]


def decode_prediction(indices: list) -> str:
    """
    CTC greedy decoding: collapse repeated characters and remove blanks.

    Args:
        indices: List of predicted class indices.

    Returns:
        Decoded text string.
    """
    chars = []
    prev = None
    for idx in indices:
        if idx != BLANK_LABEL and idx != prev:
            ch = IDX_TO_CHAR.get(idx, "")
            if ch:
                chars.append(ch)
        prev = idx
    return "".join(chars)


def compute_cer(predicted: str, target: str) -> float:
    """
    Character Error Rate = edit_distance(pred, target) / len(target).
    """
    if len(target) == 0:
        return 0.0 if len(predicted) == 0 else 1.0
    return editdistance.eval(predicted, target) / len(target)


def compute_wer(predicted: str, target: str) -> float:
    """
    Word Error Rate = edit_distance(pred_words, target_words) / len(target_words).
    """
    pred_words = predicted.split()
    target_words = target.split()
    if len(target_words) == 0:
        return 0.0 if len(pred_words) == 0 else 1.0
    return editdistance.eval(pred_words, target_words) / len(target_words)
