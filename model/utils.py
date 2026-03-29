"""
Utility functions for label encoding/decoding and metric computation.

Supports:
  - Greedy CTC decoding (original)
  - Beam search decoding with medical language model (pyctcdecode)
"""
import editdistance
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    CHAR_TO_IDX, IDX_TO_CHAR, BLANK_LABEL, CHARS,
    USE_BEAM_SEARCH, BEAM_WIDTH, LM_ALPHA, LM_BETA
)

# Lazy-loaded beam search decoder (singleton)
_beam_decoder = None


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


def _get_beam_decoder():
    """
    Lazily create and cache the beam search decoder.

    Uses pyctcdecode with a medical ARPA language model if available.
    Falls back to beam search without LM if the ARPA file is missing.
    """
    global _beam_decoder
    if _beam_decoder is not None:
        return _beam_decoder

    try:
        from pyctcdecode import build_ctcdecoder
    except ImportError:
        print("  ⚠  pyctcdecode not installed — falling back to greedy decoding")
        print("     Install: pip install pyctcdecode")
        return None

    # Build vocabulary list: index 0 = blank (""), then CHARS in order
    # pyctcdecode expects labels[0] = "" (blank)
    labels = [""] + list(CHARS)

    # Try to load the medical language model
    lm_path = os.path.join(
        os.path.dirname(__file__), "..", "postprocessing", "medical_lm.arpa"
    )

    if os.path.exists(lm_path):
        try:
            _beam_decoder = build_ctcdecoder(
                labels=labels,
                kenlm_model_path=lm_path,
                alpha=LM_ALPHA,
                beta=LM_BETA,
            )
            print(f"  ✓ Beam search decoder initialised (LM: {os.path.basename(lm_path)})")
        except Exception as e:
            print(f"  ⚠  LM load failed ({e}) — using beam search without LM")
            _beam_decoder = build_ctcdecoder(labels=labels)
    else:
        print(f"  ⚠  LM not found at {lm_path} — using beam search without LM")
        print(f"     Generate it: python postprocessing/build_lm.py")
        _beam_decoder = build_ctcdecoder(labels=labels)

    return _beam_decoder


def decode_prediction_beam(log_probs, beam_width: int = BEAM_WIDTH) -> str:
    """
    CTC beam search decoding with optional language model.

    Args:
        log_probs: numpy array of shape (seq_len, num_classes) — log probabilities.
                   Can also be a torch.Tensor (will be converted).
        beam_width: Number of beams (default from config).

    Returns:
        Decoded text string. Falls back to greedy if pyctcdecode unavailable.
    """
    import numpy as np

    # Convert torch tensor to numpy if needed
    if hasattr(log_probs, 'cpu'):
        log_probs = log_probs.cpu().numpy()

    if not isinstance(log_probs, np.ndarray):
        log_probs = np.array(log_probs)

    decoder = _get_beam_decoder()
    if decoder is None:
        # Fallback: greedy decode from log_probs
        indices = log_probs.argmax(axis=1).tolist()
        return decode_prediction(indices)

    # pyctcdecode expects log-probabilities (which we already have)
    text = decoder.decode(log_probs, beam_width=beam_width)
    return text


def smart_decode(log_probs=None, indices=None, use_beam: bool = USE_BEAM_SEARCH) -> str:
    """
    Convenience function: beam search if enabled, greedy otherwise.

    Provide either log_probs (for beam) or indices (for greedy).
    If use_beam=True and log_probs is provided, uses beam search.
    """
    if use_beam and log_probs is not None:
        return decode_prediction_beam(log_probs)
    elif indices is not None:
        return decode_prediction(indices)
    elif log_probs is not None:
        # Beam not requested or not available — greedy from log_probs
        import numpy as np
        if hasattr(log_probs, 'cpu'):
            log_probs = log_probs.cpu().numpy()
        idxs = log_probs.argmax(axis=1).tolist()
        return decode_prediction(idxs)
    else:
        return ""


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
