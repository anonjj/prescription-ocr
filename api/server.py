"""
Flask REST API for Doctor Handwriting OCR.

Exposes a single POST endpoint that accepts an image and returns
the recognised text, confidence score, and any extracted prescription fields.

Usage:
  python api/server.py              # starts on http://0.0.0.0:5000
  python api/server.py --port 8080

Android (or any HTTP client) should POST the image to:
  http://<server-ip>:5000/predict
"""

import os
import sys
import tempfile
import argparse

from flask import Flask, request, jsonify

# Add project root so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from config import CHECKPOINTS_DIR
from model.crnn import CRNN
from model.utils import decode_prediction
from preprocessing.transforms import preprocess_image
from postprocessing.lexicon import correct_prescription_text
from postprocessing.rules import extract_all
from postprocessing.confidence import compute_confidence, needs_review

app = Flask(__name__)

# ── Global model (loaded once at startup) ──────────────────────────────────
_model = None
_device = None


def load_model():
    """Load the trained CRNN checkpoint into memory."""
    global _model, _device
    _device = torch.device("cpu")  # CPU inference is fine for the API

    ckpt_path = os.path.join(CHECKPOINTS_DIR, "best_model.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found at {ckpt_path}. "
            "Train the model first (run train.py on Colab)."
        )

    _model = CRNN().to(_device)
    ckpt = torch.load(ckpt_path, map_location=_device)
    _model.load_state_dict(ckpt["model_state_dict"])
    _model.eval()

    epoch = ckpt.get("epoch", "?")
    cer = ckpt.get("best_cer", "?")
    print(f"  Model loaded — epoch {epoch}, best CER {cer}")


def run_inference(image_path: str) -> dict:
    """
    Run the full OCR pipeline on a saved image file.

    Steps:
      1. Preprocess (binarise, resize, normalise)
      2. CRNN forward pass
      3. CTC greedy decode
      4. Fuzzy-correct prescription words
      5. Extract dosage / frequency / duration

    Returns a dict with keys: raw_text, corrected_text, confidence,
    needs_review, dosage, frequency, duration.
    """
    # 1. Preprocess
    img = preprocess_image(image_path, full_pipeline=True)

    # 2. Convert to (1, 1, H, W) tensor
    img_tensor = torch.FloatTensor(img).unsqueeze(0).unsqueeze(0) / 255.0
    img_tensor = img_tensor.to(_device)

    # 3. Forward pass (no gradient needed)
    with torch.no_grad():
        log_probs = _model(img_tensor)  # (seq_len, 1, num_classes)

    # 4. Greedy decode
    _, preds = log_probs.max(2)
    pred_indices = preds.squeeze(1).cpu().tolist()
    raw_text = decode_prediction(pred_indices)

    # 5. Confidence score
    sample_log_probs = log_probs.squeeze(1)
    confidence = compute_confidence(sample_log_probs)
    review_flag = needs_review(confidence)

    # 6. Post-process: fuzzy correction + field extraction
    words = raw_text.split()
    corrections = correct_prescription_text(words) if words else []
    corrected_text = " ".join(c["corrected"] for c in corrections)
    extracted = extract_all(corrected_text)

    return {
        "raw_text": raw_text,
        "corrected_text": corrected_text,
        "confidence": round(confidence, 4),
        "needs_review": review_flag,
        "dosage": extracted.get("dosage", []),
        "frequency": extracted.get("frequency", []),
        "duration": extracted.get("duration", []),
    }


# ── Routes ──────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    """Simple health-check so the Android app can verify connectivity."""
    return jsonify({"status": "ok", "model_loaded": _model is not None})


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accept an uploaded image and return OCR results as JSON.

    Expects: multipart/form-data with field name "image".
    Returns:
      {
        "raw_text":       str,
        "corrected_text": str,
        "confidence":     float,   # 0.0 – 1.0
        "needs_review":   bool,
        "dosage":         [str],
        "frequency":      [str],
        "duration":       [str]
      }
    """
    if "image" not in request.files:
        return jsonify({"error": "No image file in request (field name: 'image')"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # Save to a temp file so preprocess_image can read it
    suffix = os.path.splitext(file.filename)[1] or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = tmp.name
        file.save(tmp_path)

    try:
        result = run_inference(tmp_path)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        os.unlink(tmp_path)  # always clean up the temp file


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OCR REST API server")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Bind address. Use 0.0.0.0 to accept connections from Android.")
    args = parser.parse_args()

    print("Loading model...")
    load_model()
    print(f"Starting server on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)
