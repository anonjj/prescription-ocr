"""
Run OCR line-by-line on a prescription image using model.ptl (TorchScript).
Uses CRAFT text detector for tight bounding boxes when available,
falls back to morphological detection otherwise.

Usage: python3 run_line_ocr.py <image_path>
"""
import os
import sys
import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import IMG_HEIGHT, IMG_WIDTH, IDX_TO_CHAR, BLANK_LABEL
from postprocessing.confidence import compute_confidence, needs_review
from postprocessing.lexicon import correct_prescription_text

# CRAFT detector — optional dependency
try:
    # craft-text-detector references torchvision.models.vgg.model_urls which was
    # removed in torchvision >=0.13. Patch it back before importing.
    import torchvision.models.vgg as _vgg
    if not hasattr(_vgg, "model_urls"):
        _vgg.model_urls = {
            "vgg16_bn": "https://download.pytorch.org/models/vgg16_bn-6c64b313.pth",
        }
    from craft_text_detector import Craft
    CRAFT_AVAILABLE = True
except ImportError:
    CRAFT_AVAILABLE = False

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "model.ptl")


# ── Model Loading ──────────────────────────────────────────────────────────────

def load_model():
    if not os.path.exists(MODEL_PATH):
        print(f"  ✗ model.ptl not found at {MODEL_PATH}")
        sys.exit(1)
    model = torch.jit.load(MODEL_PATH, map_location="cpu")
    model.eval()
    detector = "CRAFT" if CRAFT_AVAILABLE else "morphological (install craft-text-detector for better results)"
    print(f"  ✓ Loaded model.ptl  |  text detector: {detector}")
    return model


# ── CTC Decoding ───────────────────────────────────────────────────────────────

def decode_ctc(log_probs_tensor):
    """Greedy CTC decode from (seq_len, num_classes) log probs."""
    indices = log_probs_tensor.argmax(dim=1).tolist()
    chars, prev = [], None
    for idx in indices:
        if idx != BLANK_LABEL and idx != prev:
            chars.append(IDX_TO_CHAR.get(idx, ""))
        prev = idx
    return "".join(chars)


# ── Image Preprocessing ────────────────────────────────────────────────────────

def preprocess_crop(crop_bgr):
    """Apply the same pipeline as preprocessing/transforms.py to a BGR crop."""
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    h, w = gray.shape
    ratio = IMG_HEIGHT / h
    new_w = min(int(w * ratio), IMG_WIDTH)
    resized = cv2.resize(gray, (new_w, IMG_HEIGHT), interpolation=cv2.INTER_AREA)
    padded = np.ones((IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8) * 255
    padded[:, :new_w] = resized
    return padded


# ── Text Detection — CRAFT ─────────────────────────────────────────────────────

def detect_lines_craft(image_bgr):
    """
    Detect text regions with CRAFT, group into reading-order line crops.
    Returns list of (x1, y1, x2, y2) tight bounding boxes.
    """
    craft = Craft(output_dir=None, crop_type="box", cuda=False)
    result = craft.detect_text(image_bgr)
    craft.unload_craftnet_model()
    craft.unload_refinenet_model()

    boxes = result.get("boxes", [])
    if not boxes:
        return []

    img_h, img_w = image_bgr.shape[:2]
    rects = []
    for box in boxes:
        pts = np.array(box, dtype=np.float32)
        x1, y1 = int(pts[:, 0].min()), int(pts[:, 1].min())
        x2, y2 = int(pts[:, 0].max()), int(pts[:, 1].max())
        rects.append((x1, y1, x2, y2))

    # Group boxes into lines by vertical proximity
    rects.sort(key=lambda r: (r[1] + r[3]) / 2)  # sort by y-center
    heights = [r[3] - r[1] for r in rects]
    median_h = float(np.median(heights)) if heights else 30
    line_gap = median_h * 0.6

    lines = []   # each line = list of rects
    for rect in rects:
        y_center = (rect[1] + rect[3]) / 2
        placed = False
        for line in lines:
            line_y = np.mean([(r[1] + r[3]) / 2 for r in line])
            if abs(y_center - line_y) < line_gap:
                line.append(rect)
                placed = True
                break
        if not placed:
            lines.append([rect])

    # Merge each line group into one bounding rect
    merged = []
    for line in lines:
        x1 = max(0, min(r[0] for r in line) - 4)
        y1 = max(0, min(r[1] for r in line) - 4)
        x2 = min(img_w, max(r[2] for r in line) + 4)
        y2 = min(img_h, max(r[3] for r in line) + 4)
        merged.append((x1, y1, x2, y2))

    merged.sort(key=lambda r: r[1])  # top-to-bottom
    return merged


# ── Text Detection — Morphological fallback ────────────────────────────────────

def detect_lines_morphological(image_bgr, min_line_height=20, padding=8):
    """
    Fallback line detector using morphological operations.
    Returns list of (x1, y1, x2, y2).
    """
    img_h, img_w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 8
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 2))
    dilated = cv2.dilate(binary, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if h >= min_line_height and w > img_w * 0.05:
            boxes.append((x, y, x + w, y + h))

    boxes.sort(key=lambda b: b[1])
    merged = []
    for x1, y1, x2, y2 in boxes:
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(img_w, x2 + padding)
        y2 = min(img_h, y2 + padding)
        if merged and y1 < merged[-1][3]:
            merged[-1] = (min(merged[-1][0], x1), merged[-1][1],
                          max(merged[-1][2], x2), max(merged[-1][3], y2))
        else:
            merged.append([x1, y1, x2, y2])

    return [tuple(b) for b in merged]


# ── Inference ──────────────────────────────────────────────────────────────────

def run(image_path):
    model = load_model()
    img = cv2.imread(image_path)
    if img is None:
        print(f"  ✗ Cannot read image: {image_path}")
        sys.exit(1)

    if CRAFT_AVAILABLE:
        try:
            lines = detect_lines_craft(img)
        except Exception as e:
            print(f"  ! CRAFT failed ({e}), falling back to morphological detector")
            lines = detect_lines_morphological(img)
    else:
        lines = detect_lines_morphological(img)

    print(f"  ✓ Detected {len(lines)} text lines\n")
    print("=" * 60)

    for i, (x1, y1, x2, y2) in enumerate(lines):
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        processed = preprocess_crop(crop)
        tensor = torch.FloatTensor(processed).unsqueeze(0).unsqueeze(0) / 255.0

        with torch.no_grad():
            log_probs = model(tensor)  # (seq_len, 1, num_classes)

        log_probs_seq = log_probs.squeeze(1)  # (seq_len, num_classes)
        raw_text = decode_ctc(log_probs_seq)
        confidence = compute_confidence(log_probs_seq)
        review = needs_review(confidence)

        # Fuzzy correction (handles bigrams for two-word brands)
        words = raw_text.split()
        corrected = raw_text
        if words:
            corrections = correct_prescription_text(words)
            corrected = " ".join(c["corrected"] for c in corrections)

        status = "NEEDS REVIEW" if review else "OK"
        print(f"  Line {i+1}  [{x1},{y1} – {x2},{y2}]")
        print(f"    Raw:        {raw_text!r}")
        if corrected != raw_text:
            print(f"    Corrected:  {corrected!r}")
        print(f"    Confidence: {confidence:.1%}  |  {status}")
        print("-" * 60)

    print("=" * 60)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 run_line_ocr.py <image_path>")
        sys.exit(0)
    run(sys.argv[1])
