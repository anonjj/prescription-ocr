"""
Run OCR line-by-line on a prescription image using model.ptl (TorchScript).
Uses CRAFT text detector for tight bounding boxes when available,
falls back to morphological detection otherwise.

Usage: python3 run_line_ocr.py <image_path>
"""
import os
import sys
import cv2  # type: ignore
import numpy as np  # type: ignore
import torch  # type: ignore

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import IMG_HEIGHT, IMG_WIDTH, IDX_TO_CHAR, BLANK_LABEL  # type: ignore
from postprocessing.confidence import compute_confidence, needs_review  # type: ignore
from postprocessing.lexicon import correct_prescription_text  # type: ignore

# CRAFT detector — optional dependency
try:
    # craft-text-detector references torchvision.models.vgg.model_urls which was
    # removed in torchvision >=0.13. Patch it back before importing.
    import torchvision.models.vgg as _vgg  # type: ignore
    if not hasattr(_vgg, "model_urls"):
        _vgg.model_urls = {
            "vgg16_bn": "https://download.pytorch.org/models/vgg16_bn-6c64b313.pth",
        }
    from craft_text_detector import Craft  # type: ignore
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
    """Apply the same pipeline as preprocessing/transforms.py to a BGR crop.
    CLAHE-only — no binarization (matches training pipeline)."""
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    # No binarization — CLAHE only (matches training)
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

def detect_lines_morphological(image_bgr, min_line_height=12, padding=4):
    """
    Line detector using adaptive threshold + horizontal dilation.
    Tuned for phone-camera prescription images.
    Returns list of (x1, y1, x2, y2).
    """
    img_h, img_w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 5)
    binary = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 25, 12
    )

    # Clean noise
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_clean)

    # Horizontal dilation only — connects characters, keeps lines separate
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
    dilated = cv2.dilate(binary, kernel_h, iterations=2)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > img_w * 0.08 and h > min_line_height:
            boxes.append((x, y, x + w, y + h))

    boxes.sort(key=lambda b: b[1])

    # Merge only if >50% vertical overlap
    merged = []
    for box in boxes:
        x1, y1, x2, y2 = box
        if merged:
            px1, py1, px2, py2 = merged[-1]
            overlap = min(py2, y2) - max(py1, y1)
            smaller_h = min(py2 - py1, y2 - y1)
            if overlap > smaller_h * 0.5:
                merged[-1] = (min(px1, x1), min(py1, y1), max(px2, x2), max(py2, y2))
                continue
        merged.append((x1, y1, x2, y2))

    # Add padding
    result = []
    for x1, y1, x2, y2 in merged:
        result.append((
            max(0, x1 - padding),
            max(0, y1 - padding),
            min(img_w, x2 + padding),
            min(img_h, y2 + padding)
        ))

    return result


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
