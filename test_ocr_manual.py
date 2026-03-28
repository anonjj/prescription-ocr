
import cv2
import numpy as np
import torch
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import IMG_HEIGHT, IMG_WIDTH, IDX_TO_CHAR, BLANK_LABEL
from postprocessing.confidence import compute_confidence

MODEL_PATH = "models/model.ptl"

def decode_ctc(log_probs_tensor):
    indices = log_probs_tensor.argmax(dim=1).tolist()
    chars, prev = [], None
    for idx in indices:
        if idx != BLANK_LABEL and idx != prev:
            chars.append(IDX_TO_CHAR.get(idx, ""))
        prev = idx
    return "".join(chars)

def preprocess_crop(crop_bgr):
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    h, w = gray.shape
    ratio = IMG_HEIGHT / h
    new_w = min(int(w * ratio), IMG_WIDTH)
    resized = cv2.resize(gray, (new_w, IMG_HEIGHT), interpolation=cv2.INTER_AREA)
    padded = np.ones((IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8) * 255
    padded[:, :new_w] = resized
    return padded

def run_ocr_on_crop(model, crop):
    processed = preprocess_crop(crop)
    tensor = torch.FloatTensor(processed).unsqueeze(0).unsqueeze(0) / 255.0
    with torch.no_grad():
        log_probs = model(tensor)
    log_probs_seq = log_probs.squeeze(1)
    text = decode_ctc(log_probs_seq)
    confidence = compute_confidence(log_probs_seq)
    return text, confidence

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}")
        return

    model = torch.jit.load(MODEL_PATH, map_location="cpu")
    model.eval()

    img = cv2.imread("Example2.jpg")
    if img is None:
        print("Image not found")
        return

    # Manually defined crops for medications (approximate)
    # Based on 480x640 image
    crops = [
        ("Line 1 (Ultrafen)", (340, 345, 140, 40)),
        ("Line 2 (Relentus)", (340, 410, 140, 40)),
        ("Line 3 (Prozit)",   (340, 475, 140, 40)),
        ("Line 4 (Ultra-D)",  (340, 530, 140, 40)),
        ("Line 5 (Cartilix)", (340, 585, 140, 40)),
    ]

    print(f"{'Crop Name':<20} | {'Recognized Text':<20} | {'Confidence':<10}")
    print("-" * 55)

    for name, (x, y, w, h) in crops:
        crop_img = img[y:y+h, x:x+w]
        if crop_img.size == 0:
            print(f"{name:<20} | EMPTY CROP")
            continue
        
        # Save crop for inspection
        cv2.imwrite(f"/tmp/crop_{name.split()[1].lower()}.png", crop_img)
        
        text, conf = run_ocr_on_crop(model, crop_img)
        print(f"{name:<20} | {text:<20} | {conf:.1%}")

if __name__ == "__main__":
    main()
