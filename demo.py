"""
Demo inference CLI for Doctor Handwriting OCR.

Usage:
  python demo.py <image_path>         — recognize text from an image
  python demo.py --camera             — capture from webcam and recognize
"""
import os
import sys
import argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import CHECKPOINTS_DIR, BLANK_LABEL
from model.crnn import CRNN
from model.utils import decode_prediction
from preprocessing.transforms import preprocess_image
from postprocessing.lexicon import fuzzy_correct, correct_prescription_text
from postprocessing.rules import extract_all
from postprocessing.confidence import compute_confidence, needs_review


def load_model(checkpoint_path: str = None):
    """Load trained CRNN model."""
    device = torch.device("cpu")  # CPU for local inference on MacBook

    ckpt_path = checkpoint_path or os.path.join(CHECKPOINTS_DIR, "best_model.pt")
    if not os.path.exists(ckpt_path):
        print(f"  ✗ Checkpoint not found: {ckpt_path}")
        print(f"    Train the model first, or specify a checkpoint path.")
        sys.exit(1)

    model = CRNN().to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    epoch = ckpt.get("epoch", "?")
    cer = ckpt.get("best_cer", "?")
    print(f"  ✓ Model loaded (epoch {epoch}, best CER: {cer})")

    return model, device


def recognize_image(image_path: str, model, device):
    """Run OCR on a single image."""
    # Preprocess
    img = preprocess_image(image_path, full_pipeline=True)

    # Convert to tensor: (H, W) → (1, 1, H, W)
    img_tensor = torch.FloatTensor(img).unsqueeze(0).unsqueeze(0) / 255.0
    img_tensor = img_tensor.to(device)

    # Inference
    with torch.no_grad():
        log_probs = model(img_tensor)  # (seq_len, 1, num_classes)

    # Decode
    _, preds = log_probs.max(2)
    pred_indices = preds.squeeze(1).cpu().tolist()  # (seq_len,)
    raw_text = decode_prediction(pred_indices)

    # Confidence
    sample_log_probs = log_probs.squeeze(1)  # (seq_len, num_classes)
    confidence = compute_confidence(sample_log_probs)
    review_flag = needs_review(confidence)

    return {
        "raw_text": raw_text,
        "confidence": confidence,
        "needs_review": review_flag,
    }


def display_result(result: dict, image_path: str = None):
    """Pretty-print OCR result."""
    print(f"\n{'═'*60}")
    if image_path:
        print(f"  📄 Image: {image_path}")
    print(f"{'─'*60}")

    print(f"  📝 Raw Text:     {result['raw_text']}")
    print(f"  📊 Confidence:   {result['confidence']:.2%}")

    if result['needs_review']:
        print(f"  ⚠️  Status:       NEEDS HUMAN REVIEW")
    else:
        print(f"  ✅ Status:       OK")

    # Post-process: fuzzy correction
    words = result['raw_text'].split()
    if words:
        corrections = correct_prescription_text(words)
        corrected_words = [c['corrected'] for c in corrections]
        corrected_text = " ".join(corrected_words)

        if corrected_text != result['raw_text']:
            print(f"  💊 Corrected:    {corrected_text}")

        # Extract prescription components
        extracted = extract_all(corrected_text)
        if extracted['dosage']:
            print(f"  💉 Dosage:       {extracted['dosage']}")
        if extracted['frequency']:
            print(f"  🕐 Frequency:    {extracted['frequency']}")
        if extracted['duration']:
            print(f"  📅 Duration:     {extracted['duration']}")

    print(f"{'═'*60}\n")


def camera_capture():
    """Capture an image from webcam and return the saved path."""
    try:
        import cv2
    except ImportError:
        print("  ✗ cv2 not installed. Run: pip install opencv-python")
        sys.exit(1)

    print("\n  📷 Opening camera...")
    print("     Press SPACE to capture, Q to quit.\n")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("  ✗ Cannot open camera")
        sys.exit(1)

    captured_path = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Doctor Handwriting OCR — Press SPACE to capture", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):  # SPACE
            captured_path = "/tmp/ocr_capture.jpg"
            cv2.imwrite(captured_path, frame)
            print(f"  ✓ Image captured → {captured_path}")
            break
        elif key == ord('q'):  # Q
            break

    cap.release()
    cv2.destroyAllWindows()
    return captured_path


def main():
    parser = argparse.ArgumentParser(
        description="Doctor Handwriting OCR — Demo Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demo.py prescription.jpg
  python demo.py --camera
  python demo.py scan.png --checkpoint models/checkpoints/best_model.pt
        """
    )
    parser.add_argument("image", nargs="?", help="Path to prescription image")
    parser.add_argument("--camera", action="store_true", help="Capture from webcam")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    args = parser.parse_args()

    if not args.image and not args.camera:
        parser.print_help()
        sys.exit(0)

    # Load model
    model, device = load_model(args.checkpoint)

    # Get image
    if args.camera:
        image_path = camera_capture()
        if not image_path:
            print("  No image captured.")
            sys.exit(0)
    else:
        image_path = args.image
        if not os.path.exists(image_path):
            print(f"  ✗ Image not found: {image_path}")
            sys.exit(1)

    # Recognize
    result = recognize_image(image_path, model, device)
    display_result(result, image_path)


if __name__ == "__main__":
    main()
