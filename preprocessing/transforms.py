"""
Image preprocessing pipeline for handwriting OCR.
Grayscale → Denoise → CLAHE → Adaptive threshold → Deskew → Resize+Pad.
"""
import cv2
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import IMG_HEIGHT, IMG_WIDTH


def to_grayscale(img: np.ndarray) -> np.ndarray:
    """Convert to grayscale if needed."""
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def denoise(img: np.ndarray) -> np.ndarray:
    """Apply median blur denoising (fast)."""
    return cv2.medianBlur(img, 3)


def enhance_contrast(img: np.ndarray) -> np.ndarray:
    """Apply CLAHE for adaptive contrast enhancement."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img)


def adaptive_threshold(img: np.ndarray) -> np.ndarray:
    """Apply adaptive Gaussian thresholding."""
    return cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )


def deskew(img: np.ndarray) -> np.ndarray:
    """Correct image skew using moments."""
    coords = np.column_stack(np.where(img < 128))
    if len(coords) < 10:
        return img

    try:
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        if abs(angle) > 15:  # skip if angle is too extreme
            return img

        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)
    except Exception:
        return img


def resize_pad(img: np.ndarray, target_h: int = IMG_HEIGHT,
               target_w: int = IMG_WIDTH) -> np.ndarray:
    """Resize while maintaining aspect ratio, then pad to target size."""
    h, w = img.shape[:2]
    ratio = target_h / h
    new_w = min(int(w * ratio), target_w)
    new_h = target_h

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Pad to target width with white
    padded = np.ones((target_h, target_w), dtype=np.uint8) * 255
    padded[:, :new_w] = resized

    return padded


def get_augmentation_pipeline():
    """Return an albumentations augmentation pipeline for training."""
    try:
        import albumentations as A
    except ImportError:
        raise ImportError("albumentations not installed. Run: pip install albumentations")

    return A.Compose([
        A.Affine(
            scale=(0.85, 1.15), rotate=(-10, 10),
            mode=cv2.BORDER_REPLICATE, p=0.6
        ),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.GaussNoise(std_range=(0.02, 0.11), p=0.4),
        A.ElasticTransform(alpha=20, sigma=4, p=0.3),
    ])


def preprocess_image(img_path: str, full_pipeline: bool = True,
                     augment: bool = False) -> np.ndarray:
    """
    Full preprocessing pipeline for a handwriting image.

    Args:
        img_path: Path to the image file.
        full_pipeline: If True, apply all preprocessing steps.
                       If False, only grayscale + resize (for speed).

    Returns:
        Preprocessed image as numpy array (H x W, uint8).
    """
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")

    img = to_grayscale(img)

    if full_pipeline:
        img = denoise(img)
        img = enhance_contrast(img)
        img = adaptive_threshold(img)
        img = deskew(img)

    if augment:
        pipeline = get_augmentation_pipeline()
        img = pipeline(image=img)["image"]

    img = resize_pad(img)
    return img
