"""
Image preprocessing pipeline for handwriting OCR.
Grayscale → Denoise → CLAHE → Binarize (Otsu/Adaptive) → Deskew → Resize+Pad.

Augmentation levels:
  - "none": no augmentation
  - "light": original pipeline (affine, brightness, noise, elastic)
  - "strong": adds perspective, motion blur, grid distortion, coarse dropout
"""
import cv2  # type: ignore
import numpy as np  # type: ignore
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import IMG_HEIGHT, IMG_WIDTH, AUGMENT_LEVEL  # type: ignore


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


def binarize(img: np.ndarray, method: str = "otsu") -> np.ndarray:
    """Binarize image. 'otsu' works better for ink-on-paper, 'adaptive' for uneven lighting."""
    if method == "otsu":
        _, result = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return result
    else:
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

    # Invert so text is white (255) on black (0) background
    # This is standard for many OCR architectures and more stable for padding
    if np.mean(resized) > 127:
        resized = cv2.bitwise_not(resized)

    # Pad to target width with black (0)
    padded = np.zeros((target_h, target_w), dtype=np.uint8)
    padded[:, :new_w] = resized

    return padded


def get_augmentation_pipeline(level: str = "light"):
    """
    Return an albumentations augmentation pipeline for training.

    Args:
        level: "light" (original), "strong" (enhanced), or "none"

    Returns:
        albumentations.Compose pipeline
    """
    try:
        import albumentations as A  # type: ignore
    except ImportError:
        raise ImportError("albumentations not installed. Run: pip install albumentations")

    if level == "none":
        return None

    # Light augmentation — original pipeline
    light_transforms = [
        A.Affine(
            scale=(0.85, 1.15), rotate=(-10, 10),
            fill=0, p=0.6
        ),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.GaussNoise(std_range=(0.02, 0.11), p=0.4),
        A.ElasticTransform(alpha=20, sigma=4, p=0.3),
    ]

    if level == "light":
        return A.Compose(light_transforms)

    # Strong augmentation — adds camera-specific distortions
    strong_transforms = light_transforms + [
        # Perspective distortion — simulates phone camera angles
        A.Perspective(scale=(0.03, 0.08), p=0.4, pad_val=0),

        # Motion blur — simulates camera shake
        A.MotionBlur(blur_limit=(3, 7), p=0.3),

        # Grid distortion — simulates paper warping/curling
        A.GridDistortion(num_steps=5, distort_limit=0.2, p=0.3),

        # Coarse dropout — simulates ink smudges or partial occlusion
        A.CoarseDropout(
            num_holes_range=(1, 3),
            hole_height_range=(4, 8),
            hole_width_range=(4, 8),
            fill=0, p=0.2
        ),

        # Downscale then upscale — simulates low-resolution capture
        A.Downscale(scale_range=(0.5, 0.8), p=0.2),

        # Sharpen — counteracts blur augmentation variety
        A.Sharpen(alpha=(0.1, 0.3), lightness=(0.8, 1.0), p=0.2),
    ]

    return A.Compose(strong_transforms)


def preprocess_image(img_path: str, full_pipeline: bool = True,
                     augment: bool = False,
                     augment_level: str = AUGMENT_LEVEL) -> np.ndarray:
    """
    Full preprocessing pipeline for a handwriting image.

    Args:
        img_path: Path to the image file.
        full_pipeline: If True,  apply denoise + CLAHE + deskew.
                       If False, apply denoise + CLAHE only.
        augment: Whether to apply augmentation.
        augment_level: "none", "light", or "strong"

    Returns:
        Preprocessed image as numpy array (H x W, uint8).
    """
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")

    img = to_grayscale(img)
    img = denoise(img)
    img = enhance_contrast(img)

    if full_pipeline:
        # Skip binarization — CLAHE-only preserves more info on phone-camera images
        # img = binarize(img, method="otsu")
        img = deskew(img)

    if augment and augment_level != "none":
        pipeline = get_augmentation_pipeline(level=augment_level)
        if pipeline is not None:
            img = pipeline(image=img)["image"]

    img = resize_pad(img)
    return img
