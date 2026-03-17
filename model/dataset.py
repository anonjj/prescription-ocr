"""
PyTorch Dataset for handwriting OCR.
Reads from manifest CSV, applies preprocessing, encodes labels.
"""
import os
import sys
import csv
import torch
from torch.utils.data import Dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import PROCESSED_DIR, IMG_HEIGHT, IMG_WIDTH
from preprocessing.transforms import preprocess_image
from model.utils import encode_label


class HandwritingDataset(Dataset):
    """
    Dataset that reads from a manifest CSV file.

    Each row has: image_path, text_label, source_dataset, granularity, is_noisy_label
    """

    def __init__(self, csv_path: str, full_pipeline: bool = True, max_label_len: int = 50,
                 cache_tensors: bool = False, augment: bool = False):
        self.full_pipeline = full_pipeline
        self.max_label_len = max_label_len
        self.augment = augment
        # Augmentation randomises each access — caching the first result would freeze it
        self.cache_tensors = cache_tensors and not augment
        self.samples = []
        self._cache = {}  # idx -> img_tensor (populated on first access)

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_path = row["image_path"]
                label = row["text_label"]

                # Skip if label is too long or image doesn't exist
                encoded = encode_label(label)
                if len(encoded) == 0 or len(encoded) > max_label_len:
                    continue

                self.samples.append({
                    "image_path": img_path,
                    "text_label": label,
                    "encoded_label": encoded,
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Check cache first (avoids disk I/O after first epoch)
        if self.cache_tensors and idx in self._cache:
            img_tensor = self._cache[idx]
        else:
            # Load and preprocess image
            try:
                img = preprocess_image(sample["image_path"],
                                       full_pipeline=self.full_pipeline,
                                       augment=self.augment)
            except Exception:
                # Return a blank image if preprocessing fails
                import numpy as np
                img = np.ones((IMG_HEIGHT, IMG_WIDTH), dtype="uint8") * 255

            # Convert to tensor: (H, W) → (1, H, W), normalized to [0, 1]
            img_tensor = torch.FloatTensor(img).unsqueeze(0) / 255.0

            # Cache for subsequent epochs
            if self.cache_tensors:
                self._cache[idx] = img_tensor

        label_tensor = torch.IntTensor(sample["encoded_label"])

        return img_tensor, label_tensor, sample["text_label"]


def collate_fn(batch):
    """
    Custom collate function for variable-length labels.

    Returns:
        images:       (B, 1, H, W)
        labels:       concatenated 1D tensor of all labels
        label_lengths: (B,) — length of each label
        texts:        list of original text strings
    """
    images, labels, texts = zip(*batch)

    images = torch.stack(images, 0)
    label_lengths = torch.IntTensor([len(l) for l in labels])
    labels = torch.cat(labels, 0)

    return images, labels, label_lengths, texts
