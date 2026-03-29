"""
PyTorch Dataset for handwriting OCR.
Reads from manifest CSV, applies preprocessing, encodes labels.

Supports:
  - Synthetic data merging
  - Curriculum learning (sorting by difficulty)
"""
import os
import sys
import csv
import torch
import numpy as np
from torch.utils.data import Dataset, SubsetRandomSampler, Sampler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    PROCESSED_DIR, IMG_HEIGHT, IMG_WIDTH,
    USE_SYNTHETIC_DATA, AUGMENT_LEVEL
)
from preprocessing.transforms import preprocess_image
from model.utils import encode_label


class HandwritingDataset(Dataset):
    """
    Dataset that reads from a manifest CSV file.

    Each row has: image_path, text_label, source_dataset, granularity, is_noisy_label
    """

    def __init__(self, csv_path: str, full_pipeline: bool = True, max_label_len: int = 50,
                 cache_tensors: bool = False, augment: bool = False,
                 augment_level: str = AUGMENT_LEVEL,
                 include_synthetic: bool = USE_SYNTHETIC_DATA):
        self.full_pipeline = full_pipeline
        self.max_label_len = max_label_len
        self.augment = augment
        self.augment_level = augment_level
        # Augmentation randomises each access — caching the first result would freeze it
        self.cache_tensors = cache_tensors and not augment
        self.samples = []
        self._cache = {}  # idx -> img_tensor (populated on first access)

        # Load main CSV
        self._load_csv(csv_path)

        # Optionally merge synthetic data
        if include_synthetic:
            synth_csv = os.path.join(PROCESSED_DIR, "synthetic.csv")
            if os.path.exists(synth_csv):
                n_before = len(self.samples)
                self._load_csv(synth_csv)
                n_synth = len(self.samples) - n_before
                print(f"  ✓ Merged {n_synth} synthetic samples")
            else:
                print(f"  ⚠  Synthetic CSV not found: {synth_csv}")
                print(f"     Generate it: python preprocessing/generate_synthetic.py")

    def _load_csv(self, csv_path: str):
        """Load samples from a CSV file."""
        if not os.path.exists(csv_path):
            return

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_path = row["image_path"]
                label = row["text_label"]

                # Skip if label is too long or image doesn't exist
                encoded = encode_label(label)
                if len(encoded) == 0 or len(encoded) > self.max_label_len:
                    continue

                self.samples.append({
                    "image_path": img_path,
                    "text_label": label,
                    "encoded_label": encoded,
                })

    def sort_by_difficulty(self):
        """
        Sort samples by difficulty (label length as proxy).
        Shorter labels are easier to recognize.
        Used for curriculum learning.
        """
        self.samples.sort(key=lambda s: len(s["encoded_label"]))

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
                                       augment=self.augment,
                                       augment_level=self.augment_level)
            except Exception:
                # Return a blank image if preprocessing fails
                img = np.ones((IMG_HEIGHT, IMG_WIDTH), dtype="uint8") * 255

            # Convert to tensor: (H, W) → (1, H, W), normalized to [0, 1]
            img_tensor = torch.FloatTensor(img).unsqueeze(0) / 255.0

            # Cache for subsequent epochs
            if self.cache_tensors:
                self._cache[idx] = img_tensor

        label_tensor = torch.IntTensor(sample["encoded_label"])

        return img_tensor, label_tensor, sample["text_label"]


class CurriculumSampler(Sampler):
    """
    Curriculum learning sampler.

    For the first `warmup_epochs`, progressively expands the training subset
    (sorted by difficulty). After warmup, samples the full dataset randomly.

    Usage:
        sampler = CurriculumSampler(dataset, epoch=current_epoch, warmup_epochs=20)
        loader = DataLoader(dataset, sampler=sampler, ...)
    """

    def __init__(self, dataset: HandwritingDataset, epoch: int,
                 warmup_epochs: int = 20):
        self.dataset = dataset
        self.epoch = epoch
        self.warmup_epochs = warmup_epochs
        self.n = len(dataset)

    def __len__(self):
        if self.epoch < self.warmup_epochs:
            # Progressive expansion: start at 30%, reach 100% by warmup end
            frac = 0.3 + 0.7 * (self.epoch / self.warmup_epochs)
            return int(self.n * frac)
        return self.n

    def __iter__(self):
        if self.epoch < self.warmup_epochs:
            subset_size = len(self)
            # Use first `subset_size` samples (dataset should be pre-sorted)
            indices = list(range(subset_size))
            # Shuffle within the subset
            perm = torch.randperm(subset_size).tolist()
            return iter([indices[i] for i in perm])
        else:
            # Full shuffle after warmup
            return iter(torch.randperm(self.n).tolist())


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
