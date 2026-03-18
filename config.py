"""
Central configuration for Doctor Handwriting OCR project.
"""
import os
import string

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
# Data dir moved to /tmp to bypass strict macOS folder permissions on the main dir
DATA_DIR = "/tmp/ocr_data"
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
CHECKPOINTS_DIR = os.path.join(MODELS_DIR, "checkpoints")

# Create directories if they don't exist
for d in [RAW_DIR, PROCESSED_DIR, MODELS_DIR, CHECKPOINTS_DIR]:
    os.makedirs(d, exist_ok=True)

# ──────────────────────────────────────────────
# Image settings
# ──────────────────────────────────────────────
IMG_HEIGHT = 64
IMG_WIDTH = 256
IMG_CHANNELS = 1  # grayscale

# ──────────────────────────────────────────────
# Character set
# ──────────────────────────────────────────────
# 0 = CTC blank token
CHARS = string.ascii_letters + string.digits + " .,-/'()+"
BLANK_LABEL = 0
NUM_CLASSES = len(CHARS) + 1  # +1 for CTC blank

# Character ↔ index mappings
CHAR_TO_IDX = {ch: i + 1 for i, ch in enumerate(CHARS)}
IDX_TO_CHAR = {i + 1: ch for i, ch in enumerate(CHARS)}

# ──────────────────────────────────────────────
# Training hyperparameters
# ──────────────────────────────────────────────
BATCH_SIZE = 320  # Optimized for T4 GPU (was 32)
LEARNING_RATE = 1e-3
NUM_EPOCHS = 100  # Extended for longer training
PATIENCE = 15  # early stopping
LR_STEP_SIZE = 25  # Decay LR every 25 epochs (was 15)
LR_GAMMA = 0.1
NUM_WORKERS = 4

# ──────────────────────────────────────────────
# Model architecture
# ──────────────────────────────────────────────
CNN_OUTPUT_CHANNELS = 512
RNN_HIDDEN_SIZE = 256
RNN_NUM_LAYERS = 2
RNN_DROPOUT = 0.3

# ──────────────────────────────────────────────
# Post-processing
# ──────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.6  # below → NEEDS_REVIEW
FUZZY_MATCH_THRESHOLD = 80  # fuzzywuzzy score

# ──────────────────────────────────────────────
# Dataset identifiers
# ──────────────────────────────────────────────
KAGGLE_DATASETS = {
    "prescription_bd": "mamun1113/doctors-handwritten-prescription-bd-dataset",
    "ocr_processed":   "nadaarfaoui/ocr-processed-handwritten-prescriptions",
}

HUGGINGFACE_DATASETS = {
    "iam_line":       "Teklia/IAM-line",
    "ocr_handwriting": "soyeb-jim285/ocr-handwriting-data",
}

# ──────────────────────────────────────────────
# Data split ratios
# ──────────────────────────────────────────────
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
