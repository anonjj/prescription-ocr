"""
Central configuration for Doctor Handwriting OCR project.
"""
import os
import string

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
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
CHARS = string.ascii_letters + string.digits + " .,-/'()+#:[]&;*%!<>?=@^_`{|}~\"\\"
BLANK_LABEL = 0
NUM_CLASSES = len(CHARS) + 1  # +1 for CTC blank

# Character ↔ index mappings
CHAR_TO_IDX = {ch: i + 1 for i, ch in enumerate(CHARS)}
IDX_TO_CHAR = {i + 1: ch for i, ch in enumerate(CHARS)}

# ──────────────────────────────────────────────
# Training hyperparameters
# ──────────────────────────────────────────────
BATCH_SIZE = 256
LEARNING_RATE = 3e-4
NUM_EPOCHS = 100
PATIENCE = 20
LR_STEP_SIZE = 20
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
# CNN Backbone  ("vgg" | "efficientnet")
# ──────────────────────────────────────────────
CNN_BACKBONE = "vgg"           # "vgg" = original 5-block CNN, "efficientnet" = pretrained EfficientNet-B0

# ──────────────────────────────────────────────
# Sequence model  ("bilstm" | "transformer")
# ──────────────────────────────────────────────
SEQ_MODEL = "bilstm"           # "bilstm" = original BiLSTM, "transformer" = Transformer encoder
TRANSFORMER_HEADS = 4
TRANSFORMER_LAYERS = 4
TRANSFORMER_DIM = 512          # d_model (must match CNN output or gets projected)
TRANSFORMER_FF_DIM = 1024      # feed-forward inner dim
TRANSFORMER_DROPOUT = 0.1

# ──────────────────────────────────────────────
# Spatial Transformer Network (STN)
# ──────────────────────────────────────────────
USE_STN = False                # learnable geometric rectification before CNN

# ──────────────────────────────────────────────
# Beam search decoding
# ──────────────────────────────────────────────
USE_BEAM_SEARCH = True         # True → beam search w/ LM, False → greedy
BEAM_WIDTH = 10
LM_ALPHA = 0.5                # language-model weight
LM_BETA = 1.0                 # word-insertion bonus

# ──────────────────────────────────────────────
# Augmentation
# ──────────────────────────────────────────────
AUGMENT_LEVEL = "strong"       # "none" | "light" | "strong"
USE_SYNTHETIC_DATA = False     # merge synthetic CSV during training
SYNTHETIC_COUNT = 5000         # default number of synthetic samples to generate

# ──────────────────────────────────────────────
# Curriculum learning
# ──────────────────────────────────────────────
USE_CURRICULUM = False         # sort by difficulty for early epochs
CURRICULUM_WARMUP = 20         # epochs of curriculum before full shuffle

# ──────────────────────────────────────────────
# Post-processing
# ──────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.6    # below → NEEDS_REVIEW
FUZZY_MATCH_THRESHOLD = 80    # fuzzywuzzy score

# ──────────────────────────────────────────────
# Dataset identifiers
# ──────────────────────────────────────────────
KAGGLE_DATASETS = {
    "rxhandbd":        "banasmitajena/rxhandbd",
    "prescription_bd": "mamun1113/doctors-handwritten-prescription-bd-dataset",
}

HUGGINGFACE_DATASETS = {
    "iam_line": "Teklia/IAM-line",
    "iam_word": "priyank-m/IAM_words_text_recognition",
    "medical_prescription": "chinmays18/medical-prescription-dataset",
}

# ──────────────────────────────────────────────
# Data split ratios
# ──────────────────────────────────────────────
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15