"""
Kaggle Notebook Training Script for Doctor Handwriting OCR.

=== KEY DIFFERENCES vs COLAB ===
- No Drive mount needed — checkpoints saved to /kaggle/working/ (persists across sessions)
- Kaggle API credentials pre-configured (no kaggle.json upload)
- P100 GPU available (16GB VRAM vs T4's 15GB — can push batch size higher)
- 30h/week GPU quota, 12h session limit
- Internet must be ON in notebook settings to clone from GitHub & download HF datasets
- Kaggle datasets attached via "Add Data" panel — no re-download needed across sessions

=== HOW TO USE ===
1. Go to kaggle.com/code → New Notebook
2. Settings → Accelerator → GPU T4 x2
3. Settings → Internet → On
4. Copy-paste each CELL section below into separate notebook cells
5. Run cells in order

Checkpoint strategy:
  /kaggle/working/checkpoints/latest.pt   ← saved every epoch (resume point)
  /kaggle/working/checkpoints/best_model.pt ← saved when CER improves
  /kaggle/working/checkpoints/epoch_N.pt  ← milestone every 10 epochs
  On session end, Kaggle auto-saves /kaggle/working/ as notebook output.
  On next session: attach previous notebook output as a dataset to resume.
"""

# ============================================================
# CELL 1: Setup & Install Dependencies
# ============================================================
"""
!pip install -q editdistance fuzzywuzzy python-Levenshtein

import os
import torch

print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

CKPT_DIR = '/kaggle/working/checkpoints'
os.makedirs(CKPT_DIR, exist_ok=True)
print(f"Checkpoint dir: {CKPT_DIR}")
print("Setup complete")
"""

# ============================================================
# CELL 2: Clone Repo
# ============================================================
"""
import os

REPO_DIR = '/kaggle/working/Projecat'

if not os.path.exists(REPO_DIR):
    !git clone https://github.com/anonjj/prescription-ocr.git {REPO_DIR}
else:
    print("Repo already exists, pulling latest...")
    !git -C {REPO_DIR} pull

%cd {REPO_DIR}
!ls
"""

# ============================================================
# CELL 3: Download Datasets
# ============================================================
"""
# Kaggle API credentials are pre-configured in Kaggle notebooks — no upload needed.
# HuggingFace datasets download via internet (must be ON in settings).

import sys
sys.path.insert(0, '/kaggle/working/Projecat')

!python data/download_all.py

# Debug download step
import os
from config import RAW_DIR
for d in os.listdir(RAW_DIR):
    path = os.path.join(RAW_DIR, d)
    if os.path.isdir(path):
        imgs = sum(1 for _, _, f in os.walk(path) for x in f if x.endswith(('.png','.jpg','.webp')))
        print(f"{d}: {imgs} images")

# Verify
!python data/audit_datasets.py
"""

# ============================================================
# CELL 4: Prepare Data (split into train/val/test)
# ============================================================
"""
%cd /kaggle/working/Projecat

# Check if manifest + splits already exist (skip if resuming)
import os
PROCESSED_DIR = '/tmp/ocr_data/processed'

if os.path.exists(os.path.join(PROCESSED_DIR, 'train.csv')):
    print("Splits already exist, skipping data prep.")
    import pandas as pd
    for split in ['train', 'val', 'test']:
        df = pd.read_csv(os.path.join(PROCESSED_DIR, f'{split}.csv'))
        print(f"  {split}: {len(df):,} samples")
else:
    !python data/create_unified_manifest.py
    !python data/split_data.py
"""

# ============================================================
# CELL 4b: Data Distribution Check
# ============================================================
"""
import pandas as pd, os, sys
sys.path.insert(0, '/kaggle/working/Projecat')
from config import PROCESSED_DIR

for split in ['train', 'val', 'test']:
    path = os.path.join(PROCESSED_DIR, f'{split}.csv')
    if os.path.exists(path):
        df = pd.read_csv(path)
        print(f'{split}: {len(df):,} total')
        print(df['source_dataset'].value_counts().to_string())
        print()
    else:
        print(f'{split}: NOT FOUND — run Cell 4 first')
"""

# ============================================================
# CELL 5: Sanity Check
# ============================================================
"""
import os, sys, torch

REPO_DIR = '/kaggle/working/Projecat'
assert os.path.exists(REPO_DIR), f"Repo not found at {REPO_DIR} — run Cell 2 first"
%cd {REPO_DIR}
sys.path.insert(0, REPO_DIR)

from model.crnn import CRNN, count_parameters

model = CRNN()
x = torch.randn(2, 1, 64, 256)
out = model(x)
print(f"Input:      {x.shape}")
print(f"Output:     {out.shape}")
print(f"Parameters: {count_parameters(model):,}")
"""

# ============================================================
# CELL 6: Train
# ============================================================
"""
%cd /kaggle/working/Projecat
!python model/train.py --backbone efficientnet --stn --augment-level strong --beam
"""

# ============================================================
# CELL 7: Evaluate
# ============================================================
"""
%cd /kaggle/working/Projecat
!python model/evaluate.py \\
    --split test \\
    --save-predictions \\
    --checkpoint /kaggle/working/checkpoints/best_model.pt
"""

# ============================================================
# CELL 8: How to Resume Across Sessions
# ============================================================
"""
# After session ends, Kaggle saves /kaggle/working/ as notebook output.
#
# To resume in a new session:
# 1. Open your notebook → "Save Version" to lock the output
# 2. Go to the saved version → "..." menu → "Add as Dataset"
#    (or: kaggle.com/datasets → New Dataset → from notebook output)
# 3. In your new session, add that dataset via "Add Data" panel
# 4. Copy checkpoints into working dir before training:

import shutil, os, glob

# Find the attached checkpoint dataset (adjust path as needed)
prev_ckpts = glob.glob('/kaggle/input/*/checkpoints/*.pt')
if prev_ckpts:
    os.makedirs('/kaggle/working/checkpoints', exist_ok=True)
    for f in prev_ckpts:
        shutil.copy(f, '/kaggle/working/checkpoints/')
        print(f'Copied: {os.path.basename(f)}')
else:
    print('No previous checkpoints found — starting fresh')
"""

if __name__ == "__main__":
    print("=" * 60)
    print("  Kaggle Notebook Training Script")
    print("=" * 60)
    print()
    print("  Copy each CELL section into a Kaggle notebook cell.")
    print()
    print("  Settings to enable in Kaggle:")
    print("  - Accelerator: GPU T4 x2 (Do NOT use P100, it is unsupported by modern PyTorch)")
    print("  - Internet: On")
    print()
    print("  Checkpoint location: /kaggle/working/checkpoints/")
    print("  - latest.pt     saved every epoch")
    print("  - best_model.pt saved when CER improves")
    print("  - epoch_N.pt    milestone every 10 epochs")
    print()
    print("  To resume across sessions: see CELL 8")
    print("=" * 60)
rch)")
    print("  - Internet: On")
    print()
    print("  Checkpoint location: /kaggle/working/checkpoints/")
    print("  - latest.pt     saved every epoch")
    print("  - best_model.pt saved when CER improves")
    print("  - epoch_N.pt    milestone every 10 epochs")
    print()
    print("  To resume across sessions: see CELL 8")
    print("=" * 60)
