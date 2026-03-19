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
2. Settings → Accelerator → GPU P100 (or T4 x2)
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
TRAIN_CODE = """
import os
import sys
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm.auto import tqdm

REPO_DIR = '/kaggle/working/Projecat'
assert os.path.exists(REPO_DIR), f"Repo not found at {REPO_DIR}"
os.chdir(REPO_DIR)
sys.path.insert(0, REPO_DIR)
from config import (
    PROCESSED_DIR, BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, PATIENCE,
    LR_STEP_SIZE, LR_GAMMA, BLANK_LABEL
)
from model.crnn import CRNN, count_parameters
from model.dataset import HandwritingDataset, collate_fn
from model.utils import decode_prediction, compute_cer, compute_wer

CKPT_DIR = '/kaggle/working/checkpoints'
os.makedirs(CKPT_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')
if device.type == 'cuda':
    print(f'GPU: {torch.cuda.get_device_name(0)}')

EFFECTIVE_BATCH = BATCH_SIZE

# cache_tensors=False + num_workers=4: workers start before any cache is built,
# so no copy-on-write RAM explosion. Images loaded and preprocessed from disk in parallel.
train_dataset = HandwritingDataset(
    os.path.join(PROCESSED_DIR, 'train.csv'),
    full_pipeline=True,
    cache_tensors=False,
    augment=True
)
val_dataset = HandwritingDataset(
    os.path.join(PROCESSED_DIR, 'val.csv'),
    full_pipeline=True,
    cache_tensors=False
)
# ── FIX 2: Oversample prescription data for ~50/50 balance ──
rx_samples = [s for s in train_dataset.samples
              if 'prescription' in s['image_path'].lower()
              or 'bd-dataset' in s['image_path'].lower()]
train_dataset.samples.extend(rx_samples)
print(f'Train: {len(train_dataset)} (after oversampling +{len(rx_samples)} prescription dupes)')
print(f'Val:   {len(val_dataset)}')

train_loader = DataLoader(train_dataset, batch_size=EFFECTIVE_BATCH, shuffle=True,
                          collate_fn=collate_fn, num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_dataset, batch_size=EFFECTIVE_BATCH, shuffle=False,
                          collate_fn=collate_fn, num_workers=4, pin_memory=True)

# Model
model = CRNN().to(device)
ctc_loss  = nn.CTCLoss(blank=BLANK_LABEL, zero_infinity=True)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA)
scaler    = GradScaler('cuda') if device.type == 'cuda' else None
use_amp   = device.type == 'cuda'

# Resume — prefer latest.pt, fall back to best_model.pt
# To resume from a previous Kaggle session, attach that session's output
# as a dataset and copy the checkpoint:
#   !cp /kaggle/input/<your-dataset>/checkpoints/latest.pt /kaggle/working/checkpoints/
start_epoch     = 0
best_cer        = float('inf')
patience_counter = 0

for resume_name in ['latest.pt', 'best_model.pt']:
    resume_path = os.path.join(CKPT_DIR, resume_name)
    if os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scheduler_state_dict' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch      = ckpt.get('epoch', 0) + 1
        best_cer         = ckpt.get('best_cer', float('inf'))
        patience_counter = ckpt.get('patience_counter', 0)
        print(f'Resumed from {resume_name} — epoch {start_epoch}, best CER: {best_cer:.4f}')
        break

print(f'\\n{"="*70}')
print(f'  {"Epoch":>5} | {"Train Loss":>11} | {"Val Loss":>9} | {"CER":>7} | {"WER":>7} | {"Time":>6}')
print(f'{"="*70}')

for epoch in range(start_epoch, NUM_EPOCHS):
    model.train()
    total_loss = 0.0
    n_batches  = 0
    t0         = time.time()

    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS}', leave=False)
    for images, labels, label_lengths, texts in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast('cuda', enabled=use_amp):
            outputs = model(images)
            seq_len = outputs.size(0)
            bs      = images.size(0)
            input_lengths = torch.full((bs,), seq_len, dtype=torch.long, device=device)
            loss = ctc_loss(outputs, labels, input_lengths, label_lengths.to(device))

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        total_loss += loss.item()
        n_batches  += 1
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    scheduler.step()
    avg_train_loss = total_loss / max(n_batches, 1)

    # Validate
    model.eval()
    val_loss_total = 0.0
    val_cer_total  = 0.0
    val_wer_total  = 0.0
    val_n          = 0

    with torch.no_grad():
        for images, labels, label_lengths, texts in val_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with autocast('cuda', enabled=use_amp):
                outputs = model(images)
                seq_len = outputs.size(0)
                bs      = images.size(0)
                input_lengths = torch.full((bs,), seq_len, dtype=torch.long, device=device)
                loss = ctc_loss(outputs, labels, input_lengths, label_lengths.to(device))

            val_loss_total += loss.item() * bs

            _, preds = outputs.max(2)
            preds = preds.permute(1, 0).cpu()
            for i in range(bs):
                pred_text = decode_prediction(preds[i].tolist())
                val_cer_total += compute_cer(pred_text, texts[i])
                val_wer_total += compute_wer(pred_text, texts[i])
                val_n += 1

    val_loss = val_loss_total / max(val_n, 1)
    val_cer  = val_cer_total  / max(val_n, 1)
    val_wer  = val_wer_total  / max(val_n, 1)
    elapsed  = time.time() - t0

    print(f'  {epoch+1:>5} | {avg_train_loss:>11.4f} | {val_loss:>9.4f} | {val_cer:>6.4f} | {val_wer:>6.4f} | {elapsed:>5.1f}s')

    # Save best
    if val_cer < best_cer:
        best_cer         = val_cer
        patience_counter = 0
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_cer': best_cer,
            'patience_counter': patience_counter,
        }, os.path.join(CKPT_DIR, 'best_model.pt'))
        print(f'         -> Best model saved (CER: {best_cer:.4f})')
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f'\\nEarly stopping at epoch {epoch+1}')
            break

    # Always save latest.pt — at most 1 epoch lost on session timeout
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_cer': best_cer,
        'patience_counter': patience_counter,
    }, os.path.join(CKPT_DIR, 'latest.pt'))

    # Milestone snapshots every 10 epochs
    if (epoch + 1) % 10 == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_cer': best_cer,
            'patience_counter': patience_counter,
        }, os.path.join(CKPT_DIR, f'epoch_{epoch+1}.pt'))

print(f'\\nTraining complete! Best CER: {best_cer:.4f}')
print(f'Checkpoints saved to: {CKPT_DIR}')
print('Save this notebook output to preserve checkpoints for next session.')
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
    print("  - Accelerator: GPU P100")
    print("  - Internet: On")
    print()
    print("  Checkpoint location: /kaggle/working/checkpoints/")
    print("  - latest.pt     saved every epoch")
    print("  - best_model.pt saved when CER improves")
    print("  - epoch_N.pt    milestone every 10 epochs")
    print()
    print("  To resume across sessions: see CELL 8")
    print("=" * 60)
