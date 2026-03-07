"""
Google Colab Training Script for Doctor Handwriting OCR.

=== HOW TO USE ===
1. Open Google Colab (colab.research.google.com)
2. Create a new notebook
3. Set Runtime → Change runtime type → T4 GPU
4. Copy-paste each section below into separate cells
5. Run cells in order

The script will:
- Install dependencies
- Upload/download datasets
- Train the CRNN model on GPU
- Save checkpoints to Google Drive
"""

# ============================================================
# CELL 1: Setup & Install Dependencies
# ============================================================
"""
!pip install -q torch torchvision datasets kaggle Pillow opencv-python-headless pandas numpy matplotlib scikit-learn editdistance fuzzywuzzy python-Levenshtein tqdm

# Mount Google Drive for checkpoint persistence
from google.colab import drive
drive.mount('/content/drive')

import os
os.makedirs('/content/drive/MyDrive/ocr_project/checkpoints', exist_ok=True)
print("✓ Setup complete")
"""

# ============================================================
# CELL 2: Clone from GitHub
# ============================================================
"""
!git clone https://github.com/anonjj/prescription-ocr.git Projecat
%cd /content/Projecat
"""

# ============================================================
# CELL 3: Download Datasets
# ============================================================
"""
# Upload your kaggle.json
from google.colab import files
uploaded = files.upload()  # Upload kaggle.json

!mkdir -p ~/.kaggle
!mv kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# Download all datasets
!python data/download_all.py
"""

# ============================================================
# CELL 4: Prepare Data
# ============================================================
"""
# Audit datasets
!python data/audit_datasets.py

# Create unified manifest
!python data/create_unified_manifest.py

# Split into train/val/test
!python data/split_data.py
"""

# ============================================================
# CELL 5: Quick Sanity Check
# ============================================================
"""
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

# Test model forward pass
from model.crnn import CRNN, count_parameters
model = CRNN()
x = torch.randn(2, 1, 64, 256)
out = model(x)
print(f"\\nInput:      {x.shape}")
print(f"Output:     {out.shape}")
print(f"Parameters: {count_parameters(model):,}")
"""

# ============================================================
# CELL 6: Train! (This is the main training cell)
# ============================================================
TRAIN_CODE = """
import os
import sys
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, '/content/Projecat')
from config import (
    PROCESSED_DIR, BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, PATIENCE,
    LR_STEP_SIZE, LR_GAMMA, BLANK_LABEL
)
from model.crnn import CRNN, count_parameters
from model.dataset import HandwritingDataset, collate_fn
from model.utils import decode_prediction, compute_cer, compute_wer

# Config
DRIVE_CKPT_DIR = '/content/drive/MyDrive/ocr_project/checkpoints'
LOCAL_CKPT_DIR = '/content/Projecat/models/checkpoints'
os.makedirs(DRIVE_CKPT_DIR, exist_ok=True)
os.makedirs(LOCAL_CKPT_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# Load Data
train_dataset = HandwritingDataset(os.path.join(PROCESSED_DIR, 'train.csv'))
val_dataset = HandwritingDataset(os.path.join(PROCESSED_DIR, 'val.csv'))
print(f'Train: {len(train_dataset)}, Val: {len(val_dataset)}')

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          collate_fn=collate_fn, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                        collate_fn=collate_fn, num_workers=2, pin_memory=True)

# Model
model = CRNN().to(device)
ctc_loss = nn.CTCLoss(blank=BLANK_LABEL, zero_infinity=True)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA)

# Resume from checkpoint if available
start_epoch = 0
best_cer = float('inf')
resume_path = os.path.join(DRIVE_CKPT_DIR, 'best_model.pt')
if os.path.exists(resume_path):
    ckpt = torch.load(resume_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    start_epoch = ckpt.get('epoch', 0) + 1
    best_cer = ckpt.get('best_cer', float('inf'))
    print(f'Resumed from epoch {start_epoch}, best CER: {best_cer:.4f}')

patience_counter = 0

print(f'\\n{"="*70}')
print(f'  {"Epoch":>5} | {"Train Loss":>11} | {"Val Loss":>9} | {"CER":>7} | {"WER":>7} | {"Time":>6}')
print(f'{"="*70}')

for epoch in range(start_epoch, NUM_EPOCHS):
    model.train()
    total_loss = 0.0
    n_batches = 0
    t0 = time.time()

    for images, labels, label_lengths, texts in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        seq_len = outputs.size(0)
        bs = images.size(0)
        input_lengths = torch.full((bs,), seq_len, dtype=torch.long)

        loss = ctc_loss(outputs, labels, input_lengths, label_lengths)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    scheduler.step()
    avg_train_loss = total_loss / max(n_batches, 1)

    # Validate
    model.eval()
    val_loss_total = 0.0
    val_cer_total = 0.0
    val_wer_total = 0.0
    val_n = 0

    with torch.no_grad():
        for images, labels, label_lengths, texts in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            seq_len = outputs.size(0)
            bs = images.size(0)
            input_lengths = torch.full((bs,), seq_len, dtype=torch.long)

            loss = ctc_loss(outputs, labels, input_lengths, label_lengths)
            val_loss_total += loss.item() * bs

            _, preds = outputs.max(2)
            preds = preds.permute(1, 0)

            for i in range(bs):
                pred_text = decode_prediction(preds[i].cpu().tolist())
                val_cer_total += compute_cer(pred_text, texts[i])
                val_wer_total += compute_wer(pred_text, texts[i])
                val_n += 1

    val_loss = val_loss_total / max(val_n, 1)
    val_cer = val_cer_total / max(val_n, 1)
    val_wer = val_wer_total / max(val_n, 1)
    elapsed = time.time() - t0

    print(f'  {epoch+1:>5} | {avg_train_loss:>11.4f} | {val_loss:>9.4f} | {val_cer:>6.4f} | {val_wer:>6.4f} | {elapsed:>5.1f}s')

    # Save best
    if val_cer < best_cer:
        best_cer = val_cer
        patience_counter = 0
        for ckpt_dir in [LOCAL_CKPT_DIR, DRIVE_CKPT_DIR]:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_cer': best_cer,
            }, os.path.join(ckpt_dir, 'best_model.pt'))
        print(f'         ↳ Best model saved (CER: {best_cer:.4f})')
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f'\\nEarly stopping at epoch {epoch+1}')
            break

    # Save periodic checkpoint to Drive
    if (epoch + 1) % 5 == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_cer': best_cer,
        }, os.path.join(DRIVE_CKPT_DIR, f'epoch_{epoch+1}.pt'))

print(f'\\nTraining complete! Best CER: {best_cer:.4f}')
"""

# ============================================================
# CELL 7: Evaluate
# ============================================================
"""
!python model/evaluate.py --split test --save-predictions --checkpoint /content/drive/MyDrive/ocr_project/checkpoints/best_model.pt
"""

# ============================================================
# CELL 8: Download Best Checkpoint (to use locally)
# ============================================================
"""
from google.colab import files
files.download('/content/drive/MyDrive/ocr_project/checkpoints/best_model.pt')
"""

if __name__ == "__main__":
    print("=" * 60)
    print("  Google Colab Training Script")
    print("=" * 60)
    print()
    print("  This file is meant to be used in Google Colab.")
    print("  Copy each cell section into a Colab notebook cell.")
    print()
    print("  Steps:")
    print("  1. Open colab.research.google.com")
    print("  2. Create new notebook → Set GPU runtime (T4)")
    print("  3. Copy-paste cells from this file")
    print("  4. Run cells in order")
    print()
    print("  The training saves checkpoints to Google Drive,")
    print("  so you can resume if the session disconnects.")
    print("=" * 60)
