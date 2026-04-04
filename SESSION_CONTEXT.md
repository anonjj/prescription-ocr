# 🩺 Project Status: Doctor Handwriting OCR

## 📍 Current State
- **Dataset**: Integrated **~135,000 images** across IAM (Line/Word), RxHandBD, and Medical Prescriptions.
- **Model**: CRNN with **EfficientNet-B0** backbone and **BiLSTM** sequence model.
- **Last Status**: Model is training successfully with **CER dropping from 90% → 32%** by Epoch 10.

---

## 🛠 Critical Fixes Applied (DO NOT REVERT)

### 1. Image Polarity (The "White-on-Black" Rule)
Handwriting OCR is much more stable when text is high-value (white/255) and background is low-value (black/0).
- **Transforms**: `preprocessing/transforms.py` now explicitly inverts images if thcodey are dark-on-light.
- **Padding**: Changed from white (255) to black (0).
- **Augmentation**: All `albumentations` fill values set to `0`.

### 2. Data Pipeline Resilience
- **RxHandBD**: Fixed Kaggle ID to `banasmitajena/rxhandbd` and expanded `CHARS` in `config.py` to prevent these samples from being filtered out.
- **Medical Prescription**: Added JSON parsing to `download_huggingface.py` to extract text from Donut-style `ground_truth` columns.
- **Clean Manifest**: Updated to **strip** invalid characters instead of deleting rows, and allowed **single-character** labels.

### 3. Training Stability
- **Learning Rate**: Lowered to **`3e-4`** (from `1e-3`) to prevent loss divergence.
- **AMP (Mixed Precision)**: Implemented a robust `get_autocast()` wrapper in `model/train.py` to handle both old and new PyTorch versions on Kaggle.

---

## 🚀 Next Steps

### 1. Execute Background Training
To reach production quality (<10% CER), run the following in the Kaggle notebook:
1.  **Save Version** → **Save & Run All (Commit)**.
2.  Let it run for the full **12-hour background window**.
3.  The model will save `best_model.pt` automatically.

### 2. Evaluation
Once training is complete, run:
```bash
python model/evaluate.py --split test --checkpoint models/checkpoints/best_model.pt
```

### 3. Deployment
The model is ready to be loaded into the `api/server.py` or used in `demo.py`.

---

## 📝 Environment Notes
- **GPU**: Tesla T4 (Kaggle).
- **Libraries**: Requires `pyctcdecode` and `kenlm` for Beam Search decoding.
- **Data Path**: RAW data sits in `/tmp/ocr_data/raw/` (volatile across sessions).
