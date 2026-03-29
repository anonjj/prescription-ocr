
# Doctor Handwriting Recognition Project
### Machine Learning Project Documentation

**Project Topic:** Recognition of Doctor's Handwritten Prescriptions
**Language Scope:** English only
**Recognition Level:** Word / Line level
**Author:** Student Project
**Last Updated:** 2026-03-16

---

# 1. Project Overview

The goal of this project is to develop a **machine learning system capable of recognizing handwritten medical prescriptions written by doctors**.

The system:
- Accepts **images of handwritten prescriptions**
- Recognizes **words or lines** and converts them into **machine-readable text**
- Runs **fully on-device** via an Android application (no internet required)
- Extracts structured fields: dosage, frequency, duration

This project is strictly a **research / academic prototype** and **not intended for clinical use**.

---

# 2. Motivation

Healthcare systems still rely heavily on handwritten prescriptions. Manual transcription can cause errors and inefficiencies.

Handwritten prescription recognition is challenging because of:
- Highly variable handwriting styles between doctors
- Medical abbreviations and shorthand
- Poor scan or photo quality
- Overlapping and connected characters

This project explores modern deep learning methods (CRNN + CTC) to address these challenges.

---

# 3. System Architecture

The full pipeline:

```
Image Input  (camera or gallery)
      ↓
Image Preprocessing
  • Grayscale conversion
  • Denoising (median blur)
  • Contrast enhancement (CLAHE)
  • Adaptive binarisation
  • Deskew
  • Resize + pad to 64 × 256
      ↓
CRNN Model  (CNN → BiLSTM → CTC)
      ↓
CTC Greedy Decode
      ↓
Post-Processing
  • Fuzzy medical dictionary correction
  • Dosage / frequency / duration extraction
  • Confidence scoring + review flag
      ↓
Final Text Output
```

---

# 4. Model Architecture — CRNN + CTC

The implemented model is a **Convolutional Recurrent Neural Network (CRNN)** trained with **CTC (Connectionist Temporal Classification)** loss.

```
CNN Feature Extractor  →  BiLSTM Sequence Model  →  Linear + Log-Softmax  →  CTC
```

### CNN Backbone

| Block | Input → Output | Operation |
|-------|---------------|-----------|
| Block 1 | 1 → 64 | Conv3×3, BN, ReLU, MaxPool 2×2 |
| Block 2 | 64 → 128 | Conv3×3, BN, ReLU, MaxPool 2×2 |
| Block 3 | 128 → 256 | 2× Conv3×3, BN, ReLU, MaxPool 2×1 |
| Block 4 | 256 → 512 | 2× Conv3×3, BN, ReLU, MaxPool 2×1 |
| Block 5 | 512 → 512 | Conv3×3, BN, ReLU, AdaptiveAvgPool → height 1 |

### BiLSTM

- Hidden size: 256 (×2 bidirectional = 512)
- Layers: 2
- Dropout: 0.3

### Output

- Linear: 512 → NUM\_CLASSES (79 characters + CTC blank)
- Log-softmax over character dimension

### Character Set

```
ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 .,-/'()+
```

---

# 5. Training

### Environment

Training was performed on **Kaggle** (NVIDIA P100 GPU) due to the MacBook Air development machine lacking a GPU.

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Batch size | 64 |
| Learning rate | 1e-3 |
| LR decay | ×0.1 every 25 epochs |
| Max epochs | 100 |
| Early stopping patience | 15 |
| Optimiser | Adam |
| Loss | CTC |

### Results

| Metric | Value |
|--------|-------|
| Target CER | **< 0.15** (under 15%) |

CER is targeted to be under 15% using the new EfficientNet + STN architecture + medical LM beam search.

---

# 6. Preprocessing Pipeline

Implemented in `preprocessing/transforms.py`:

| Step | Function | Purpose |
|------|----------|---------|
| Grayscale | `to_grayscale` | Remove colour information |
| Denoise | `denoise` | Median blur to remove scan noise |
| Contrast | `enhance_contrast` | CLAHE for adaptive contrast |
| Binarise | `adaptive_threshold` | Adaptive Gaussian thresholding |
| Deskew | `deskew` | Correct rotation using `minAreaRect` |
| Resize | `resize_pad` | Scale to 64 px height, pad width to 256 px |

**Input:** any image file
**Output:** `numpy.ndarray` of shape `(64, 256)`, `uint8`

---

# 7. Post-Processing

Implemented across three modules:

### `postprocessing/lexicon.py`
- Fuzzy-matches each recognised word against a medical dictionary
- Uses RapidFuzz with an 80-point similarity threshold

### `postprocessing/rules.py`
- Regex-based extraction of:
  - **Dosage** (e.g. `500 mg`, `1 tablet`)
  - **Frequency** (e.g. `twice daily`, `1-0-1`)
  - **Duration** (e.g. `7 days`, `2 weeks`)

### `postprocessing/confidence.py`
- Computes a confidence score from the model's log-probabilities
- Flags predictions below 0.6 as `NEEDS_REVIEW`

---

# 8. Dataset Strategy

Training used a combination of:

| Dataset | Source | Purpose |
|---------|--------|---------|
| IAM Line | HuggingFace — Teklia/IAM-line | General handwriting pretraining |
| Medical Prescription Words | HuggingFace — avi-kai | Domain vocabulary |
| Doctor Prescription BD | Kaggle — mamun1113 | Prescription fine-tuning |
| OCR Processed Prescriptions | Kaggle — nadaarfaoui | Additional training data |

Split: **70% train / 15% val / 15% test**

---

# 9. Android Application

A fully offline Android app (`android/`) built in Kotlin.

### Features
- Pick an image from the **gallery**
- Capture a photo with the **camera**
- On-device inference using **PyTorch Mobile** (no server, no internet)
- Displays recognised text and confidence

### On-Device Inference

The trained model is exported to TorchScript Lite Interpreter format (`.ptl`) and bundled inside the APK.

**Export command (run once after training):**
```bash
python3 export_mobile_model.py
# Then copy:
cp models/model.ptl android/app/src/main/assets/model.ptl
```

### Android Preprocessing (mirrors Python pipeline)

Since OpenCV is not available on Android, the preprocessing is reimplemented in Kotlin:
1. Extract per-pixel luminance from the Bitmap
2. Otsu's threshold to binarise (matches `adaptive_threshold` in training)
3. Scale to 64 px height, pad width to 256 px with white
4. Normalise to `[0, 1]`

### Key Files

| File | Purpose |
|------|---------|
| `MainActivity.kt` | All UI logic, preprocessing, inference, CTC decode |
| `activity_main.xml` | Layout: image preview, buttons, result card |
| `AndroidManifest.xml` | Permissions: camera, storage |
| `app/build.gradle` | Dependencies: PyTorch Mobile, Coil |
| `res/xml/file_paths.xml` | FileProvider paths for camera output |

### Dependencies

| Library | Version | Purpose |
|---------|---------|---------|
| `pytorch_android_lite` | 2.1.0 | On-device model inference |
| `coil` | 2.5.0 | Image loading / preview |
| Material Components | 1.11.0 | UI styling |

---

# 10. REST API (Optional — for laptop-based testing)

A Flask server (`api/server.py`) wraps the same inference pipeline for testing from a browser or other devices on the same network.

```bash
python3 api/server.py --port 5001
```

**Endpoints:**

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Check if server and model are ready |
| POST | `/predict` | Send image (multipart), receive JSON result |

**Response format:**
```json
{
  "raw_text":       "amoxicillin 500 mg",
  "corrected_text": "Amoxicillin 500 mg",
  "confidence":     0.74,
  "needs_review":   false,
  "dosage":         ["500 mg"],
  "frequency":      [],
  "duration":       []
}
```

---

# 11. Project Structure

```
Projecat/
├── config.py                     Central configuration (paths, hyperparameters)
├── export_mobile_model.py        Export CRNN → model.ptl for Android
├── demo.py                       CLI demo (image or webcam)
│
├── model/
│   ├── crnn.py                   CRNN architecture
│   └── utils.py                  Label encode/decode, CER, WER
│
├── preprocessing/
│   └── transforms.py             Full image preprocessing pipeline
│
├── postprocessing/
│   ├── lexicon.py                Fuzzy medical dictionary correction
│   ├── rules.py                  Dosage/frequency/duration extraction
│   └── confidence.py             Confidence scoring
│
├── data/
│   ├── download_all.py           Download all datasets
│   └── split_data.py             Train/val/test split
│
├── baselines/
│   └── run_hf_baseline.py        HuggingFace TrOCR baseline comparison
│
├── api/
│   ├── server.py                 Flask REST API
│   └── requirements.txt          API dependencies
│
├── android/                      Android app (Kotlin)
│   ├── app/src/main/
│   │   ├── java/com/ocr/prescriptionocr/MainActivity.kt
│   │   ├── res/layout/activity_main.xml
│   │   ├── res/values/strings.xml
│   │   ├── res/values/themes.xml
│   │   ├── res/xml/file_paths.xml
│   │   └── assets/model.ptl      (copy here after export)
│   ├── app/build.gradle
│   └── settings.gradle
│
├── models/
│   ├── checkpoints/best_model.pt  Trained checkpoint
│   └── model.ptl                  Mobile-exported model
│
└── notebooks/
    └── train_colab.py             Colab training script
```

---

# 12. Evaluation Metrics

| Metric | Description |
|--------|-------------|
| CER | Character Error Rate = edit\_distance(pred, target) / len(target) |
| WER | Word Error Rate = edit\_distance(pred\_words, target\_words) / len(target\_words) |

**Target:** CER **<0.15** on test set.

---

# 13. Ethical Considerations

- System is **not clinically validated**
- Predictions may contain errors — always require human verification
- Medical datasets must be anonymised and privacy-compliant
- Not suitable for real prescription dispensing

---

# 14. Future Improvements

- Fine-tune on more prescription-specific data to reduce CER below 15%
- Replace CRNN with TrOCR (transformer-based) for higher accuracy
- Add full prescription parsing (drug name, patient info, doctor signature)
- Multilingual support
- On-device post-processing (medical dictionary bundled in APK)
