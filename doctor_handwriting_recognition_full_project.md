
# Doctor Handwriting Recognition Project
### Machine Learning Project Documentation

**Project Topic:** Recognition of Doctor's Handwritten Prescriptions  
**Language Scope:** English only  
**Recognition Level:** Word / Line level  
**Author:** Student Project  
**Date:** 2026-03-07

---

# 1. Project Overview

The goal of this project is to develop a **machine learning system capable of recognizing handwritten medical prescriptions written by doctors**.

The system will:
- Accept **images of handwritten prescriptions**
- Recognize **words or lines**
- Convert them into **machine readable text**

The focus is specifically on:

- English handwriting
- Pen-on-paper writing
- Doctor style prescription writing
- Word and line recognition

This project is strictly a **research / academic prototype** and **not intended for clinical use**.

---

# 2. Motivation

Healthcare systems still rely heavily on handwritten prescriptions.  
Manual transcription can cause errors and inefficiencies.

Optical Character Recognition (OCR) systems can automatically convert:

- handwritten text
- printed text

into digital text for storage and analysis.

However, handwritten prescription recognition remains challenging because of:

- highly variable handwriting styles
- medical abbreviations
- poor scan quality
- overlapping characters

Therefore, this project aims to explore modern machine learning methods to improve handwriting recognition for prescriptions.

---

# 3. Key Challenges

Important technical challenges include:

1. Handwriting variability between doctors
2. Image noise or low contrast
3. Cursive writing with connected characters
4. Medical abbreviations
5. Small datasets of prescription handwriting

These challenges require a pipeline that includes:

- preprocessing
- segmentation
- feature extraction
- recognition
- post‑processing

---

# 4. System Architecture

The proposed OCR pipeline:

```
Image Input
     ↓
Image Preprocessing
     ↓
Text Segmentation
     ↓
Feature Extraction
     ↓
Recognition Model
     ↓
Post Processing
     ↓
Final Text Output
```

---

# 5. Dataset Strategy

Datasets are divided into **three categories**:

1. General handwriting datasets
2. Prescription-specific datasets
3. Stress testing datasets

## 5.1 General Handwriting (Pretraining)

These teach the model how handwriting works.

### IAM Line Dataset
- ~10.4k handwritten text lines
- Train / validation / test splits

### IAM Words Dataset
- ~115k handwritten word images

These datasets help the model learn:

- character shapes
- writing styles
- spacing patterns

---

## 5.2 Prescription Specific Datasets

These datasets represent the **actual target domain**.

### Doctor Handwritten Prescription BD Dataset (Kaggle)

Word-level dataset of real prescriptions.

Typical structure:

```
train/
validation/
test/
```

Used for **fine-tuning** the model.

---

### OCR Processed Handwritten Prescriptions (Kaggle)

Dataset created from OCR outputs.

Important note:

- labels may contain noise
- used mainly for evaluation or weak supervision

---

### Medical Prescription Handwritten Words (HuggingFace)

Small dataset containing around **46 labeled medical words**.

Example words:

- Amoxicillin
- Fever
- Tablet
- Syrup

Used mainly for:

- quick testing
- sanity checks

---

## 5.3 Stress Test Dataset

### Illegible Medical Prescription Images

Contains extremely messy prescriptions.

Purpose:

- robustness testing
- evaluating model behavior in difficult cases

Not used for training.

---

# 6. Dataset Selection Rule

Recommended training hierarchy:

### Stage 1 — Pretraining

Train model on:

- IAM Line dataset
- IAM Words dataset

Goal:

Learn general handwriting patterns.

---

### Stage 2 — Domain Fine‑Tuning

Fine‑tune using:

- Doctor Prescription BD dataset

Goal:

Adapt model to medical writing styles.

---

### Stage 3 — Evaluation

Evaluate on:

- OCR processed prescription dataset
- Illegible prescriptions dataset

Goal:

Measure robustness and real‑world performance.

---

# 7. Model Architecture Options

Two main model approaches are recommended.

## 7.1 CRNN + CTC

Architecture:

```
CNN → BiLSTM → CTC Decoder
```

Advantages:

- Efficient training
- Works well for word recognition
- Lightweight enough for student projects

---

## 7.2 Transformer OCR (Donut / TrOCR)

Transformer based architecture.

Advantages:

- end‑to‑end document understanding
- strong OCR capability

But:

- heavier to train
- requires more compute

---

# 8. Preprocessing Pipeline

Before training, images are processed using:

1. grayscale conversion
2. adaptive thresholding
3. noise removal
4. skew correction
5. contrast enhancement
6. resizing

Purpose:

Improve recognition accuracy.

---

# 9. Segmentation Strategy

Segmentation splits the prescription into:

- lines
- words

Techniques used:

- connected component analysis
- contour detection
- projection profiles

If dataset already contains word crops, segmentation can be skipped.

---

# 10. Post Processing

Post processing improves predictions.

Techniques include:

- medical dictionary matching
- fuzzy string matching
- dosage pattern detection

Examples of dosage patterns:

```
500 mg
1-0-1
2 times daily
```

Low confidence predictions will be flagged for **human review**.

---

# 11. Evaluation Metrics

Key metrics used:

### CER
Character Error Rate

### WER
Word Error Rate

### Exact Match Accuracy

### Prescription Accuracy Rate

Measures correct recognition of:

- drug name
- dosage
- frequency

---

# 12. Training Environment

The project will run training **online using free GPU environments** because the development machine is a **MacBook Air**.

Local machine is used for:

- coding
- preprocessing
- debugging

Training runs on:

### Kaggle Notebooks (Recommended)

Typical resources:

| Resource | Value |
|--------|------|
GPU | NVIDIA T4 |
RAM | ~16GB |
Session | ~9 hours |

Kaggle allows training even if the laptop is closed.

---

# 13. Training Workflow

Recommended workflow:

1. Write training code locally
2. Upload datasets to Kaggle
3. Train model on GPU
4. Save checkpoints
5. Resume training if session ends

---

# 14. Checkpoint Strategy

Training scripts must include checkpoint saving.

Example:

```python
torch.save(model.state_dict(), "checkpoint.pt")
```

This allows training to resume after interruptions.

---

# 15. Implementation Timeline

## Week 1
Dataset collection and cleaning

## Week 2
Preprocessing pipeline

## Week 3
Baseline model testing

## Week 4
Model training

## Week 5
Post processing development

## Week 6
Evaluation and report writing

---

# 16. Ethical Considerations

Important limitations:

- System is **not clinically validated**
- Predictions may contain errors
- Human verification is required

Medical data must also be:

- anonymized
- privacy compliant

---

# 17. Project Resources

## Datasets

IAM Line Dataset  
https://huggingface.co/datasets/Teklia/IAM-line

IAM Words Dataset  
https://huggingface.co/datasets/priyank-m/IAM_words_text_recognition

Medical Prescription Handwritten Words  
https://huggingface.co/datasets/avi-kai/Medical_Prescription_Handwritten_Words

Doctor Handwritten Prescription BD Dataset  
https://www.kaggle.com/datasets/mamun1113/doctors-handwritten-prescription-bd-dataset

OCR Processed Prescriptions  
https://www.kaggle.com/datasets/nadaarfaoui/ocr-processed-handwritten-prescriptions

Illegible Prescription Dataset  
https://www.kaggle.com/datasets/mehaksingal/illegible-medical-prescription-images-dataset

---

# 18. Future Improvements

Potential extensions:

- full prescription parsing
- structured extraction
- multilingual support
- improved segmentation using object detection
- domain specific language models

---

# 19. Conclusion

This project proposes a machine learning system for recognizing handwritten medical prescriptions.

By combining:

- handwriting datasets
- domain-specific prescription data
- deep learning OCR models
- medical post processing

the system aims to improve recognition of doctor handwriting.

The project will run using **free GPU resources** and follow a structured training strategy for best results.

