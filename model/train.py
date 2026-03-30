"""
Training loop for CRNN + CTC handwriting recognition.
Supports CPU, CUDA, and MPS (Apple Silicon).

Features:
  - Configurable backbone (VGG / EfficientNet)
  - Configurable sequence model (BiLSTM / Transformer)
  - Optional STN (Spatial Transformer Network)
  - Beam search or greedy decoding for validation
  - Curriculum learning
  - Strong augmentation
"""
import os
import sys
import time
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    PROCESSED_DIR, CHECKPOINTS_DIR,
    BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, PATIENCE,
    LR_STEP_SIZE, LR_GAMMA, NUM_WORKERS, BLANK_LABEL,
    CNN_BACKBONE, SEQ_MODEL, USE_STN,
    USE_BEAM_SEARCH, AUGMENT_LEVEL,
    USE_CURRICULUM, CURRICULUM_WARMUP,
    USE_SYNTHETIC_DATA,
)
from model.crnn import CRNN, count_parameters
from model.dataset import HandwritingDataset, CurriculumSampler, collate_fn
from model.utils import decode_prediction, smart_decode, compute_cer, compute_wer

# Mixed precision support (CUDA only)
try:
    from torch.cuda.amp import autocast, GradScaler
    AMP_AVAILABLE = True
except ImportError:
    AMP_AVAILABLE = False


def get_device():
    """Select best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def validate(model, val_loader, ctc_loss, device, use_amp=False,
             use_beam=USE_BEAM_SEARCH):
    """Run validation and compute loss + CER + WER."""
    model.eval()
    total_loss = 0.0
    total_cer = 0.0
    total_wer = 0.0
    n_samples = 0

    with torch.no_grad():
        for images, labels, label_lengths, texts in val_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if use_amp and AMP_AVAILABLE:
                with get_autocast(device.type):
                    outputs = model(images)
                    seq_len = outputs.size(0)
                    batch_size = images.size(0)
                    input_lengths = torch.full((batch_size,), seq_len, dtype=torch.long, device=device)
                    loss = ctc_loss(outputs, labels, input_lengths, label_lengths.to(device))
            else:
                outputs = model(images)
                seq_len = outputs.size(0)
                batch_size = images.size(0)
                input_lengths = torch.full((batch_size,), seq_len, dtype=torch.long)
                loss = ctc_loss(outputs, labels, input_lengths, label_lengths)

            total_loss += loss.item() * batch_size

            # Decode predictions
            if use_beam:
                # Beam search: need per-sample log probs
                outputs_np = outputs.permute(1, 0, 2)  # (B, seq_len, classes)
                for i in range(batch_size):
                    pred_text = smart_decode(
                        log_probs=outputs_np[i], use_beam=True
                    )
                    target_text = texts[i]
                    total_cer += compute_cer(pred_text, target_text)
                    total_wer += compute_wer(pred_text, target_text)
                    n_samples += 1
            else:
                # Greedy decoding
                _, preds = outputs.max(2)  # (seq_len, batch)
                preds = preds.permute(1, 0).cpu()  # (batch, seq_len)

                for i in range(batch_size):
                    pred_text = decode_prediction(preds[i].tolist())
                    target_text = texts[i]
                    total_cer += compute_cer(pred_text, target_text)
                    total_wer += compute_wer(pred_text, target_text)
                    n_samples += 1

    avg_loss = total_loss / max(n_samples, 1)
    avg_cer = total_cer / max(n_samples, 1)
    avg_wer = total_wer / max(n_samples, 1)

    return avg_loss, avg_cer, avg_wer


def train(epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, lr=LEARNING_RATE,
          resume_from=None, cache_tensors=False, augment=True,
          backbone=CNN_BACKBONE, seq_model=SEQ_MODEL, use_stn=USE_STN,
          use_beam=USE_BEAM_SEARCH, augment_level=AUGMENT_LEVEL,
          use_curriculum=USE_CURRICULUM, curriculum_warmup=CURRICULUM_WARMUP,
          use_synthetic=USE_SYNTHETIC_DATA):
    """Main training function."""
    device = get_device()
    print(f"\n  Device: {device}")

    # Mixed precision (CUDA only)
    use_amp = device.type == "cuda" and AMP_AVAILABLE
    scaler = GradScaler() if use_amp else None
    if use_amp:
        print("  Mixed precision: enabled")

    # ── Architecture info ──
    print(f"  Backbone: {backbone}")
    print(f"  Sequence model: {seq_model}")
    print(f"  STN: {'enabled' if use_stn else 'disabled'}")
    print(f"  Decoding: {'beam search' if use_beam else 'greedy'}")
    print(f"  Augmentation: {augment_level if augment else 'disabled'}")
    print(f"  Curriculum: {'enabled (warmup={})'.format(curriculum_warmup) if use_curriculum else 'disabled'}")
    print(f"  Synthetic data: {'enabled' if use_synthetic else 'disabled'}")

    # ── Load Data ──
    train_csv = os.path.join(PROCESSED_DIR, "train.csv")
    val_csv = os.path.join(PROCESSED_DIR, "val.csv")

    if not os.path.exists(train_csv):
        print(f"  Training data not found: {train_csv}")
        print("  Run the data pipeline first.")
        return

    train_dataset = HandwritingDataset(
        train_csv, full_pipeline=True,
        cache_tensors=cache_tensors, augment=augment,
        augment_level=augment_level,
        include_synthetic=use_synthetic,
    )
    val_dataset = HandwritingDataset(
        val_csv, full_pipeline=True, cache_tensors=cache_tensors
    )

    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples:   {len(val_dataset)}")

    if cache_tensors:
        cache_mb = (len(train_dataset) + len(val_dataset)) * 64 / 1024
        print(f"  Tensor caching: enabled (~{cache_mb:.1f} MB)")

    # Sort by difficulty for curriculum learning
    if use_curriculum:
        train_dataset.sort_by_difficulty()
        print(f"  Dataset sorted by label length for curriculum learning")

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=NUM_WORKERS, pin_memory=True
    )

    # ── Model ──
    model = CRNN(
        backbone=backbone, seq_model=seq_model, use_stn=use_stn
    ).to(device)
    print(f"  Parameters: {count_parameters(model):,}")

    # ── Loss & Optimizer ──
    ctc_loss = nn.CTCLoss(blank=BLANK_LABEL, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA)

    start_epoch = 0
    best_cer = float("inf")
    patience_counter = 0

    # ── Resume from checkpoint ──
    if resume_from and os.path.exists(resume_from):
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_cer = checkpoint.get("best_cer", float("inf"))
        print(f"  Resumed from epoch {start_epoch}, best CER: {best_cer:.4f}")

    # ── Training Loop ──
    print(f"\n{'='*70}")
    print(f"  {'Epoch':>5} | {'Train Loss':>11} | {'Val Loss':>9} | {'CER':>7} | {'WER':>7} | {'Time':>6}")
    print(f"{'='*70}")

    for epoch in range(start_epoch, epochs):
        model.train()
        total_train_loss = 0.0
        n_batches = 0
        t0 = time.time()

        # Create DataLoader with curriculum sampler or regular shuffle
        if use_curriculum:
            sampler = CurriculumSampler(
                train_dataset, epoch=epoch,
                warmup_epochs=curriculum_warmup
            )
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, sampler=sampler,
                collate_fn=collate_fn, num_workers=NUM_WORKERS, pin_memory=True
            )
        else:
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True,
                collate_fn=collate_fn, num_workers=NUM_WORKERS, pin_memory=True
            )

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False)
        for images, labels, label_lengths, texts in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with get_autocast(device.type):
                    outputs = model(images)
                    seq_len = outputs.size(0)
                    batch_size_actual = images.size(0)
                    input_lengths = torch.full((batch_size_actual,), seq_len, dtype=torch.long, device=device)
                    loss = ctc_loss(outputs, labels, input_lengths, label_lengths.to(device))

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                seq_len = outputs.size(0)
                batch_size_actual = images.size(0)
                input_lengths = torch.full((batch_size_actual,), seq_len, dtype=torch.long)
                loss = ctc_loss(outputs, labels, input_lengths, label_lengths)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            total_train_loss += loss.item()
            n_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        scheduler.step()
        avg_train_loss = total_train_loss / max(n_batches, 1)

        # Validate
        val_loss, val_cer, val_wer = validate(
            model, val_loader, ctc_loss, device, use_amp, use_beam=use_beam
        )
        elapsed = time.time() - t0

        print(f"  {epoch+1:>5} | {avg_train_loss:>11.4f} | {val_loss:>9.4f} | {val_cer:>6.4f} | {val_wer:>6.4f} | {elapsed:>5.1f}s")

        # ── Save best checkpoint ──
        if val_cer < best_cer:
            best_cer = val_cer
            patience_counter = 0
            ckpt_path = os.path.join(CHECKPOINTS_DIR, "best_model.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_cer": best_cer,
                "backbone": backbone,
                "seq_model": seq_model,
                "use_stn": use_stn,
            }, ckpt_path)
            print(f"         ↳ Best model saved (CER: {best_cer:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\n  Early stopping at epoch {epoch+1} (patience={PATIENCE})")
                break

    # Save final checkpoint
    final_path = os.path.join(CHECKPOINTS_DIR, "final_model.pt")
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_cer": best_cer,
        "backbone": backbone,
        "seq_model": seq_model,
        "use_stn": use_stn,
    }, final_path)

    print(f"\n{'='*70}")
    print(f"  Training complete! Best CER: {best_cer:.4f}")
    print(f"  Best checkpoint → {os.path.join(CHECKPOINTS_DIR, 'best_model.pt')}")
    print(f"{'='*70}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CRNN model")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--cache", action="store_true", help="Cache tensors in RAM (faster after epoch 1)")
    parser.add_argument("--no-augment", action="store_true", help="Disable training augmentation")

    # Architecture selection
    parser.add_argument("--backbone", type=str, default=CNN_BACKBONE,
                        choices=["vgg", "efficientnet"],
                        help="CNN backbone (default: vgg)")
    parser.add_argument("--seq-model", type=str, default=SEQ_MODEL,
                        choices=["bilstm", "transformer"],
                        help="Sequence model (default: bilstm)")
    parser.add_argument("--stn", action="store_true", default=USE_STN,
                        help="Enable Spatial Transformer Network")

    # Decoding
    parser.add_argument("--beam", action="store_true", default=USE_BEAM_SEARCH,
                        help="Use beam search decoding for validation")
    parser.add_argument("--greedy", action="store_true",
                        help="Force greedy decoding for validation")

    # Augmentation
    parser.add_argument("--augment-level", type=str, default=AUGMENT_LEVEL,
                        choices=["none", "light", "strong"],
                        help="Augmentation level (default: strong)")

    # Curriculum learning
    parser.add_argument("--curriculum", action="store_true", default=USE_CURRICULUM,
                        help="Enable curriculum learning")
    parser.add_argument("--curriculum-warmup", type=int, default=CURRICULUM_WARMUP,
                        help="Curriculum warmup epochs (default: 20)")

    # Synthetic data
    parser.add_argument("--synthetic", action="store_true", default=USE_SYNTHETIC_DATA,
                        help="Include synthetic training data")

    args = parser.parse_args()

    # --greedy overrides --beam
    use_beam = args.beam and not args.greedy

    train(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
          resume_from=args.resume, cache_tensors=args.cache,
          augment=not args.no_augment,
          backbone=args.backbone, seq_model=args.seq_model, use_stn=args.stn,
          use_beam=use_beam, augment_level=args.augment_level,
          use_curriculum=args.curriculum,
          curriculum_warmup=args.curriculum_warmup,
          use_synthetic=args.synthetic)
ot args.no_augment,
          backbone=args.backbone, seq_model=args.seq_model, use_stn=args.stn,
          use_beam=use_beam, augment_level=args.augment_level,
          use_curriculum=args.curriculum,
          curriculum_warmup=args.curriculum_warmup,
          use_synthetic=args.synthetic)
