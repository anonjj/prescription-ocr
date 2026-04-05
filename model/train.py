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
import shutil
import subprocess
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    PROJECT_ROOT, PROCESSED_DIR, CHECKPOINTS_DIR,
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
from contextlib import nullcontext

# Mixed precision support
try:
    # Modern torch.amp
    from torch.amp import autocast, GradScaler
    def get_autocast(device_type):
        if device_type == "cuda":
            return autocast(device_type="cuda")
        return nullcontext()
    def get_scaler(device_type):
        if device_type == "cuda":
            try:
                return GradScaler("cuda")
            except TypeError:
                return GradScaler()
        return None
    AMP_AVAILABLE = True
except (ImportError, AttributeError):
    try:
        # Legacy torch.cuda.amp
        from torch.cuda.amp import autocast, GradScaler
        def get_autocast(device_type):
            if device_type == "cuda":
                return autocast()
            return nullcontext()
        def get_scaler(device_type):
            if device_type == "cuda":
                return GradScaler()
            return None
        AMP_AVAILABLE = True
    except ImportError:
        def get_autocast(device_type): return nullcontext()
        def get_scaler(device_type): return None
        AMP_AVAILABLE = False


def get_device():
    """Select best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _run_git_command(args, env):
    """Run a git command and return stripped stdout."""
    result = subprocess.run(
        args,
        cwd=PROJECT_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def push_checkpoint_to_github(checkpoint_path: str, epoch_num: int, best_cer: float,
                              github_push_path: str, github_repository: str,
                              github_branch: str):
    """
    Copy the latest best checkpoint into the repo and push it to GitHub.

    This is opt-in and is intended for long-running Kaggle jobs. Failures are
    reported but should not stop training.
    """
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if not token:
        print("         ↳ GitHub push skipped: GITHUB_TOKEN/GH_TOKEN not set")
        return

    repo_name = github_repository or os.environ.get("GITHUB_REPOSITORY")
    if not repo_name:
        print("         ↳ GitHub push skipped: repository not configured")
        return

    branch_name = github_branch or os.environ.get("GITHUB_BRANCH", "main")
    commit_name = os.environ.get("GITHUB_COMMIT_NAME", "Kaggle Checkpoint Bot")
    commit_email = os.environ.get("GITHUB_COMMIT_EMAIL", "checkpoint-bot@users.noreply.github.com")

    rel_push_path = github_push_path.strip().lstrip("/")
    dest_path = os.path.join(PROJECT_ROOT, rel_push_path)
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    shutil.copy2(checkpoint_path, dest_path)

    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"

    try:
        _run_git_command(["git", "config", "user.name", commit_name], env)
        _run_git_command(["git", "config", "user.email", commit_email], env)
        _run_git_command(["git", "add", "-f", rel_push_path], env)

        staged_diff = subprocess.run(
            ["git", "diff", "--cached", "--quiet", "--", rel_push_path],
            cwd=PROJECT_ROOT,
            env=env,
        )
        if staged_diff.returncode == 0:
            print("         ↳ GitHub push skipped: checkpoint unchanged")
            return

        commit_message = f"backup: checkpoint epoch {epoch_num} cer {best_cer:.4f}"
        _run_git_command(["git", "commit", "-m", commit_message, "--", rel_push_path], env)

        push_url = f"https://x-access-token:{token}@github.com/{repo_name}.git"
        _run_git_command(["git", "push", push_url, f"HEAD:{branch_name}"], env)
        print(f"         ↳ Checkpoint pushed to GitHub: {repo_name}@{branch_name} -> {rel_push_path}")
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        if stderr:
            print(f"         ↳ GitHub push failed: {stderr.splitlines()[-1]}")
        else:
            print(f"         ↳ GitHub push failed with exit code {exc.returncode}")


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


def freeze_backbone(model):
    """Freeze the CNN backbone for prescription-specific fine-tuning."""
    for param in model.cnn.parameters():
        param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Frozen backbone: {trainable:,} / {total:,} params trainable")


def resolve_data_paths(finetune: bool = False):
    """Return train/val CSV paths for base training or fine-tuning."""
    prefix = "finetune_" if finetune else ""
    return (
        os.path.join(PROCESSED_DIR, f"{prefix}train.csv"),
        os.path.join(PROCESSED_DIR, f"{prefix}val.csv"),
    )


def train(epochs=None, batch_size=None, lr=None,
          resume_from=None, cache_tensors=False, augment=True,
          backbone=CNN_BACKBONE, seq_model=SEQ_MODEL, use_stn=USE_STN,
          use_beam=USE_BEAM_SEARCH, augment_level=AUGMENT_LEVEL,
          use_curriculum=USE_CURRICULUM, curriculum_warmup=CURRICULUM_WARMUP,
          use_synthetic=USE_SYNTHETIC_DATA, finetune=False,
          checkpoint_name: str = "best_model.pt",
          final_checkpoint_name: str = "final_model.pt",
          push_best_to_github: bool = False,
          push_after_epoch: int = 10,
          push_every_n_epochs: int = 10,
          github_push_path: str = "checkpoint_exports/best_model.pt",
          github_repository: str = None,
          github_branch: str = "main"):
    """Main training function."""
    device = get_device()
    print(f"\n  Device: {device}")

    # Mixed precision (CUDA only)
    use_amp = device.type == "cuda" and AMP_AVAILABLE
    scaler = get_scaler(device.type) if use_amp else None
    if use_amp:
        print("  Mixed precision: enabled")

    if finetune:
        epochs = epochs if epochs is not None else 50
        batch_size = batch_size if batch_size is not None else 64
        lr = lr if lr is not None else 1e-5
        patience = max(PATIENCE, 20)
        scheduler_step_size = min(LR_STEP_SIZE, 10)
        train_csv, val_csv = resolve_data_paths(finetune=True)
    else:
        epochs = epochs if epochs is not None else NUM_EPOCHS
        batch_size = batch_size if batch_size is not None else BATCH_SIZE
        lr = lr if lr is not None else LEARNING_RATE
        patience = PATIENCE
        scheduler_step_size = LR_STEP_SIZE
        train_csv, val_csv = resolve_data_paths(finetune=False)

    # ── Architecture info ──
    print(f"  Mode: {'fine-tune' if finetune else 'base training'}")
    print(f"  Backbone: {backbone}")
    print(f"  Sequence model: {seq_model}")
    print(f"  STN: {'enabled' if use_stn else 'disabled'}")
    print(f"  Decoding: {'beam search' if use_beam else 'greedy'}")
    print(f"  Augmentation: {augment_level if augment else 'disabled'}")
    print(f"  Curriculum: {'enabled (warmup={})'.format(curriculum_warmup) if use_curriculum else 'disabled'}")
    print(f"  Synthetic data: {'enabled' if use_synthetic else 'disabled'}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {lr:g}")
    print(f"  LR step size: {scheduler_step_size}")
    print(f"  Early stopping patience: {patience}")
    print(f"  Best checkpoint file: {checkpoint_name}")
    print(f"  Final checkpoint file: {final_checkpoint_name}")
    if push_best_to_github:
        print(f"  GitHub backup: enabled from epoch {push_after_epoch}")
        print(f"  GitHub backup interval: every {push_every_n_epochs} epoch(s)")
        print(f"  GitHub path: {github_push_path}")
        print(f"  GitHub branch: {github_branch}")
    else:
        print("  GitHub backup: disabled")

    # ── Load Data ──
    if not os.path.exists(train_csv) or not os.path.exists(val_csv):
        print(f"  Training data not found: {train_csv}")
        if finetune:
            print("  Run: python data/split_data.py --finetune")
        else:
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

    if finetune:
        freeze_backbone(model)
        if use_synthetic:
            print("  Fine-tune mode: synthetic data is still enabled")

    # ── Loss & Optimizer ──
    ctc_loss = nn.CTCLoss(blank=BLANK_LABEL, zero_infinity=True)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=scheduler_step_size, gamma=LR_GAMMA
    )

    start_epoch = 0
    best_cer = float("inf")
    patience_counter = 0

    # ── Resume from checkpoint ──
    if resume_from and os.path.exists(resume_from):
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        checkpoint_finetune = checkpoint.get("finetune", False)

        if finetune and not checkpoint_finetune:
            print("  Loaded pretrained checkpoint for fine-tuning")
            print("  Optimizer state reset for frozen-backbone training")
        else:
            if "optimizer_state_dict" in checkpoint:
                try:
                    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                except ValueError as exc:
                    print(f"  Optimizer state skipped: {exc}")
                else:
                    if "scheduler_state_dict" in checkpoint:
                        try:
                            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                        except ValueError as exc:
                            print(f"  Scheduler state skipped: {exc}")
                    if scaler is not None and "scaler_state_dict" in checkpoint:
                        try:
                            scaler.load_state_dict(checkpoint["scaler_state_dict"])
                        except ValueError as exc:
                            print(f"  AMP scaler state skipped: {exc}")
                    start_epoch = checkpoint.get("epoch", 0) + 1
                    best_cer = checkpoint.get("best_cer", float("inf"))
                    print(f"  Resumed from epoch {start_epoch}, best CER: {best_cer:.4f}")
            elif checkpoint.get("epoch") is not None:
                start_epoch = checkpoint.get("epoch", 0) + 1
                best_cer = checkpoint.get("best_cer", float("inf"))
    elif resume_from:
        print(f"  Resume checkpoint not found: {resume_from}")

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
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                seq_len = outputs.size(0)
                batch_size_actual = images.size(0)
                input_lengths = torch.full((batch_size_actual,), seq_len, dtype=torch.long)
                loss = ctc_loss(outputs, labels, input_lengths, label_lengths)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=5.0)
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
            ckpt_path = os.path.join(CHECKPOINTS_DIR, checkpoint_name)
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
                "best_cer": best_cer,
                "backbone": backbone,
                "seq_model": seq_model,
                "use_stn": use_stn,
                "finetune": finetune,
                "train_csv": train_csv,
                "val_csv": val_csv,
                "checkpoint_name": checkpoint_name,
                "final_checkpoint_name": final_checkpoint_name,
            }, ckpt_path)
            print(f"         ↳ Best model saved (CER: {best_cer:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n  Early stopping at epoch {epoch+1} (patience={patience})")
                break

        should_push_checkpoint = (
            push_best_to_github
            and push_every_n_epochs > 0
            and (epoch + 1) >= push_after_epoch
            and ((epoch + 1 - push_after_epoch) % push_every_n_epochs == 0)
        )
        if should_push_checkpoint:
            best_ckpt_path = os.path.join(CHECKPOINTS_DIR, checkpoint_name)
            if os.path.exists(best_ckpt_path):
                push_checkpoint_to_github(
                    checkpoint_path=best_ckpt_path,
                    epoch_num=epoch + 1,
                    best_cer=best_cer,
                    github_push_path=github_push_path,
                    github_repository=github_repository,
                    github_branch=github_branch,
                )
            else:
                print("         ↳ GitHub push skipped: best checkpoint file not found")

    # Save final checkpoint
    final_path = os.path.join(CHECKPOINTS_DIR, final_checkpoint_name)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "best_cer": best_cer,
        "backbone": backbone,
        "seq_model": seq_model,
        "use_stn": use_stn,
        "finetune": finetune,
        "train_csv": train_csv,
        "val_csv": val_csv,
        "checkpoint_name": checkpoint_name,
        "final_checkpoint_name": final_checkpoint_name,
    }, final_path)

    print(f"\n{'='*70}")
    print(f"  Training complete! Best CER: {best_cer:.4f}")
    print(f"  Best checkpoint → {os.path.join(CHECKPOINTS_DIR, checkpoint_name)}")
    print(f"  Final checkpoint → {final_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CRNN model")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--cache", action="store_true", help="Cache tensors in RAM (faster after epoch 1)")
    parser.add_argument("--no-augment", action="store_true", help="Disable training augmentation")
    parser.add_argument("--finetune", action="store_true",
                        help="Fine-tune on prescription-only splits with a frozen CNN backbone")
    parser.add_argument("--checkpoint-name", type=str, default="best_model.pt",
                        help="Filename for the best checkpoint inside models/checkpoints")
    parser.add_argument("--final-checkpoint-name", type=str, default="final_model.pt",
                        help="Filename for the final checkpoint inside models/checkpoints")
    parser.add_argument("--push-best-to-github", action="store_true",
                        help="Commit and push the best checkpoint to GitHub when it improves")
    parser.add_argument("--push-after-epoch", type=int, default=10,
                        help="Do not push checkpoints before this 1-based epoch number")
    parser.add_argument("--push-every-n-epochs", type=int, default=10,
                        help="Push the current best checkpoint every N epochs after push-after-epoch")
    parser.add_argument("--github-push-path", type=str, default="checkpoint_exports/best_model.pt",
                        help="Repo-relative destination path for the pushed checkpoint")
    parser.add_argument("--github-repository", type=str, default=None,
                        help="GitHub repo in owner/name format; falls back to GITHUB_REPOSITORY")
    parser.add_argument("--github-branch", type=str, default="main",
                        help="GitHub branch to push the checkpoint commit to")

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
          use_synthetic=args.synthetic,
          finetune=args.finetune,
          checkpoint_name=args.checkpoint_name,
          final_checkpoint_name=args.final_checkpoint_name,
          push_best_to_github=args.push_best_to_github,
          push_after_epoch=args.push_after_epoch,
          push_every_n_epochs=args.push_every_n_epochs,
          github_push_path=args.github_push_path,
          github_repository=args.github_repository,
          github_branch=args.github_branch)
