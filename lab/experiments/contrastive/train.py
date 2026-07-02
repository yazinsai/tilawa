"""
Training script for QuranCLAP contrastive model.

Two-phase training:
  Phase 1: Freeze audio/text encoders, train only projection heads (5 epochs)
  Phase 2: Unfreeze last 2 layers of each encoder, fine-tune (10 epochs)

Uses InfoNCE (CLIP-style) contrastive loss with learnable temperature.
"""

import sys
import os
import json
import time
import math
from pathlib import Path
from datetime import datetime

# Force unbuffered output
os.environ["PYTHONUNBUFFERED"] = "1"

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from model import QuranCLAP, contrastive_loss, get_tokenizer
from dataset import get_dataloaders, collate_fn

EXPERIMENT_DIR = Path(__file__).resolve().parent
CHECKPOINTS_DIR = EXPERIMENT_DIR / "checkpoints"
LOGS_DIR = EXPERIMENT_DIR / "logs"


def get_device():
    """Get best available device."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_epoch(model, tokenizer, dataloader, optimizer, device, epoch, max_text_len=64):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        audio = batch["audio"].to(device)
        texts = batch["text"]

        # Tokenize text
        text_enc = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_text_len,
            return_tensors="pt",
        )
        input_ids = text_enc["input_ids"].to(device)
        text_attention_mask = text_enc["attention_mask"].to(device)

        # Forward
        logits_a, logits_t = model(
            audio_values=audio,
            input_ids=input_ids,
            text_attention_mask=text_attention_mask,
        )

        loss = contrastive_loss(logits_a, logits_t)

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Free MPS memory
        if device.type == "mps":
            torch.mps.empty_cache()

        total_loss += loss.item()
        num_batches += 1

        if (batch_idx + 1) % 10 == 0:
            print(f"  Epoch {epoch}, Batch {batch_idx + 1}/{len(dataloader)}: loss={loss.item():.4f}, "
                  f"temp={model.logit_scale.exp().item():.2f}")

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def validate(model, tokenizer, dataloader, device, max_text_len=64):
    """Run validation and compute loss + batch accuracy."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    num_batches = 0

    for batch in dataloader:
        audio = batch["audio"].to(device)
        texts = batch["text"]

        text_enc = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_text_len,
            return_tensors="pt",
        )
        input_ids = text_enc["input_ids"].to(device)
        text_attention_mask = text_enc["attention_mask"].to(device)

        logits_a, logits_t = model(
            audio_values=audio,
            input_ids=input_ids,
            text_attention_mask=text_attention_mask,
        )

        loss = contrastive_loss(logits_a, logits_t)
        total_loss += loss.item()

        # Batch accuracy: does the audio match the correct text within the batch?
        preds = logits_a.argmax(dim=-1)
        labels = torch.arange(len(texts), device=device)
        total_correct += (preds == labels).sum().item()
        total_samples += len(texts)
        num_batches += 1

        # Free MPS memory
        if device.type == "mps":
            torch.mps.empty_cache()

    avg_loss = total_loss / max(num_batches, 1)
    accuracy = total_correct / max(total_samples, 1)
    return avg_loss, accuracy


def train(
    source: str = "everyayah",
    batch_size: int = 4,
    phase1_epochs: int = 5,
    phase2_epochs: int = 10,
    phase1_lr: float = 1e-3,
    phase2_lr: float = 1e-4,
    embed_dim: int = 256,
    max_samples: int = 500,
    save_every: int = 5,
    device_override: str = None,
    max_surah: int = None,
):
    """
    Full training pipeline.

    Phase 1: Frozen encoders, train projection heads.
    Phase 2: Unfreeze last 2 layers, fine-tune everything.
    """
    device = torch.device(device_override) if device_override else get_device()
    print(f"Using device: {device}")

    # Create directories
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"\nLoading data (source={source}, max_surah={max_surah})...")
    train_loader, val_loader, test_loader = get_dataloaders(
        source=source,
        batch_size=batch_size,
        max_samples_per_split=max_samples,
        max_surah=max_surah,
    )
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Load model
    print("\nLoading QuranCLAP model...")
    model = QuranCLAP(
        embed_dim=embed_dim,
        freeze_audio=True,
        freeze_text=True,
    )
    model = model.to(device)
    print(f"Total parameters: {model.get_total_params():,}")
    print(f"Trainable parameters (Phase 1): {model.get_trainable_params():,}")

    # Load tokenizer
    tokenizer = get_tokenizer()

    # Training log
    training_log = {
        "config": {
            "source": source,
            "batch_size": batch_size,
            "phase1_epochs": phase1_epochs,
            "phase2_epochs": phase2_epochs,
            "phase1_lr": phase1_lr,
            "phase2_lr": phase2_lr,
            "embed_dim": embed_dim,
            "device": str(device),
        },
        "epochs": [],
    }

    # =========================================================================
    # Phase 1: Train projection heads only
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"Phase 1: Training projection heads ({phase1_epochs} epochs)")
    print(f"{'='*60}")

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=phase1_lr,
        weight_decay=0.01,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=phase1_epochs, eta_min=phase1_lr * 0.01)

    best_val_loss = float("inf")

    for epoch in range(1, phase1_epochs + 1):
        t0 = time.time()

        train_loss = train_epoch(model, tokenizer, train_loader, optimizer, device, epoch)
        val_loss, val_acc = validate(model, tokenizer, val_loader, device)
        scheduler.step()

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch}/{phase1_epochs}: "
              f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
              f"val_batch_acc={val_acc:.2%}, lr={lr:.6f}, time={elapsed:.1f}s")

        training_log["epochs"].append({
            "phase": 1,
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_batch_acc": val_acc,
            "lr": lr,
            "time_s": elapsed,
        })

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), CHECKPOINTS_DIR / "best_phase1.pt")
            print(f"  --> Saved best Phase 1 model (val_loss={val_loss:.4f})")

        # Periodic checkpoint
        if epoch % save_every == 0:
            torch.save(model.state_dict(), CHECKPOINTS_DIR / f"phase1_epoch{epoch}.pt")

    # Save final Phase 1
    torch.save(model.state_dict(), CHECKPOINTS_DIR / "phase1_final.pt")

    # =========================================================================
    # Phase 2: Unfreeze last 2 encoder layers and fine-tune
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"Phase 2: Fine-tuning with unfrozen encoders ({phase2_epochs} epochs)")
    print(f"{'='*60}")

    model.unfreeze_audio_encoder(last_n_layers=2)
    model.unfreeze_text_encoder(last_n_layers=2)
    print(f"Trainable parameters (Phase 2): {model.get_trainable_params():,}")

    optimizer = AdamW(
        [
            {"params": [p for n, p in model.named_parameters()
                        if p.requires_grad and "proj" in n], "lr": phase2_lr},
            {"params": [p for n, p in model.named_parameters()
                        if p.requires_grad and "proj" not in n], "lr": phase2_lr * 0.1},
        ],
        weight_decay=0.01,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=phase2_epochs, eta_min=phase2_lr * 0.01)

    for epoch in range(1, phase2_epochs + 1):
        t0 = time.time()

        train_loss = train_epoch(model, tokenizer, train_loader, optimizer, device, epoch)
        val_loss, val_acc = validate(model, tokenizer, val_loader, device)
        scheduler.step()

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch}/{phase2_epochs}: "
              f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
              f"val_batch_acc={val_acc:.2%}, lr={lr:.6f}, time={elapsed:.1f}s")

        training_log["epochs"].append({
            "phase": 2,
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_batch_acc": val_acc,
            "lr": lr,
            "time_s": elapsed,
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), CHECKPOINTS_DIR / "best_model.pt")
            print(f"  --> Saved best model (val_loss={val_loss:.4f})")

        if epoch % save_every == 0:
            torch.save(model.state_dict(), CHECKPOINTS_DIR / f"phase2_epoch{epoch}.pt")

    # Save final model
    torch.save(model.state_dict(), CHECKPOINTS_DIR / "final_model.pt")

    # Save training log
    log_path = LOGS_DIR / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(log_path, "w") as f:
        json.dump(training_log, f, indent=2)
    print(f"\nTraining log saved to: {log_path}")

    # Final summary
    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved in: {CHECKPOINTS_DIR}")
    print(f"{'='*60}")

    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train QuranCLAP model")
    parser.add_argument("--source", default="everyayah", choices=["everyayah", "quran_md"])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--phase1-epochs", type=int, default=5)
    parser.add_argument("--phase2-epochs", type=int, default=10)
    parser.add_argument("--phase1-lr", type=float, default=1e-3)
    parser.add_argument("--phase2-lr", type=float, default=1e-4)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--device", default=None, help="Force device (cpu, mps, cuda)")
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--max-surah", type=int, default=None, help="Limit to surahs 1..N")
    args = parser.parse_args()

    train(
        source=args.source,
        batch_size=args.batch_size,
        phase1_epochs=args.phase1_epochs,
        phase2_epochs=args.phase2_epochs,
        phase1_lr=args.phase1_lr,
        phase2_lr=args.phase2_lr,
        embed_dim=args.embed_dim,
        max_samples=args.max_samples,
        device_override=args.device,
        save_every=args.save_every,
        max_surah=args.max_surah,
    )
