"""FastConformer phoneme CTC training — AGENT EDITS THIS FILE.

Each run starts from the pretrained base model and trains for a fixed
time budget. The agent modifies hyperparameters, architecture choices,
and augmentation settings to maximize exported-model accuracy on the
non-streaming verse recognition benchmark.

Time budget: 5 minutes wall-clock (300 seconds).
"""

import json
import time
import sys
from pathlib import Path

# ============================================================
# HYPERPARAMETERS — Agent edits these
# ============================================================

# Optimizer
LEARNING_RATE = 1e-4
WARMUP_STEPS = 100
WEIGHT_DECAY = 0.0

# Batching
TRAIN_BATCH_SIZE = 32
GRAD_ACCUMULATION = 2
NUM_WORKERS = 8

# Architecture
FREEZE_ENCODER_LAYERS = 10  # 0-17
FREEZE_PREPROCESSOR = True
DROPOUT = 0.0  # applied to encoder output

# Augmentation
AUGMENTATION = {
    "speed": {"prob": 0.3, "sr": 16000, "resample_type": "kaiser_fast",
              "min_speed_rate": 0.9, "max_speed_rate": 1.1, "num_rates": 5},
    "gain": {"prob": 0.3, "min_gain_dbfs": -10, "max_gain_dbfs": 5},
    "white_noise": {"prob": 0.3, "min_level": -80, "max_level": -50},
    "shift": {"prob": 0.2, "min_shift_ms": -200.0, "max_shift_ms": 200.0},
    "silence": {"prob": 0.2, "min_start_silence_secs": 0.0, "max_start_silence_secs": 0.4,
                 "min_end_silence_secs": 0.0, "max_end_silence_secs": 0.5},
}

# Training schedule
VAL_CHECK_INTERVAL = 50
EARLY_STOPPING_PATIENCE = 4

# Time budget
TIME_BUDGET_SECONDS = 300  # 5 minutes

# ============================================================
# TRAINING LOGIC — Agent can also edit this section
# ============================================================

def train():
    """Run one training experiment. Returns best val_loss."""
    import torch
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, Timer
    from omegaconf import open_dict

    sys.path.insert(0, str(Path(__file__).parent))
    from prepare import setup_model, DATA_DIR

    start_time = time.time()

    model = setup_model(
        freeze_encoder_layers=FREEZE_ENCODER_LAYERS,
        freeze_preprocessor=FREEZE_PREPROCESSOR,
    )

    train_manifest = Path(DATA_DIR) / "manifests" / "train_manifest.jsonl"
    val_manifest = Path(DATA_DIR) / "manifests" / "val_manifest.jsonl"

    with open_dict(model.cfg):
        model.cfg.train_ds.manifest_filepath = str(train_manifest)
        model.cfg.train_ds.is_tarred = False
        model.cfg.train_ds.tarred_audio_filepaths = None
        model.cfg.train_ds.batch_size = TRAIN_BATCH_SIZE
        model.cfg.train_ds.num_workers = NUM_WORKERS
        model.cfg.train_ds.shuffle = True
        model.cfg.train_ds.pin_memory = True
        model.cfg.train_ds.max_duration = 30.0
        model.cfg.train_ds.min_duration = 0.3
        model.cfg.train_ds.shuffle_n = 2048
        model.cfg.train_ds.use_start_end_token = False
        model.cfg.train_ds.augmentor = AUGMENTATION

        model.cfg.validation_ds.manifest_filepath = str(val_manifest)
        model.cfg.validation_ds.batch_size = TRAIN_BATCH_SIZE
        model.cfg.validation_ds.num_workers = NUM_WORKERS
        model.cfg.validation_ds.shuffle = False
        model.cfg.validation_ds.pin_memory = True
        model.cfg.validation_ds.max_duration = 30.0
        model.cfg.validation_ds.min_duration = 0.3
        model.cfg.validation_ds.use_start_end_token = False

        model.cfg.test_ds.manifest_filepath = str(val_manifest)
        model.cfg.test_ds.batch_size = TRAIN_BATCH_SIZE
        model.cfg.test_ds.shuffle = False
        model.cfg.test_ds.num_workers = NUM_WORKERS
        model.cfg.test_ds.pin_memory = True
        model.cfg.test_ds.use_start_end_token = False

        model.cfg.tokenizer.dir = "/tmp"

        model.cfg.optim.lr = LEARNING_RATE
        model.cfg.optim.weight_decay = WEIGHT_DECAY
        if "sched" in model.cfg.optim and "warmup_steps" in model.cfg.optim.sched:
            model.cfg.optim.sched.warmup_steps = WARMUP_STEPS

    # Apply dropout to encoder output
    if DROPOUT > 0 and hasattr(model, 'encoder'):
        import torch.nn as nn
        # NeMo FastConformer exposes final_dropout on the encoder
        if hasattr(model.encoder, 'final_dropout'):
            model.encoder.final_dropout = nn.Dropout(DROPOUT)
        # Also try per-layer dropout if available
        if hasattr(model.encoder, 'layers'):
            for layer in model.encoder.layers:
                if hasattr(layer, 'dropout'):
                    layer.dropout.p = DROPOUT

    timer = Timer(duration={"seconds": TIME_BUDGET_SECONDS})
    checkpoint_dir = Path("/tmp/autoresearch_checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)

    checkpoint_cb = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename="phoneme-{step:06d}-{val_loss:.4f}",
        monitor="val_loss", mode="min", save_top_k=1,
        every_n_train_steps=VAL_CHECK_INTERVAL, save_last=True,
    )

    trainer = pl.Trainer(
        accelerator="gpu", devices=1,
        max_steps=99999,  # timer enforces budget
        precision="bf16-mixed",
        gradient_clip_val=1.0,
        accumulate_grad_batches=GRAD_ACCUMULATION,
        log_every_n_steps=10,
        val_check_interval=VAL_CHECK_INTERVAL,
        num_sanity_val_steps=0,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=EARLY_STOPPING_PATIENCE, mode="min"),
            checkpoint_cb,
            timer,
        ],
    )

    model.set_trainer(trainer)
    model.setup_training_data(model.cfg.train_ds)
    model.setup_validation_data(model.cfg.validation_ds)

    trainer.fit(model)

    best_score = checkpoint_cb.best_model_score
    val_loss = float(best_score) if best_score is not None else float("inf")
    elapsed = time.time() - start_time

    # Reload best checkpoint weights and save as .nemo for export
    # (trainer.fit() leaves the model at the LAST step, not the best)
    best_ckpt = checkpoint_cb.best_model_path
    if best_ckpt and Path(best_ckpt).exists():
        import torch
        ckpt = torch.load(best_ckpt, map_location="cuda", weights_only=False)
        model.load_state_dict(ckpt["state_dict"], strict=False)
        print(f"Reloaded best checkpoint: {best_ckpt}")

    nemo_path = checkpoint_dir / "model.nemo"
    model.save_to(str(nemo_path))

    result = {
        "val_loss": val_loss,
        "elapsed_seconds": elapsed,
        "steps": trainer.global_step,
        "nemo_path": str(nemo_path),
        "best_checkpoint": str(best_ckpt) if best_ckpt else None,
        "config": {
            "lr": LEARNING_RATE, "warmup": WARMUP_STEPS,
            "batch_size": TRAIN_BATCH_SIZE, "grad_accum": GRAD_ACCUMULATION,
            "freeze_layers": FREEZE_ENCODER_LAYERS, "dropout": DROPOUT,
            "weight_decay": WEIGHT_DECAY,
        },
    }
    Path("/tmp/autoresearch_result.json").write_text(json.dumps(result, indent=2))
    print(f"\n{'='*60}")
    print(f"RESULT: val_loss={val_loss:.6f}  steps={trainer.global_step}  time={elapsed:.1f}s")
    print(f"{'='*60}")
    return val_loss


if __name__ == "__main__":
    train()
