"""Fine-tune NVIDIA Arabic FastConformer on Quran data using Modal GPUs.

Training target:
  nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0

Primary data:
  rabah2026/Quran-Ayah-Corpus (train/validation splits)

Optional domain-noise blend:
  RetaSy/quranic_audio_dataset (phone-quality crowd recordings)

Artifacts are written to Modal volume "nvidia-fastconformer-quran-training":
  /training/<output_name>/
    manifests/train_manifest.jsonl
    manifests/val_manifest.jsonl
    checkpoints/...
    model/model.nemo
    model/training_metadata.json

Usage:
  modal run --detach scripts/train_nvidia_fastconformer_modal.py \
    --output-name nvidia-fastconformer-quran-ft-v1

  modal run scripts/train_nvidia_fastconformer_modal.py \
    --download-only \
    --output-name nvidia-fastconformer-quran-ft-v1
"""

from __future__ import annotations

import json
import types
from dataclasses import dataclass
from pathlib import Path

import modal

app = modal.App("nvidia-fastconformer-quran-ft")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install(
        "torch>=2.2",
        "nemo_toolkit[asr]>=2.7.0",
        "datasets>=3.0,<4.0",
        "lightning>=2.4,<3.0",
        "omegaconf>=2.3,<3.0",
        "soundfile",
        "librosa",
    )
)

vol = modal.Volume.from_name("nvidia-fastconformer-quran-training", create_if_missing=True)

BASE_MODEL_ID = "nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0"
PRIMARY_DATASET_ID = "rabah2026/Quran-Ayah-Corpus"
RETASY_DATASET_ID = "RetaSy/quranic_audio_dataset"
BAD_RETASY_LABELS = {"in_correct", "not_related_quran", "not_match_aya"}


def _install_kaldialign_fallback() -> None:
    """Install a tiny kaldialign-compatible fallback if package is absent.

    NeMo imports `kaldialign` in context-biasing paths even when that code path
    is not used for our training/inference setup.
    """
    import sys
    import types

    try:
        import kaldialign  # noqa: F401
        return
    except Exception:
        pass

    def align(ref, hyp, eps="<eps>"):
        ref = list(ref)
        hyp = list(hyp)
        n, m = len(ref), len(hyp)

        dp = [[0] * (m + 1) for _ in range(n + 1)]
        bt = [[None] * (m + 1) for _ in range(n + 1)]

        for i in range(1, n + 1):
            dp[i][0] = i
            bt[i][0] = "D"
        for j in range(1, m + 1):
            dp[0][j] = j
            bt[0][j] = "I"

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = 0 if ref[i - 1] == hyp[j - 1] else 1
                sub = dp[i - 1][j - 1] + cost
                ins = dp[i][j - 1] + 1
                delete = dp[i - 1][j] + 1
                best = min(sub, ins, delete)
                dp[i][j] = best
                if best == sub:
                    bt[i][j] = "S"
                elif best == ins:
                    bt[i][j] = "I"
                else:
                    bt[i][j] = "D"

        out = []
        i, j = n, m
        while i > 0 or j > 0:
            move = bt[i][j]
            if move == "S":
                out.append((ref[i - 1], hyp[j - 1]))
                i -= 1
                j -= 1
            elif move == "I":
                out.append((eps, hyp[j - 1]))
                j -= 1
            else:
                out.append((ref[i - 1], eps))
                i -= 1
        out.reverse()
        return out

    mod = types.ModuleType("kaldialign")
    mod.align = align
    sys.modules["kaldialign"] = mod


def _safe_text(text: str) -> str:
    text = (text or "").strip()
    return " ".join(text.split())


def _enable_ctc_only_training_step(model) -> None:
    """Patch NeMo hybrid model to optimize only CTC loss.

    NeMo's default hybrid training step always computes RNNT loss, which
    triggers numba CUDA kernels requiring libNVVM. For our pipeline we only
    consume CTC transcripts, so CTC-only adaptation is sufficient and avoids
    the RNNT/NVVM runtime path.
    """
    import torch

    def _ctc_training_step(self, batch, batch_nb):
        signal, signal_len, transcript, transcript_len = batch
        encoded, encoded_len = self.forward(input_signal=signal, input_signal_length=signal_len)
        del signal

        log_probs = self.ctc_decoder(encoder_output=encoded)
        ctc_loss = self.ctc_loss(
            log_probs=log_probs,
            targets=transcript,
            input_lengths=encoded_len,
            target_lengths=transcript_len,
        )
        loss_value = self.add_auxiliary_losses(ctc_loss)

        if hasattr(self, "_trainer") and self._trainer is not None:
            sample_id = self._trainer.global_step
            log_every_n_steps = max(int(self._trainer.log_every_n_steps), 1)
        else:
            sample_id = batch_nb
            log_every_n_steps = 1

        logs = {
            "learning_rate": self._optimizer.param_groups[0]["lr"],
            "global_step": torch.tensor(self.trainer.global_step, dtype=torch.float32),
            "train_ctc_loss": ctc_loss,
            "train_loss": loss_value,
        }

        if (sample_id + 1) % log_every_n_steps == 0:
            self.ctc_wer.update(
                predictions=log_probs,
                targets=transcript,
                targets_lengths=transcript_len,
                predictions_lengths=encoded_len,
            )
            ctc_wer, _, _ = self.ctc_wer.compute()
            self.ctc_wer.reset()
            logs["training_batch_wer_ctc"] = ctc_wer

        self.log_dict(logs)
        return {"loss": loss_value}

    model.training_step = types.MethodType(_ctc_training_step, model)
    model.ctc_loss_weight = 1.0
    try:
        from omegaconf import open_dict

        with open_dict(model.cfg):
            model.cfg.ctc_loss_weight = 1.0
    except Exception:
        pass


@dataclass
class BuildStats:
    written: int = 0
    skipped_empty: int = 0
    skipped_duration: int = 0
    bytes_written: int = 0


@app.function(
    image=image,
    cpu=8,
    memory=32768,
    timeout=60 * 60 * 10,
    volumes={"/training": vol},
)
def prepare_data(
    output_name: str = "nvidia-fastconformer-quran-ft-v1",
    max_train_samples: int = 120_000,
    max_val_samples: int = 8_000,
    include_retasy: bool = True,
    max_retasy_samples: int = 6_000,
    min_duration: float = 0.6,
    max_duration: float = 30.0,
    force_rebuild: bool = False,
):
    import io
    import soundfile as sf
    from datasets import Audio, load_dataset

    base = Path(f"/training/{output_name}")
    manifests_dir = base / "manifests"
    audio_root = base / "audio"
    train_manifest = manifests_dir / "train_manifest.jsonl"
    val_manifest = manifests_dir / "val_manifest.jsonl"
    metadata_path = manifests_dir / "data_metadata.json"

    if train_manifest.exists() and val_manifest.exists() and not force_rebuild:
        print("Manifests already exist, skipping rebuild.")
        return {
            "train_manifest": str(train_manifest),
            "val_manifest": str(val_manifest),
            "metadata": str(metadata_path),
        }

    manifests_dir.mkdir(parents=True, exist_ok=True)
    (audio_root / "train").mkdir(parents=True, exist_ok=True)
    (audio_root / "val").mkdir(parents=True, exist_ok=True)
    (audio_root / "retasy").mkdir(parents=True, exist_ok=True)

    def write_audio(audio_obj, out_path: Path) -> float:
        if isinstance(audio_obj, dict) and audio_obj.get("bytes") is not None:
            raw = audio_obj["bytes"]
            out_path.write_bytes(raw)
            # Use metadata duration when available from sample; fallback decode.
            try:
                with sf.SoundFile(io.BytesIO(raw)) as f:
                    return float(len(f) / float(f.samplerate))
            except Exception:
                return -1.0

        if isinstance(audio_obj, dict) and audio_obj.get("array") is not None:
            arr = audio_obj["array"]
            sr = int(audio_obj.get("sampling_rate", 16000))
            sf.write(out_path, arr, sr)
            return float(len(arr) / max(sr, 1))

        raise ValueError(f"Unsupported audio object format: {type(audio_obj)}")

    def build_rabah_split(split: str, limit: int, subdir: str):
        stats = BuildStats()
        ds = load_dataset(PRIMARY_DATASET_ID, split=split, streaming=True)
        ds = ds.cast_column("audio", Audio(sampling_rate=16000, decode=False))

        manifest_path = train_manifest if split == "train" else val_manifest
        with manifest_path.open("w", encoding="utf-8") as mf:
            for idx, sample in enumerate(ds):
                if stats.written >= limit:
                    break

                text = _safe_text(sample.get("text", ""))
                if not text:
                    stats.skipped_empty += 1
                    continue

                duration = float(sample.get("duration", -1.0))
                if duration > 0 and (duration < min_duration or duration > max_duration):
                    stats.skipped_duration += 1
                    continue

                out_file = audio_root / subdir / f"{split}_{idx:09d}.wav"
                measured = write_audio(sample["audio"], out_file)
                effective_duration = duration if duration > 0 else measured
                if effective_duration <= 0:
                    effective_duration = measured

                if effective_duration <= 0:
                    stats.skipped_duration += 1
                    out_file.unlink(missing_ok=True)
                    continue

                if effective_duration < min_duration or effective_duration > max_duration:
                    stats.skipped_duration += 1
                    out_file.unlink(missing_ok=True)
                    continue

                stats.written += 1
                stats.bytes_written += out_file.stat().st_size
                row = {
                    "audio_filepath": str(out_file),
                    "duration": round(float(effective_duration), 4),
                    "text": text,
                }
                mf.write(json.dumps(row, ensure_ascii=False) + "\n")

                if stats.written % 5000 == 0:
                    print(
                        f"[{split}] written={stats.written:,} "
                        f"skipped_dur={stats.skipped_duration:,} "
                        f"skipped_empty={stats.skipped_empty:,}"
                    )

        return stats

    print(f"Building manifests from {PRIMARY_DATASET_ID}...")
    train_stats = build_rabah_split("train", max_train_samples, "train")
    val_stats = build_rabah_split("validation", max_val_samples, "val")

    retasy_stats = BuildStats()
    if include_retasy and max_retasy_samples > 0:
        from datasets import Audio, load_dataset

        print(f"Adding RetaSy blend ({max_retasy_samples} max samples)...")
        retasy = load_dataset(RETASY_DATASET_ID, split="train")
        retasy = retasy.cast_column("audio", Audio(sampling_rate=16000, decode=False))

        with train_manifest.open("a", encoding="utf-8") as mf:
            for idx, sample in enumerate(retasy):
                if retasy_stats.written >= max_retasy_samples:
                    break

                label = sample.get("final_label")
                if label in BAD_RETASY_LABELS:
                    continue

                text = _safe_text(sample.get("Aya", ""))
                if not text:
                    retasy_stats.skipped_empty += 1
                    continue

                duration = float(sample.get("duration_ms", 0)) / 1000.0
                if duration <= 0:
                    duration = -1.0
                if duration > 0 and (duration < min_duration or duration > max_duration):
                    retasy_stats.skipped_duration += 1
                    continue

                out_file = audio_root / "retasy" / f"retasy_{idx:09d}.wav"
                measured = write_audio(sample["audio"], out_file)
                effective_duration = duration if duration > 0 else measured
                if effective_duration <= 0:
                    effective_duration = measured

                if effective_duration <= 0:
                    retasy_stats.skipped_duration += 1
                    out_file.unlink(missing_ok=True)
                    continue

                if effective_duration < min_duration or effective_duration > max_duration:
                    retasy_stats.skipped_duration += 1
                    out_file.unlink(missing_ok=True)
                    continue

                retasy_stats.written += 1
                retasy_stats.bytes_written += out_file.stat().st_size
                row = {
                    "audio_filepath": str(out_file),
                    "duration": round(float(effective_duration), 4),
                    "text": text,
                }
                mf.write(json.dumps(row, ensure_ascii=False) + "\n")

                if retasy_stats.written % 1000 == 0:
                    print(
                        f"[retasy] written={retasy_stats.written:,} "
                        f"skipped_dur={retasy_stats.skipped_duration:,} "
                        f"skipped_empty={retasy_stats.skipped_empty:,}"
                    )

    metadata = {
        "primary_dataset": PRIMARY_DATASET_ID,
        "retasy_dataset": RETASY_DATASET_ID if include_retasy else None,
        "max_train_samples": max_train_samples,
        "max_val_samples": max_val_samples,
        "include_retasy": include_retasy,
        "max_retasy_samples": max_retasy_samples,
        "min_duration": min_duration,
        "max_duration": max_duration,
        "train": train_stats.__dict__,
        "validation": val_stats.__dict__,
        "retasy_blend": retasy_stats.__dict__,
        "train_manifest": str(train_manifest),
        "val_manifest": str(val_manifest),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    vol.commit()
    print("Data preparation complete.")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return {
        "train_manifest": str(train_manifest),
        "val_manifest": str(val_manifest),
        "metadata": str(metadata_path),
    }


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=60 * 60 * 12,
    volumes={"/training": vol},
)
def train(
    output_name: str = "nvidia-fastconformer-quran-ft-v1",
    train_batch_size: int = 16,
    val_batch_size: int = 16,
    grad_accum: int = 2,
    max_steps: int = 10_000,
    learning_rate: float = 5e-5,
    warmup_steps: int = 500,
    freeze_encoder_layers: int = 8,
    freeze_preprocessor: bool = True,
    freeze_rnnt_decoder: bool = True,
    ctc_only_finetune: bool = True,
    val_check_interval: int = 500,
    num_workers: int = 8,
    wait_for_manifests_minutes: int = 0,
):
    import lightning.pytorch as pl
    import time
    import torch
    from omegaconf import open_dict

    _install_kaldialign_fallback()
    from nemo.collections.asr.models import EncDecHybridRNNTCTCBPEModel

    base = Path(f"/training/{output_name}")
    manifests_dir = base / "manifests"
    train_manifest = manifests_dir / "train_manifest.jsonl"
    val_manifest = manifests_dir / "val_manifest.jsonl"
    metadata_path = manifests_dir / "data_metadata.json"
    checkpoints_dir = base / "checkpoints"
    output_dir = base / "model"
    output_dir.mkdir(parents=True, exist_ok=True)

    if wait_for_manifests_minutes > 0:
        deadline = time.time() + (60 * int(wait_for_manifests_minutes))
        while time.time() < deadline:
            if train_manifest.exists() and val_manifest.exists():
                break
            print("Waiting for manifests to be prepared...")
            time.sleep(60)

    if not train_manifest.exists() or not val_manifest.exists():
        raise FileNotFoundError("Missing manifests. Run prepare_data first.")

    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Loading base model: {BASE_MODEL_ID}")
    model = EncDecHybridRNNTCTCBPEModel.from_pretrained(
        model_name=BASE_MODEL_ID,
        map_location="cuda" if torch.cuda.is_available() else "cpu",
    )

    with open_dict(model.cfg):
        # Required fields must be resolved (not "???") before setup_*_data.
        model.cfg.train_ds.manifest_filepath = str(train_manifest)
        model.cfg.train_ds.is_tarred = False
        model.cfg.train_ds.tarred_audio_filepaths = None
        model.cfg.train_ds.batch_size = int(train_batch_size)
        model.cfg.train_ds.shuffle = True
        model.cfg.train_ds.num_workers = int(num_workers)
        model.cfg.train_ds.pin_memory = True
        model.cfg.train_ds.max_duration = 30.0
        model.cfg.train_ds.min_duration = 0.6
        model.cfg.train_ds.shuffle_n = 2048

        model.cfg.validation_ds.manifest_filepath = str(val_manifest)
        model.cfg.validation_ds.batch_size = int(val_batch_size)
        model.cfg.validation_ds.shuffle = False
        model.cfg.validation_ds.num_workers = int(num_workers)
        model.cfg.validation_ds.pin_memory = True
        model.cfg.validation_ds.use_start_end_token = False

        model.cfg.test_ds.manifest_filepath = str(val_manifest)
        model.cfg.test_ds.batch_size = int(val_batch_size)
        model.cfg.test_ds.shuffle = False
        model.cfg.test_ds.num_workers = int(num_workers)
        model.cfg.test_ds.pin_memory = True
        model.cfg.test_ds.use_start_end_token = False

        model.cfg.tokenizer.dir = "/tmp"
        model.cfg.optim.lr = float(learning_rate)
        if "sched" in model.cfg.optim and "warmup_steps" in model.cfg.optim.sched:
            model.cfg.optim.sched.warmup_steps = int(warmup_steps)

    if freeze_preprocessor and hasattr(model, "preprocessor"):
        for p in model.preprocessor.parameters():
            p.requires_grad = False

    if hasattr(model, "encoder") and hasattr(model.encoder, "layers"):
        freeze_n = max(0, min(int(freeze_encoder_layers), len(model.encoder.layers)))
        for idx, layer in enumerate(model.encoder.layers):
            req_grad = idx >= freeze_n
            for p in layer.parameters():
                p.requires_grad = req_grad
        print(f"Encoder freeze: first {freeze_n}/{len(model.encoder.layers)} layers")
    else:
        print("Warning: encoder layers not found; skipping partial encoder freeze.")

    if freeze_rnnt_decoder:
        if hasattr(model, "decoder"):
            for p in model.decoder.parameters():
                p.requires_grad = False
        if hasattr(model, "joint"):
            for p in model.joint.parameters():
                p.requires_grad = False
        print("RNNT decoder/joint frozen (CTC-heavy adaptation).")

    if ctc_only_finetune:
        _enable_ctc_only_training_step(model)
        print("CTC-only finetune mode enabled (RNNT loss path disabled).")

    # Keep CTC decoding active for transcript-side behavior checks.
    try:
        model.change_decoding_strategy(decoder_type="ctc", verbose=False)
    except Exception:
        pass

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"Params total={total_params:,} "
        f"trainable={trainable_params:,} "
        f"({(trainable_params / max(total_params, 1)) * 100:.2f}%)"
    )

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        max_steps=int(max_steps),
        precision="bf16-mixed" if torch.cuda.is_available() else "32-true",
        default_root_dir=str(checkpoints_dir),
        gradient_clip_val=1.0,
        accumulate_grad_batches=int(grad_accum),
        log_every_n_steps=20,
        num_sanity_val_steps=0,
        val_check_interval=int(val_check_interval),
        limit_val_batches=0 if ctc_only_finetune else 1.0,
        enable_checkpointing=True,
        enable_progress_bar=True,
    )

    model.set_trainer(trainer)
    model.setup_training_data(model.cfg.train_ds)
    if not ctc_only_finetune:
        model.setup_validation_data(model.cfg.validation_ds)

    print("\n" + "=" * 72)
    print("NVIDIA FastConformer Quran fine-tune")
    print(f"output_name:        {output_name}")
    print(f"train_manifest:     {train_manifest}")
    print(f"val_manifest:       {val_manifest}")
    print(f"batch/train:        {train_batch_size}")
    print(f"batch/val:          {val_batch_size}")
    print(f"grad_accum:         {grad_accum}")
    print(f"max_steps:          {max_steps}")
    print(f"learning_rate:      {learning_rate}")
    print(f"warmup_steps:       {warmup_steps}")
    print(f"freeze_layers:      {freeze_encoder_layers}")
    print(f"freeze_preprocessor:{freeze_preprocessor}")
    print(f"freeze_rnnt_decoder:{freeze_rnnt_decoder}")
    print(f"ctc_only_finetune:  {ctc_only_finetune}")
    print("=" * 72 + "\n")

    trainer.fit(model)

    nemo_path = output_dir / "model.nemo"
    model.save_to(str(nemo_path))

    train_meta = {
        "base_model": BASE_MODEL_ID,
        "train_manifest": str(train_manifest),
        "val_manifest": str(val_manifest),
        "train_batch_size": train_batch_size,
        "val_batch_size": val_batch_size,
        "grad_accum": grad_accum,
        "max_steps": max_steps,
        "learning_rate": learning_rate,
        "warmup_steps": warmup_steps,
        "freeze_encoder_layers": freeze_encoder_layers,
        "freeze_preprocessor": freeze_preprocessor,
        "freeze_rnnt_decoder": freeze_rnnt_decoder,
        "ctc_only_finetune": ctc_only_finetune,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "data_metadata_path": str(metadata_path),
    }
    (output_dir / "training_metadata.json").write_text(
        json.dumps(train_meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    vol.commit()
    print(f"Saved fine-tuned model to {nemo_path}")
    return {"model_path": str(nemo_path), "metadata_path": str(output_dir / "training_metadata.json")}


@app.function(
    image=image,
    volumes={"/training": vol},
)
def download_model(output_name: str = "nvidia-fastconformer-quran-ft-v1"):
    """Download model artifacts from Modal volume."""
    model_dir = Path(f"/training/{output_name}/model")
    if not model_dir.exists():
        print(f"No model directory found at: {model_dir}")
        return {}

    files: dict[str, bytes] = {}
    for f in model_dir.rglob("*"):
        if f.is_file():
            rel = str(f.relative_to(model_dir))
            data = f.read_bytes()
            files[rel] = data
            print(f"  {rel}: {len(data):,} bytes")
    return files


@app.local_entrypoint()
def main(
    output_name: str = "nvidia-fastconformer-quran-ft-v1",
    max_train_samples: int = 120_000,
    max_val_samples: int = 8_000,
    include_retasy: bool = True,
    max_retasy_samples: int = 6_000,
    min_duration: float = 0.6,
    max_duration: float = 30.0,
    force_rebuild_data: bool = False,
    train_batch_size: int = 16,
    val_batch_size: int = 16,
    grad_accum: int = 2,
    max_steps: int = 10_000,
    learning_rate: float = 5e-5,
    warmup_steps: int = 500,
    freeze_encoder_layers: int = 8,
    freeze_preprocessor: bool = True,
    freeze_rnnt_decoder: bool = True,
    ctc_only_finetune: bool = True,
    val_check_interval: int = 500,
    num_workers: int = 8,
    download_only: bool = False,
    download_after_train: bool = False,
):
    local_out_dir = Path("data") / output_name

    if not download_only:
        print("Preparing manifests/audio on Modal volume...")
        prepare_data.remote(
            output_name=output_name,
            max_train_samples=max_train_samples,
            max_val_samples=max_val_samples,
            include_retasy=include_retasy,
            max_retasy_samples=max_retasy_samples,
            min_duration=min_duration,
            max_duration=max_duration,
            force_rebuild=force_rebuild_data,
        )

        print("\nStarting GPU fine-tuning job...")
        train.remote(
            output_name=output_name,
            train_batch_size=train_batch_size,
            val_batch_size=val_batch_size,
            grad_accum=grad_accum,
            max_steps=max_steps,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            freeze_encoder_layers=freeze_encoder_layers,
            freeze_preprocessor=freeze_preprocessor,
            freeze_rnnt_decoder=freeze_rnnt_decoder,
            ctc_only_finetune=ctc_only_finetune,
            val_check_interval=val_check_interval,
            num_workers=num_workers,
        )

    if not download_after_train and not download_only:
        print(
            "\nTraining job submitted. Skipping local download "
            "(set --download-after-train to fetch artifacts automatically)."
        )
        return

    print(f"\nDownloading model artifacts for {output_name}...")
    local_out_dir.mkdir(parents=True, exist_ok=True)
    files = download_model.remote(output_name=output_name)
    if not files:
        print("No files to download.")
        return

    total = 0
    for rel, data in files.items():
        out_path = local_out_dir / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(data)
        total += len(data)
        print(f"  saved {rel} ({len(data):,} bytes)")

    print(f"\nDownloaded to {local_out_dir} ({total / 1e6:.1f} MB)")
