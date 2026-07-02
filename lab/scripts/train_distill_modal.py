"""
Knowledge distillation: large wav2vec2-xlsr-53-arabic → small wav2vec2-base CTC.

Teacher: jonatasgrosman/wav2vec2-large-xlsr-53-arabic (315M, 1.2GB)
Student: wav2vec2-base + Arabic CTC head (95M, ~380MB)

Loss = α * CTC_loss(student, labels) + (1-α) * KL(student || teacher)

Output: Modal volume → data/ctc-base-distilled/

Usage:
    modal run scripts/train_distill_modal.py
"""
import modal

app = modal.App("ctc-distill-quran")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install(
        "torch",
        "transformers>=4.40,<5",
        "datasets>=3.0,<4.0",
        "accelerate",
        "soundfile",
        "librosa",
    )
)

vol = modal.Volume.from_name("ctc-quran-training", create_if_missing=True)


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=5400,  # 1.5 hours
    volumes={"/training": vol},
    memory=32768,
)
def train(alpha: float = 0.5, temperature: float = 2.0, max_steps: int = 3000):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import math
    from pathlib import Path
    from dataclasses import dataclass
    from typing import Any
    from torch.utils.data import IterableDataset, DataLoader
    from datasets import load_dataset, Audio
    from transformers import (
        Wav2Vec2ForCTC,
        Wav2Vec2Processor,
        Wav2Vec2Config,
        Wav2Vec2Model,
    )
    import numpy as np

    BATCH_SIZE = 16
    GRAD_ACCUM = 2  # effective batch = 32
    OUTPUT_DIR = Path("/training/ctc-base-distilled")
    CHECKPOINT_DIR = Path("/training/distill-checkpoints")
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load teacher model ──
    print("Loading teacher model (wav2vec2-large-xlsr-53-arabic)...")
    teacher_processor = Wav2Vec2Processor.from_pretrained(
        "jonatasgrosman/wav2vec2-large-xlsr-53-arabic"
    )
    teacher = Wav2Vec2ForCTC.from_pretrained(
        "jonatasgrosman/wav2vec2-large-xlsr-53-arabic"
    )
    teacher.eval()
    teacher.to(device)
    for p in teacher.parameters():
        p.requires_grad = False

    vocab_size = teacher_processor.tokenizer.vocab_size
    pad_token_id = teacher_processor.tokenizer.pad_token_id

    # ── Create student model ──
    student_pretrained = Path("/training/ctc-base-finetuned")
    if student_pretrained.exists():
        print(f"Loading pre-fine-tuned student from {student_pretrained}...")
        student = Wav2Vec2ForCTC.from_pretrained(str(student_pretrained))
    else:
        print("Creating fresh wav2vec2-base student with Arabic CTC head...")
        base_config = Wav2Vec2Config.from_pretrained("facebook/wav2vec2-base")
        base_config.vocab_size = vocab_size
        base_config.ctc_loss_reduction = "mean"
        base_config.pad_token_id = pad_token_id
        base_config.ctc_zero_infinity = True
        student = Wav2Vec2ForCTC(base_config)
        pretrained = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        student.wav2vec2.load_state_dict(pretrained.state_dict())
        del pretrained

    student.freeze_feature_encoder()
    student.to(device)
    student = torch.compile(student)
    student.train()

    processor = teacher_processor

    print(f"Teacher: {sum(p.numel() for p in teacher.parameters()):,} params")
    print(f"Student: {sum(p.numel() for p in student.parameters()):,} params")
    print(f"  Trainable: {sum(p.numel() for p in student.parameters() if p.requires_grad):,}")
    print(f"Alpha: {alpha}, Temperature: {temperature}, Batch: {BATCH_SIZE}x{GRAD_ACCUM}")

    # ── Streaming dataset with batched DataLoader ──
    class StreamingCTCDataset(IterableDataset):
        def __init__(self, stream, processor, max_audio_len):
            self.stream = stream
            self.processor = processor
            self.max_audio_len = max_audio_len

        def __iter__(self):
            import io
            import soundfile as sf
            for sample in self.stream:
                try:
                    audio_data = sample["audio"]
                    if isinstance(audio_data, dict) and "array" in audio_data:
                        audio = audio_data["array"]
                    elif isinstance(audio_data, dict) and "bytes" in audio_data:
                        audio, sr = sf.read(io.BytesIO(audio_data["bytes"]), dtype="float32")
                        if sr != 16000:
                            import librosa
                            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                    else:
                        continue
                    if len(audio) > self.max_audio_len:
                        audio = audio[:self.max_audio_len]
                    input_values = self.processor.feature_extractor(
                        audio, sampling_rate=16000, return_tensors="np",
                    ).input_values[0]
                    with self.processor.as_target_processor():
                        label_ids = self.processor(sample["text"]).input_ids
                    yield {
                        "input_values": input_values,
                        "labels": label_ids,
                    }
                except Exception:
                    continue

    @dataclass
    class DistillCollator:
        pad_token_id: int

        def __call__(self, features):
            # Pad input_values
            max_iv = max(len(f["input_values"]) for f in features)
            iv_padded = []
            iv_lengths = []
            for f in features:
                iv = f["input_values"]
                iv_lengths.append(len(iv))
                pad_len = max_iv - len(iv)
                iv_padded.append(np.pad(iv, (0, pad_len)))
            # Pad labels
            max_lbl = max(len(f["labels"]) for f in features)
            labels_padded = []
            label_lengths = []
            for f in features:
                lbl = f["labels"]
                label_lengths.append(len(lbl))
                pad_len = max_lbl - len(lbl)
                labels_padded.append(lbl + [self.pad_token_id] * pad_len)
            return {
                "input_values": torch.tensor(np.stack(iv_padded), dtype=torch.float32),
                "labels": torch.tensor(labels_padded, dtype=torch.long),
                "input_lengths": torch.tensor(iv_lengths, dtype=torch.long),
                "label_lengths": torch.tensor(label_lengths, dtype=torch.long),
            }

    print("Loading EveryAyah dataset (streaming)...")
    everyayah = load_dataset("tarteel-ai/everyayah", split="train", streaming=True)
    everyayah = everyayah.cast_column("audio", Audio(sampling_rate=16000))
    everyayah = everyayah.filter(lambda x: x["duration"] <= 20.0)

    max_audio_len = int(20.0 * 16000)
    dataset = StreamingCTCDataset(everyayah, processor, max_audio_len)
    collator = DistillCollator(pad_token_id=pad_token_id)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collator, num_workers=4)

    # ── Optimizer + scheduler ──
    optimizer = torch.optim.AdamW(
        [p for p in student.parameters() if p.requires_grad],
        lr=3e-4,
        weight_decay=0.01,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_steps, eta_min=1e-5)

    ctc_loss_fn = nn.CTCLoss(blank=pad_token_id, reduction="mean", zero_infinity=True)

    print(f"\n{'='*60}")
    print(f"  Knowledge distillation ({max_steps} steps, effective batch={BATCH_SIZE*GRAD_ACCUM})")
    print(f"{'='*60}\n")

    step = 0
    micro_step = 0
    running_loss = 0.0
    running_ctc = 0.0
    running_kl = 0.0

    optimizer.zero_grad()

    for batch in loader:
        if step >= max_steps:
            break

        try:
            input_values = batch["input_values"].to(device)
            labels = batch["labels"].to(device)
            label_lengths = batch["label_lengths"].to(device)
            B = input_values.size(0)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                # Teacher forward (no grad)
                with torch.no_grad():
                    teacher_logits = teacher(input_values).logits

                # Student forward
                student_logits = student(input_values).logits

                T_student = student_logits.size(1)
                T_teacher = teacher_logits.size(1)

                # CTC loss — compute per-sample input lengths from model downsampling
                # wav2vec2 downsamples by ~320x
                input_sample_lengths = (batch["input_lengths"].float() / 320).long().clamp(min=1, max=T_student)
                input_lengths = input_sample_lengths.to(device)

                log_probs = F.log_softmax(student_logits, dim=-1).permute(1, 0, 2)
                ctc = ctc_loss_fn(log_probs, labels, input_lengths, label_lengths)

                # KL divergence — align temporal dims
                if T_teacher != T_student:
                    teacher_aligned = F.interpolate(
                        teacher_logits.permute(0, 2, 1),
                        size=T_student,
                        mode="linear",
                        align_corners=False,
                    ).permute(0, 2, 1)
                else:
                    teacher_aligned = teacher_logits

                student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
                teacher_probs = F.softmax(teacher_aligned / temperature, dim=-1)
                kl = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temperature ** 2)

                loss = (alpha * ctc + (1 - alpha) * kl) / GRAD_ACCUM

            loss.backward()
            micro_step += 1

            running_loss += loss.item() * GRAD_ACCUM
            running_ctc += ctc.item()
            running_kl += kl.item()

            if micro_step % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step += 1

                if step % 100 == 0:
                    n = min(100, step) * GRAD_ACCUM
                    avg_loss = running_loss / n * GRAD_ACCUM
                    avg_ctc = running_ctc / n * GRAD_ACCUM
                    avg_kl = running_kl / n * GRAD_ACCUM
                    lr = scheduler.get_last_lr()[0]
                    print(f"Step {step}/{max_steps}: loss={avg_loss:.4f} "
                          f"(ctc={avg_ctc:.4f}, kl={avg_kl:.4f}) lr={lr:.2e}")
                    running_loss = 0.0
                    running_ctc = 0.0
                    running_kl = 0.0

                if step % 1000 == 0:
                    ckpt_path = CHECKPOINT_DIR / f"step_{step}"
                    student.save_pretrained(str(ckpt_path))
                    vol.commit()
                    print(f"  Saved checkpoint at step {step}")

        except Exception as e:
            print(f"  Skipping batch (error: {e})")
            optimizer.zero_grad()
            micro_step = 0
            continue

    # ── Save final model ──
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving distilled model to {OUTPUT_DIR}...")
    student.save_pretrained(str(OUTPUT_DIR))
    processor.save_pretrained(str(OUTPUT_DIR))
    vol.commit()

    print("Done! Distilled model saved to Modal volume.")


@app.function(
    image=image,
    volumes={"/training": vol},
)
def download_model():
    from pathlib import Path

    model_dir = Path("/training/ctc-base-distilled")
    if not model_dir.exists():
        print("No distilled model found! Run training first.")
        return {}

    files = {}
    for f in model_dir.rglob("*"):
        if f.is_file():
            rel = str(f.relative_to(model_dir))
            files[rel] = f.read_bytes()
            print(f"  {rel}: {len(files[rel]):,} bytes")
    return files


@app.local_entrypoint()
def main():
    from pathlib import Path

    print("Starting knowledge distillation on Modal GPU...")
    train.remote(alpha=0.5, temperature=2.0, max_steps=3000)

    print("\nDownloading distilled model...")
    out_dir = Path("data/ctc-base-distilled")
    out_dir.mkdir(parents=True, exist_ok=True)

    files = download_model.remote()
    for name, data in files.items():
        path = out_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        print(f"  Saved {name} ({len(data):,} bytes)")

    total = sum(len(d) for d in files.values())
    print(f"\nDistilled model saved to {out_dir} ({total / 1e6:.0f} MB total)")
