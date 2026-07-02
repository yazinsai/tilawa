"""
Fine-tune wav2vec2-base with Arabic CTC head on Quran audio (Modal A10G).

Creates a small CTC model (~380MB) that can score verses via CTC alignment,
matching the vocabulary of the large wav2vec2-xlsr-53-arabic model.

Output: Modal volume "ctc-quran-training" → data/ctc-base-finetuned/

Usage:
    modal run scripts/train_ctc_base_modal.py
"""
import modal

app = modal.App("ctc-base-quran")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install(
        "torch",
        "transformers>=4.40,<5",
        "datasets>=3.0,<4.0",
        "accelerate",
        "evaluate",
        "jiwer",
        "soundfile",
        "librosa",
    )
)

vol = modal.Volume.from_name("ctc-quran-training", create_if_missing=True)


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=10800,  # 3 hours
    volumes={"/training": vol},
)
def train():
    import torch
    import json
    from pathlib import Path
    from dataclasses import dataclass
    from typing import Any, Dict, List, Union
    from datasets import load_dataset, Audio, interleave_datasets
    from transformers import (
        Wav2Vec2ForCTC,
        Wav2Vec2Processor,
        Wav2Vec2Config,
        Wav2Vec2Model,
        Trainer,
        TrainingArguments,
    )
    import evaluate
    import numpy as np

    OUTPUT_DIR = Path("/training/ctc-base-finetuned")
    CHECKPOINT_DIR = Path("/training/ctc-checkpoints")

    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Get Arabic vocabulary from the large model ──
    print("Loading Arabic CTC processor from large model...")
    large_processor = Wav2Vec2Processor.from_pretrained(
        "jonatasgrosman/wav2vec2-large-xlsr-53-arabic"
    )
    vocab_size = large_processor.tokenizer.vocab_size
    pad_token_id = large_processor.tokenizer.pad_token_id
    print(f"Arabic vocabulary size: {vocab_size}, pad_id: {pad_token_id}")

    # ── Create small model with Arabic CTC head ──
    print("Creating wav2vec2-base with Arabic CTC head...")
    base_config = Wav2Vec2Config.from_pretrained("facebook/wav2vec2-base")
    base_config.vocab_size = vocab_size
    base_config.ctc_loss_reduction = "mean"
    base_config.pad_token_id = pad_token_id
    base_config.ctc_zero_infinity = True

    # Initialize full model with CTC head
    model = Wav2Vec2ForCTC(base_config)

    # Load pretrained SSL weights (everything except the CTC head)
    print("Loading pretrained wav2vec2-base weights...")
    pretrained = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
    model.wav2vec2.load_state_dict(pretrained.state_dict())
    del pretrained

    # Freeze feature extractor (CNN layers) — only fine-tune transformer + CTC head
    model.freeze_feature_encoder()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,} ({total_params * 4 / 1e6:.0f} MB)")
    print(f"Trainable params: {trainable_params:,}")

    # ── Use large model's processor for consistent tokenization ──
    processor = large_processor

    # ── Load datasets ──
    print("Loading EveryAyah dataset (streaming)...")
    everyayah = load_dataset("tarteel-ai/everyayah", split="train", streaming=True)
    everyayah = everyayah.cast_column("audio", Audio(sampling_rate=16000))
    everyayah = everyayah.filter(lambda x: x["duration"] <= 20.0)
    everyayah = everyayah.remove_columns(["duration", "reciter"])

    print("Loading RetaSy crowd-sourced dataset...")
    BAD_LABELS = {"in_correct", "not_related_quran", "not_match_aya"}
    retasy = load_dataset("RetaSy/quranic_audio_dataset", split="train")
    labels = retasy["final_label"]
    durations = retasy["duration_ms"]
    good_indices = [
        i for i, (label, dur) in enumerate(zip(labels, durations))
        if (label not in BAD_LABELS if label else True) and dur <= 30000
    ]
    retasy = retasy.select(good_indices)
    print(f"  RetaSy samples after filtering: {len(retasy)}")
    retasy = retasy.rename_column("Aya", "text")
    retasy = retasy.remove_columns([
        c for c in retasy.column_names if c not in ("audio", "text")
    ])
    retasy = retasy.cast_column("audio", Audio(sampling_rate=16000))
    retasy_stream = retasy.to_iterable_dataset()

    # Interleave: 85% EveryAyah, 15% RetaSy
    dataset = interleave_datasets(
        [everyayah, retasy_stream],
        probabilities=[0.85, 0.15],
        seed=42,
    )
    print("Datasets interleaved (85% EveryAyah + 15% RetaSy)")

    # ── Prepare dataset ──
    import io
    import soundfile as sf

    def prepare_dataset(batch):
        audio_data = batch["audio"]
        # Handle both streaming formats: {"array": [...]} or {"bytes": b"..."}
        if isinstance(audio_data, dict) and "array" in audio_data:
            audio_array = audio_data["array"]
        elif isinstance(audio_data, dict) and "bytes" in audio_data:
            audio_array, sr = sf.read(io.BytesIO(audio_data["bytes"]), dtype="float32")
            if sr != 16000:
                import librosa
                audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)
        else:
            raise ValueError(f"Unexpected audio format: {type(audio_data)}")

        batch["input_values"] = processor.feature_extractor(
            audio_array,
            sampling_rate=16000,
            return_tensors="np",
        ).input_values[0]

        # Tokenize text to character IDs for CTC
        with processor.as_target_processor():
            batch["labels"] = processor(batch["text"]).input_ids

        return batch

    dataset = dataset.map(prepare_dataset, remove_columns=["audio", "text"])

    # ── Data collator for CTC ──
    @dataclass
    class DataCollatorCTCWithPadding:
        processor: Any

        def __call__(self, features):
            input_values = [{"input_values": f["input_values"]} for f in features]
            batch = self.processor.feature_extractor.pad(
                input_values, return_tensors="pt", padding=True,
            )

            label_features = [{"input_ids": f["labels"]} for f in features]
            labels_batch = self.processor.tokenizer.pad(
                label_features, return_tensors="pt", padding=True,
            )

            # Replace padding with -100 so CTC loss ignores it
            labels = labels_batch["input_ids"].masked_fill(
                labels_batch.attention_mask.ne(1), -100
            )
            batch["labels"] = labels
            return batch

    data_collator = DataCollatorCTCWithPadding(processor=processor)

    # ── Metrics ──
    wer_metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)
        # Replace -100 in labels
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.batch_decode(pred_ids)
        label_str = processor.batch_decode(label_ids, group_tokens=False)
        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    # ── Training ──
    training_args = TrainingArguments(
        output_dir=str(CHECKPOINT_DIR),
        per_device_train_batch_size=32,
        gradient_accumulation_steps=2,
        learning_rate=1e-4,
        warmup_steps=500,
        max_steps=5000,
        bf16=True,
        logging_steps=100,
        save_steps=1000,
        remove_unused_columns=False,
        dataloader_pin_memory=True,
        dataloader_num_workers=4,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        processing_class=processor.feature_extractor,
    )

    print("\n" + "=" * 60)
    print("  Training wav2vec2-base CTC on Quran audio (5000 steps)")
    print("  Model: 95M params, Arabic CTC vocabulary")
    print("  Data: EveryAyah + RetaSy")
    print("=" * 60 + "\n")

    trainer.train()

    # ── Save ──
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving model to {OUTPUT_DIR}...")
    model.save_pretrained(str(OUTPUT_DIR))
    processor.save_pretrained(str(OUTPUT_DIR))
    vol.commit()

    print("Done! Model saved to Modal volume.")


@app.function(
    image=image,
    volumes={"/training": vol},
)
def download_model():
    """Download the trained model files."""
    from pathlib import Path

    model_dir = Path("/training/ctc-base-finetuned")
    if not model_dir.exists():
        print("No model found! Run training first.")
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

    print("Starting CTC-base training on Modal GPU...")
    train.remote()

    print("\nDownloading trained model...")
    out_dir = Path("data/ctc-base-finetuned")
    out_dir.mkdir(parents=True, exist_ok=True)

    files = download_model.remote()
    for name, data in files.items():
        path = out_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        print(f"  Saved {name} ({len(data):,} bytes)")

    total = sum(len(d) for d in files.values())
    print(f"\nModel saved to {out_dir} ({total / 1e6:.0f} MB total)")
