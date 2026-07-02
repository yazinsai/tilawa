"""
Train whisper-small LoRA on Modal's cloud GPUs.

Datasets:
    - tarteel-ai/everyayah: 127K+ professional recitations (streaming)
    - RetaSy/quranic_audio_dataset: 6.8K phone-quality crowd-sourced recordings

Usage:
    modal run scripts/train_modal.py
"""
import modal

app = modal.App("whisper-quran-lora")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install(
        "torch",
        "transformers",
        "peft",
        "datasets>=3.0,<4.0",
        "accelerate",
        "evaluate",
        "jiwer",
        "soundfile",
        "librosa",
    )
)

vol = modal.Volume.from_name("whisper-quran-training", create_if_missing=True)


@app.function(
    image=image,
    gpu="A10G",
    timeout=7200,
    volumes={"/training": vol},
)
def train():
    import torch
    from dataclasses import dataclass
    from typing import Any, Dict, List, Union
    from pathlib import Path
    from datasets import load_dataset, Audio, interleave_datasets
    from transformers import (
        WhisperForConditionalGeneration,
        WhisperProcessor,
        Seq2SeqTrainingArguments,
        Seq2SeqTrainer,
    )
    from peft import LoraConfig, get_peft_model
    import evaluate

    OUTPUT_DIR = Path("/training/lora-adapter-small")
    CHECKPOINT_DIR = Path("/training/checkpoints")

    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Load processor + model ──
    print("Loading processor and model...")
    processor = WhisperProcessor.from_pretrained(
        "openai/whisper-small", language="arabic", task="transcribe"
    )
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
    model.config.forced_decoder_ids = None
    model.generation_config.forced_decoder_ids = None

    # ── Apply LoRA ──
    print("Applying LoRA...")
    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Load EveryAyah dataset (streaming) ──
    print("Loading EveryAyah dataset (streaming)...")
    everyayah = load_dataset("tarteel-ai/everyayah", split="train", streaming=True)
    everyayah = everyayah.cast_column("audio", Audio(sampling_rate=16000))
    everyayah = everyayah.filter(lambda x: x["duration"] <= 30.0)
    # Keep only audio + text to match RetaSy schema
    everyayah = everyayah.remove_columns(["duration", "reciter"])

    # ── Load RetaSy crowd-sourced dataset ──
    print("Loading RetaSy crowd-sourced dataset...")
    BAD_LABELS = {"in_correct", "not_related_quran", "not_match_aya"}
    retasy = load_dataset("RetaSy/quranic_audio_dataset", split="train")

    # Filter using metadata columns only (avoids triggering audio decode)
    labels = retasy["final_label"]
    durations = retasy["duration_ms"]
    good_indices = [
        i for i, (label, dur) in enumerate(zip(labels, durations))
        if (label not in BAD_LABELS if label else True) and dur <= 30000
    ]
    retasy = retasy.select(good_indices)
    print(f"  RetaSy samples after filtering: {len(retasy)}")

    # Rename Aya -> text to match everyayah schema
    retasy = retasy.rename_column("Aya", "text")
    # Drop extra columns to match everyayah
    retasy = retasy.remove_columns([
        c for c in retasy.column_names
        if c not in ("audio", "text")
    ])
    retasy = retasy.cast_column("audio", Audio(sampling_rate=16000))

    # Convert to streaming for interleaving
    retasy_stream = retasy.to_iterable_dataset()

    # Interleave: 85% everyayah, 15% RetaSy (phone-quality)
    dataset = interleave_datasets(
        [everyayah, retasy_stream],
        probabilities=[0.85, 0.15],
        seed=42,
    )
    print("Datasets interleaved (85% EveryAyah + 15% RetaSy phone-quality)")

    def prepare_dataset(batch):
        audio = batch["audio"]
        batch["input_features"] = processor(
            audio["array"],
            sampling_rate=audio["sampling_rate"],
            return_tensors="np",
        ).input_features[0]
        batch["labels"] = processor.tokenizer(batch["text"]).input_ids
        return batch

    dataset = dataset.map(
        prepare_dataset,
        remove_columns=["audio", "text"],
    )

    # ── Data collator ──
    @dataclass
    class DataCollatorSpeechSeq2SeqWithPadding:
        processor: Any
        decoder_start_token_id: int = None

        def __call__(self, features):
            input_features = [
                {"input_features": f["input_features"]} for f in features
            ]
            batch = self.processor.feature_extractor.pad(
                input_features, return_tensors="pt"
            )
            label_features = [{"input_ids": f["labels"]} for f in features]
            labels_batch = self.processor.tokenizer.pad(
                label_features, return_tensors="pt"
            )
            labels = labels_batch["input_ids"].masked_fill(
                labels_batch.attention_mask.ne(1), -100
            )
            if (
                labels[:, 0] == self.processor.tokenizer.bos_token_id
            ).all().cpu().item():
                labels = labels[:, 1:]
            batch["labels"] = labels
            return batch

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    # ── Training args ──
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(CHECKPOINT_DIR),
        per_device_train_batch_size=16,
        gradient_accumulation_steps=1,
        learning_rate=1e-5,
        warmup_steps=100,
        max_steps=3000,
        fp16=True,
        logging_steps=50,
        save_steps=500,
        remove_unused_columns=False,
        dataloader_pin_memory=True,
        report_to="none",
    )

    # ── Train ──
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        processing_class=processor.feature_extractor,
    )

    print("\n" + "=" * 60)
    print("  Starting training (3000 steps on A10G)")
    print("  Datasets: EveryAyah (pro) + RetaSy (phone-quality)")
    print("=" * 60 + "\n")

    trainer.train()

    # ── Save ──
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving LoRA adapter to {OUTPUT_DIR}...")
    model.save_pretrained(str(OUTPUT_DIR))
    processor.save_pretrained(str(OUTPUT_DIR))
    vol.commit()

    print("Done! Adapter saved to Modal volume.")


@app.function(
    image=image,
    volumes={"/training": vol},
)
def download_adapter():
    """Download the trained adapter files."""
    from pathlib import Path
    import json

    adapter_dir = Path("/training/lora-adapter-small")
    if not adapter_dir.exists():
        print("No adapter found! Run training first.")
        return {}

    files = {}
    for f in adapter_dir.iterdir():
        if f.is_file():
            files[f.name] = f.read_bytes()
            print(f"  {f.name}: {len(files[f.name]):,} bytes")

    return files


@app.local_entrypoint()
def main():
    from pathlib import Path

    print("Starting training on Modal GPU...")
    train.remote()

    print("\nDownloading adapter...")
    out_dir = Path("data/lora-adapter-small")
    out_dir.mkdir(parents=True, exist_ok=True)

    files = download_adapter.remote()
    for name, data in files.items():
        (out_dir / name).write_bytes(data)
        print(f"  Saved {name} ({len(data):,} bytes)")

    print(f"\nAdapter saved to {out_dir}")
    print("Done!")
