"""
Fine-tune jonatasgrosman/wav2vec2-large-xlsr-53-arabic on Quran audio (Modal A10G).

This model already hits 81% on our benchmark. Fine-tuning on Quran-specific
recitations should improve accuracy on tajweed patterns and short verses.

Dataset: Buraaq/quran-md-ayahs (187K samples, 30 reciters)
Approach: Freeze CNN feature extractor, fine-tune transformer + existing CTC head
          with the model's native 51-token Arabic vocabulary.

Usage:
    modal run scripts/train_ctc_modal.py                  # train + download
    modal run scripts/train_ctc_modal.py --download-only  # just download
"""
import modal

app = modal.App("wav2vec2-quran-ctc")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libsndfile1-dev")
    .pip_install(
        "torch",
        "transformers>=4.40",
        "datasets>=3.0,<4.0",
        "accelerate",
        "evaluate",
        "jiwer",
        "soundfile",
        "librosa",
    )
)

vol = modal.Volume.from_name("wav2vec2-quran-ctc-v2", create_if_missing=True)

BASE_MODEL = "jonatasgrosman/wav2vec2-large-xlsr-53-arabic"


@app.function(
    image=image,
    gpu="A10G",
    timeout=21600,  # 6 hours
    volumes={"/training": vol},
)
def train():
    import re
    import torch
    from pathlib import Path
    from dataclasses import dataclass
    from typing import Any, Union
    from datasets import load_dataset, Audio
    from transformers import (
        Wav2Vec2ForCTC,
        Wav2Vec2Processor,
        TrainingArguments,
        Trainer,
    )

    OUTPUT_DIR = Path("/training/ctc-model")
    CHECKPOINT_DIR = Path("/training/ctc-checkpoints")

    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # -- Load processor + model (already has Arabic CTC head) --
    print(f"Loading {BASE_MODEL}...")
    processor = Wav2Vec2Processor.from_pretrained(BASE_MODEL)
    model = Wav2Vec2ForCTC.from_pretrained(
        BASE_MODEL,
        attention_dropout=0.05,
        hidden_dropout=0.05,
        feat_proj_dropout=0.0,
        mask_time_prob=0.05,
        layerdrop=0.05,
        ctc_loss_reduction="mean",
        ctc_zero_infinity=True,
    )

    # Freeze feature extractor (CNN layers) — only fine-tune transformer + CTC head
    model.freeze_feature_encoder()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total/1e6:.0f}M total, {trainable/1e6:.0f}M trainable")

    # -- Diacritic stripping (keep character forms as-is for model's native vocab) --
    _DIACRITICS = re.compile(
        '[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06DC'
        '\u06DF-\u06E4\u06E7\u06E8\u06EA-\u06ED\u0640]'
    )

    def strip_diacritics(text):
        """Strip diacritics only. Don't normalize alef/taa marbuta —
        the model's vocab has separate tokens for those."""
        text = _DIACRITICS.sub('', text)
        return ' '.join(text.split())

    # -- Load dataset --
    print("Loading Buraaq/quran-md-ayahs...")
    ds = load_dataset("Buraaq/quran-md-ayahs", split="train", streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))

    # Hold out 3 reciters for validation
    VAL_RECITERS = {
        "saood_ash_shuraym",
        "abdulbaset_abdulsamad_mojawwad",
        "hani_ar_rifai",
    }

    train_ds = ds.filter(lambda x: x["reciter_id"] not in VAL_RECITERS)

    _skip_count = [0]

    def prepare(batch):
        import io
        import librosa

        try:
            audio_data = batch["audio"]
            audio_bytes = audio_data["bytes"]
            audio_array, _ = librosa.load(
                io.BytesIO(audio_bytes), sr=16000, mono=True
            )

            # Skip very long audio (>15s to prevent OOM)
            if len(audio_array) > 16000 * 15:
                raise ValueError(f"Audio too long: {len(audio_array)/16000:.1f}s")

            input_values = processor(
                audio_array,
                sampling_rate=16000,
                return_tensors=None,
            ).input_values[0]

            # Strip diacritics but keep character forms for model's vocab
            text = strip_diacritics(batch["ayah_ar"])
            labels = processor.tokenizer(text).input_ids

            if len(labels) == 0:
                raise ValueError("Empty label sequence")

            return {"input_values": input_values, "labels": labels, "_valid": True}
        except Exception as e:
            _skip_count[0] += 1
            if _skip_count[0] <= 10:
                print(f"Skipping bad sample ({_skip_count[0]}): {e}")
            return {"input_values": [0.0], "labels": [0], "_valid": False}

    all_columns = [
        "surah_id", "ayah_id", "surah_name_ar", "surah_name_en",
        "surah_name_tr", "ayah_count", "ayah_ar", "ayah_en", "ayah_tr",
        "reciter_id", "reciter_name", "audio",
    ]
    train_ds = train_ds.map(prepare, remove_columns=all_columns)
    train_ds = train_ds.filter(lambda x: x["_valid"])
    train_ds = train_ds.remove_columns(["_valid"])

    # -- Data collator --
    @dataclass
    class DataCollatorCTCWithPadding:
        processor: Any
        padding: Union[bool, str] = True

        def __call__(self, features):
            input_values = [{"input_values": f["input_values"]} for f in features]
            batch = self.processor.feature_extractor.pad(
                input_values, padding=self.padding, return_tensors="pt"
            )

            labels = [f["labels"] for f in features]
            max_len = max(len(l) for l in labels)
            padded = [l + [-100] * (max_len - len(l)) for l in labels]
            batch["labels"] = torch.tensor(padded)
            return batch

    data_collator = DataCollatorCTCWithPadding(processor=processor)

    # -- Training --
    model.gradient_checkpointing_enable()

    training_args = TrainingArguments(
        output_dir=str(CHECKPOINT_DIR),
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        learning_rate=1e-5,  # lower LR for fine-tuning (not from scratch)
        warmup_steps=100,
        max_steps=2000,
        fp16=True,
        logging_steps=25,
        save_steps=500,
        save_total_limit=3,
        remove_unused_columns=False,
        dataloader_pin_memory=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=data_collator,
        processing_class=processor.feature_extractor,
    )

    print("\n" + "=" * 60)
    print(f"  Fine-tuning {BASE_MODEL}")
    print("  2000 steps on A10G, lr=1e-5")
    print("  Dataset: Buraaq/quran-md-ayahs (30 reciters)")
    print("=" * 60 + "\n")

    trainer.train()

    # -- Save with save_pretrained (includes tokenizer + config) --
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving model to {OUTPUT_DIR}...")
    model.save_pretrained(str(OUTPUT_DIR))
    processor.save_pretrained(str(OUTPUT_DIR))
    vol.commit()
    print("Done!")


@app.function(image=image, volumes={"/training": vol})
def download_model():
    from pathlib import Path

    vol.reload()

    model_dir = Path("/training/ctc-model")
    if not model_dir.exists():
        ckpt_dir = Path("/training/ctc-checkpoints")
        if ckpt_dir.exists():
            checkpoints = sorted(ckpt_dir.glob("checkpoint-*"))
            if checkpoints:
                model_dir = checkpoints[-1]
                print(f"No final model found, using checkpoint: {model_dir.name}")
            else:
                print("No checkpoints found!")
                return {}
        else:
            print("No training output found!")
            return {}

    print(f"Downloading from {model_dir}...")
    files = {}
    for f in model_dir.rglob("*"):
        if f.is_file():
            # Skip optimizer/training state (large, not needed for inference)
            if f.name in ("optimizer.pt", "rng_state.pth", "scaler.pt",
                          "scheduler.pt", "training_args.bin"):
                continue
            rel = str(f.relative_to(model_dir))
            files[rel] = f.read_bytes()
            size_mb = len(files[rel]) / (1024 * 1024)
            print(f"  {rel}: {size_mb:.1f} MB")
    return files


@app.local_entrypoint()
def main(download_only: bool = False):
    from pathlib import Path

    if not download_only:
        print("Starting fine-tuning on Modal GPU...")
        train.remote()

    print("\nDownloading model...")
    out_dir = Path("data/ctc-model")
    out_dir.mkdir(parents=True, exist_ok=True)

    files = download_model.remote()
    for name, data in files.items():
        path = out_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        print(f"  Saved {name} ({len(data):,} bytes)")

    print(f"\nModel saved to {out_dir}")
