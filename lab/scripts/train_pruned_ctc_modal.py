"""Fine-tune layer-pruned Rabah CTC models on Quran audio (Modal A100-80GB).

Loads the full 24-layer rabah2026 model, prunes to N layers, and fine-tunes
the CTC head + upper transformer layers on EveryAyah + RetaSy.

Output: Modal volume "ctc-quran-training" → data/rabah-ctc-pruned-{layers}l-{strategy}-finetuned/

Usage:
    modal run --detach scripts/train_pruned_ctc_modal.py --layers 8 --strategy evenly_spaced
    modal run scripts/train_pruned_ctc_modal.py --layers 8 --strategy evenly_spaced --download-only
"""
import modal

app = modal.App("rabah-ctc-pruned-finetune")

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

SOURCE_MODEL = "rabah2026/wav2vec2-large-xlsr-53-arabic-quran-v_final"


def _evenly_spaced_indices(total_layers: int, keep_layers: int) -> list[int]:
    if keep_layers >= total_layers:
        return list(range(total_layers))
    if keep_layers <= 1:
        return [total_layers - 1]
    raw = [
        round(i * (total_layers - 1) / (keep_layers - 1))
        for i in range(keep_layers)
    ]
    fixed = []
    prev = -1
    for idx in raw:
        idx = max(idx, prev + 1)
        fixed.append(min(idx, total_layers - 1))
        prev = fixed[-1]
    return fixed


def _first_n_indices(total_layers: int, keep_layers: int) -> list[int]:
    return list(range(min(keep_layers, total_layers)))


STRATEGIES = {
    "evenly_spaced": _evenly_spaced_indices,
    "first_n": _first_n_indices,
}


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=10800,  # 3 hours
    volumes={"/training": vol},
)
def train(layers: int = 8, strategy: str = "evenly_spaced"):
    import torch
    import json
    from pathlib import Path
    from dataclasses import dataclass
    from typing import Any
    from datasets import load_dataset, Audio, interleave_datasets
    from transformers import (
        Wav2Vec2ForCTC,
        Wav2Vec2Processor,
        Trainer,
        TrainingArguments,
    )
    import numpy as np

    output_name = f"rabah-ctc-pruned-{layers}l-{strategy}-finetuned"
    OUTPUT_DIR = Path(f"/training/{output_name}")
    CHECKPOINT_DIR = Path(f"/training/{output_name}-checkpoints")

    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Load source model ──
    print(f"Loading source model from {SOURCE_MODEL}...")
    processor = Wav2Vec2Processor.from_pretrained(SOURCE_MODEL)
    model = Wav2Vec2ForCTC.from_pretrained(SOURCE_MODEL)

    total_layers = len(model.wav2vec2.encoder.layers)
    print(f"Source model has {total_layers} transformer layers")

    # ── Prune ──
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy '{strategy}'. Choices: {list(STRATEGIES)}")

    strategy_fn = STRATEGIES[strategy]
    keep_indices = strategy_fn(total_layers, layers)
    print(f"Pruning to {layers} layers using '{strategy}': keeping indices {keep_indices}")

    model.wav2vec2.encoder.layers = torch.nn.ModuleList(
        [model.wav2vec2.encoder.layers[i] for i in keep_indices]
    )
    model.config.num_hidden_layers = layers
    model.wav2vec2.config.num_hidden_layers = layers

    # ── Freeze CNN (always) ──
    model.freeze_feature_encoder()

    # ── Freeze lower half of transformer layers ──
    n_layers = len(model.wav2vec2.encoder.layers)
    freeze_up_to = n_layers // 2
    for i, layer in enumerate(model.wav2vec2.encoder.layers):
        if i < freeze_up_to:
            for param in layer.parameters():
                param.requires_grad = False

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,} ({total_params * 4 / 1e6:.0f} MB fp32)")
    print(f"Trainable params: {trainable_params:,} ({trainable_params / total_params * 100:.1f}%)")
    print(f"Frozen: CNN + first {freeze_up_to}/{n_layers} transformer layers")

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

            labels = labels_batch["input_ids"].masked_fill(
                labels_batch.attention_mask.ne(1), -100
            )
            batch["labels"] = labels
            return batch

    data_collator = DataCollatorCTCWithPadding(processor=processor)

    # ── Training ──
    training_args = TrainingArguments(
        output_dir=str(CHECKPOINT_DIR),
        per_device_train_batch_size=16,
        gradient_accumulation_steps=2,
        learning_rate=3e-5,
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
    print(f"  Fine-tuning Rabah CTC ({layers} layers, {strategy})")
    print(f"  Kept layers: {keep_indices}")
    print(f"  LR: 3e-5, batch: 16, grad_accum: 2, steps: 5000")
    print("=" * 60 + "\n")

    trainer.train()

    # ── Save ──
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving model to {OUTPUT_DIR}...")
    model.save_pretrained(str(OUTPUT_DIR))
    processor.save_pretrained(str(OUTPUT_DIR))

    metadata = {
        "source_model": SOURCE_MODEL,
        "layers": layers,
        "strategy": strategy,
        "keep_indices": keep_indices,
        "freeze_up_to": freeze_up_to,
        "learning_rate": 3e-5,
        "max_steps": 5000,
        "batch_size": 16,
        "grad_accum": 2,
    }
    with open(str(OUTPUT_DIR / "training_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    vol.commit()
    print(f"Done! Model saved to Modal volume: {output_name}")


@app.function(
    image=image,
    volumes={"/training": vol},
)
def download_model(layers: int = 8, strategy: str = "evenly_spaced"):
    """Download the trained model files from Modal volume."""
    from pathlib import Path

    output_name = f"rabah-ctc-pruned-{layers}l-{strategy}-finetuned"
    model_dir = Path(f"/training/{output_name}")
    if not model_dir.exists():
        print(f"No model found at {model_dir}! Run training first.")
        return {}

    files = {}
    for f in model_dir.rglob("*"):
        if f.is_file():
            rel = str(f.relative_to(model_dir))
            files[rel] = f.read_bytes()
            print(f"  {rel}: {len(files[rel]):,} bytes")
    return files


@app.local_entrypoint()
def main(
    layers: int = 8,
    strategy: str = "evenly_spaced",
    download_only: bool = False,
):
    from pathlib import Path

    output_name = f"rabah-ctc-pruned-{layers}l-{strategy}-finetuned"

    if not download_only:
        print(f"Starting fine-tuning: {layers} layers, {strategy} pruning...")
        train.remote(layers=layers, strategy=strategy)

    print(f"\nDownloading {output_name}...")
    out_dir = Path(f"data/{output_name}")
    out_dir.mkdir(parents=True, exist_ok=True)

    files = download_model.remote(layers=layers, strategy=strategy)
    if not files:
        print("No files to download.")
        return

    for name, data in files.items():
        path = out_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        print(f"  Saved {name} ({len(data):,} bytes)")

    total = sum(len(d) for d in files.values())
    print(f"\nModel saved to {out_dir} ({total / 1e6:.0f} MB total)")
