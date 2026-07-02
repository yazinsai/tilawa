"""
Train a LoRA adapter on Whisper Base for Quranic recitation.

Dataset: tarteel-ai/everyayah (streamed, not downloaded)
Base model: openai/whisper-base
Output: data/lora-adapter/ (~2-4MB)

Usage:
    python scripts/train_lora.py [--max_steps 5000] [--batch_size 8]
"""
import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Union

import torch
import numpy as np
from datasets import load_dataset, Audio
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from peft import LoraConfig, get_peft_model
import evaluate


# ── Config ──────────────────────────────────────────────────────────────────

BASE_MODEL = "openai/whisper-small"
DATASET_ID = "tarteel-ai/everyayah"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "lora-adapter-small"
CHECKPOINT_DIR = Path(__file__).parent.parent / "data" / "lora-checkpoints-small"


# ── Data Collator ───────────────────────────────────────────────────────────

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int = None

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # Pad input features
        input_features = [
            {"input_features": feature["input_features"]} for feature in features
        ]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        # Pad labels
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(
            label_features, return_tensors="pt"
        )

        # Replace padding with -100 to ignore in loss
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # Cut BOS token if prepended
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


# ── Dataset Preparation ─────────────────────────────────────────────────────

def prepare_dataset(batch, processor):
    """Process a single example from the dataset."""
    audio = batch["audio"]

    # Extract input features
    batch["input_features"] = processor(
        audio["array"],
        sampling_rate=audio["sampling_rate"],
        return_tensors="np",
    ).input_features[0]

    # Tokenize the text (Uthmani script with diacritics)
    batch["labels"] = processor.tokenizer(batch["text"]).input_ids

    return batch


# ── Metrics ─────────────────────────────────────────────────────────────────

def make_compute_metrics(processor):
    wer_metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # Replace -100 with pad token id
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    return compute_metrics


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--save_steps", type=int, default=250)
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Max steps: {args.max_steps}")
    print(f"Batch size: {args.batch_size}")
    est_hours = (args.max_steps * 32) / 3600
    print(f"Estimated time: ~{est_hours:.1f} hours")

    # ── Load processor ──
    print("Loading processor...")
    processor = WhisperProcessor.from_pretrained(
        BASE_MODEL, language="arabic", task="transcribe"
    )

    # ── Load model ──
    print("Loading base model...")
    model = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL)

    # Clear forced decoder IDs for training
    model.config.forced_decoder_ids = None
    model.generation_config.forced_decoder_ids = None

    # ── Apply LoRA ──
    print(f"Applying LoRA (rank={args.lora_rank}, alpha={args.lora_alpha})...")
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Load dataset ──
    local_data = Path(__file__).parent.parent / "data" / "train_subset"
    if local_data.exists():
        print(f"Loading local dataset from {local_data}...")
        from datasets import Dataset as HFDataset
        dataset = HFDataset.load_from_disk(str(local_data))
        dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    else:
        print("Loading dataset (streaming)...")
        dataset = load_dataset(DATASET_ID, split="train", streaming=True)
        dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
        dataset = dataset.filter(lambda x: x["duration"] <= 30.0)

    # Process dataset
    dataset = dataset.map(
        lambda batch: prepare_dataset(batch, processor),
        remove_columns=["audio", "text", "duration", "reciter"],
    )

    # ── Data collator ──
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    # ── Training args ──
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(CHECKPOINT_DIR),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        learning_rate=args.learning_rate,
        warmup_steps=100,
        max_steps=args.max_steps,
        fp16=False,  # MPS does not support FP16
        logging_steps=25,
        save_steps=args.save_steps,
        remove_unused_columns=False,
        dataloader_pin_memory=False,  # MPS compatibility
        report_to="none",
    )

    # ── Trainer ──
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        processing_class=processor.feature_extractor,
    )

    # ── Train ──
    print("\n" + "=" * 60)
    print("  Starting training...")
    print("=" * 60 + "\n")

    resume_from = str(CHECKPOINT_DIR) if args.resume else None
    trainer.train(resume_from_checkpoint=resume_from)

    # ── Save adapter ──
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving LoRA adapter to {OUTPUT_DIR}...")
    model.save_pretrained(str(OUTPUT_DIR))
    processor.save_pretrained(str(OUTPUT_DIR))

    print("Done!")


if __name__ == "__main__":
    main()
