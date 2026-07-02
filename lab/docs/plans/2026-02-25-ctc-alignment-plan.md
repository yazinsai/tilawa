# CTC Alignment Experiment Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a CTC forced-alignment experiment that identifies Quran verses by scoring audio against known text, replacing free-form transcription with constrained matching.

**Architecture:** wav2vec2 CTC encoder produces frame-level character logits from audio. Greedy decode gives rough text for candidate pruning (Levenshtein top-50). Then CTC forward algorithm scores each candidate against the frame logits. Best score wins. No decoder, no hallucination.

**Tech Stack:** PyTorch (MPS), transformers (Wav2Vec2ForCTC), Modal (GPU training), shared.quran_db, shared.normalizer

---

### Task 1: CTC scorer utility

**Files:**
- Create: `experiments/ctc-alignment/ctc_scorer.py`

**Step 1: Implement batch CTC scoring**

```python
"""CTC forced-alignment scorer.

Given frame-level log-probabilities and candidate texts,
scores how well each candidate explains the audio using
the CTC forward algorithm.
"""
import torch
import torch.nn.functional as F


def score_candidates(
    logits: torch.Tensor,
    candidates: list[dict],
    tokenize_fn: callable,
    blank_id: int,
) -> list[tuple[dict, float]]:
    """Score candidate verses against audio frame logits.

    Args:
        logits: (1, T, V) raw model output
        candidates: list of verse dicts with "text_clean" field
        tokenize_fn: text -> list[int] character index mapping
        blank_id: CTC blank token index

    Returns:
        List of (candidate, score) sorted best-first.
        Score is normalized negative log-likelihood (lower = better match).
    """
    log_probs = F.log_softmax(logits, dim=-1)  # (1, T, V)
    log_probs = log_probs.squeeze(0).T  # NOT this
    # CTC loss expects (T, N, C)
    log_probs = F.log_softmax(logits, dim=-1).permute(1, 0, 2)  # (T, 1, V)
    T = log_probs.size(0)

    # Tokenize all candidates
    encoded = []
    target_lengths = []
    for c in candidates:
        ids = tokenize_fn(c["text_clean"])
        if len(ids) == 0:
            ids = [blank_id]  # fallback for empty text
        encoded.append(ids)
        target_lengths.append(len(ids))

    N = len(candidates)
    # Expand log_probs for batch: (T, 1, V) -> (T, N, V)
    log_probs_batch = log_probs.expand(T, N, -1).contiguous()
    input_lengths = torch.full((N,), T, dtype=torch.long)
    target_lengths_t = torch.tensor(target_lengths, dtype=torch.long)

    # Concatenate targets (CTC loss accepts 1D concatenated targets)
    all_targets = torch.tensor(
        [idx for seq in encoded for idx in seq], dtype=torch.long
    )

    # Batch CTC scoring
    losses = F.ctc_loss(
        log_probs_batch,
        all_targets,
        input_lengths,
        target_lengths_t,
        blank=blank_id,
        reduction="none",
        zero_infinity=True,
    )  # (N,)

    # Normalize by input length to make scores comparable across chunks
    scores = (losses / T).tolist()

    results = list(zip(candidates, scores))
    results.sort(key=lambda x: x[1])  # lower = better
    return results
```

**Step 2: Verify scorer runs**

```bash
.venv/bin/python -c "
import torch, sys; sys.path.insert(0, '.')
from experiments.ctc_alignment.ctc_scorer import score_candidates
# Smoke test with random logits
logits = torch.randn(1, 50, 32)  # 50 frames, 32 vocab
candidates = [{'text_clean': 'abc'}, {'text_clean': 'xyz'}]
tokenize = lambda t: [ord(c) % 30 + 1 for c in t]
results = score_candidates(logits, candidates, tokenize, blank_id=0)
print(f'Scored {len(results)} candidates, best={results[0][1]:.3f}')
"
```

Expected: prints score for 2 candidates without error.

---

### Task 2: Experiment run.py with pre-trained Arabic CTC

**Files:**
- Create: `experiments/ctc-alignment/run.py`

Use `jonatasgrosman/wav2vec2-large-xlsr-53-arabic` — already fine-tuned for Arabic ASR with CTC head. 315M params. No training needed for baseline.

**Step 1: Implement run.py**

```python
import sys
import math
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from shared.audio import load_audio
from shared.quran_db import QuranDB
from shared.normalizer import normalize_arabic
from experiments.ctc_alignment.ctc_scorer import score_candidates

MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-arabic"
TOP_K = 50  # candidates for CTC re-scoring

_model = None
_processor = None
_db = None
_device = None


def _ensure_loaded():
    global _model, _processor, _db, _device
    if _model is not None:
        return
    _processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
    _model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)
    _model.eval()
    _device = "mps" if torch.backends.mps.is_available() else "cpu"
    _model.to(_device)
    _db = QuranDB()


def _tokenize_for_ctc(text: str) -> list[int]:
    """Convert Arabic text to token IDs using model's tokenizer."""
    # Wav2Vec2CTCTokenizer maps characters to IDs
    encoded = _processor.tokenizer(text, return_tensors=None)
    return encoded["input_ids"]


def predict(audio_path: str) -> dict:
    _ensure_loaded()
    audio = load_audio(audio_path)

    # 1. Encode audio → frame-level logits
    inputs = _processor(
        audio, sampling_rate=16000, return_tensors="pt", padding=True
    )
    inputs = {k: v.to(_device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = _model(**inputs).logits  # (1, T, V)

    # 2. Greedy decode → rough text for candidate pruning
    pred_ids = torch.argmax(logits, dim=-1)[0]
    rough_text = _processor.decode(pred_ids.cpu().numpy())
    rough_text_normalized = normalize_arabic(rough_text)

    if not rough_text_normalized.strip():
        return {
            "surah": 0, "ayah": 0, "ayah_end": None,
            "score": 0.0, "transcript": "",
        }

    # 3. Prune: Levenshtein top-K candidates
    candidates = _db.search(rough_text_normalized, top_k=TOP_K)

    # 4. CTC re-score each candidate
    blank_id = _processor.tokenizer.pad_token_id
    scored = score_candidates(
        logits.cpu(),
        candidates,
        _tokenize_for_ctc,
        blank_id=blank_id,
    )

    if not scored:
        return {
            "surah": 0, "ayah": 0, "ayah_end": None,
            "score": 0.0, "transcript": rough_text_normalized,
        }

    best_candidate, best_loss = scored[0]

    # Convert normalized CTC loss to 0-1 confidence
    confidence = math.exp(-best_loss)

    return {
        "surah": best_candidate["surah"],
        "ayah": best_candidate["ayah"],
        "ayah_end": best_candidate.get("ayah_end"),
        "score": round(confidence, 4),
        "transcript": rough_text_normalized,
    }


def model_size() -> int:
    """wav2vec2-large-xlsr-53-arabic ~1.2GB."""
    return 1_200 * 1024 * 1024
```

**Step 2: Test single prediction**

```bash
.venv/bin/python -c "
import sys; sys.path.insert(0, '.')
from experiments.ctc_alignment.run import predict
result = predict('benchmark/test_corpus/001001.mp3')
print(result)
"
```

Expected: returns a dict with surah/ayah/score. May or may not be correct — that's what the benchmark tells us.

---

### Task 3: Register in benchmark + run

**Files:**
- Modify: `benchmark/runner.py` — add to EXPERIMENT_REGISTRY

**Step 1: Add registry entry**

In `benchmark/runner.py`, add to `EXPERIMENT_REGISTRY`:

```python
"ctc-alignment": EXPERIMENTS_DIR / "ctc-alignment" / "run.py",
```

**Step 2: Run benchmark for this experiment only**

```bash
.venv/bin/python -m benchmark.runner --experiment ctc-alignment
```

This gives us the pre-trained baseline accuracy against all 37 test samples. Record the number.

**Step 3: Commit baseline experiment**

```bash
git add experiments/ctc-alignment/ctc_scorer.py experiments/ctc-alignment/run.py benchmark/runner.py
git commit -m "feat: add ctc-alignment experiment with pre-trained Arabic CTC model"
```

---

### Task 4: Modal training script for Quran CTC

**Files:**
- Create: `scripts/train_ctc_modal.py`

Train `wav2vec2-xls-r-300m` with a CTC head on `Buraaq/quran-md-ayahs` (187K samples, 30 reciters).

**Step 1: Write Modal training script**

Key differences from existing `train_modal.py`:
- `Wav2Vec2ForCTC` instead of Whisper
- CTC loss instead of seq2seq
- Custom Arabic character vocabulary (normalized, no diacritics)
- `Buraaq/quran-md-ayahs` dataset
- Freeze feature extractor, fine-tune transformer + CTC head

```python
"""
Train wav2vec2 CTC model on Quran audio (Modal A10G).

Usage:
    modal run scripts/train_ctc_modal.py
"""
import modal

app = modal.App("wav2vec2-quran-ctc")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
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

vol = modal.Volume.from_name("wav2vec2-quran-ctc", create_if_missing=True)

# Arabic characters after normalization (shared.normalizer compatible)
VOCAB_CHARS = list("ابتثجحخدذرزسشصضطظعغفقكلمنهويء ")
# Build vocab: <pad>=0, <s>=1, </s>=2, <unk>=3, | (word boundary)=4, then chars
VOCAB = {"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3, "|": 4}
for i, c in enumerate(VOCAB_CHARS):
    VOCAB[c] = i + 5


@app.function(
    image=image,
    gpu="A10G",
    timeout=10800,  # 3 hours
    volumes={"/training": vol},
)
def train():
    import json
    import re
    import torch
    from pathlib import Path
    from dataclasses import dataclass
    from typing import Any, Dict, List, Union
    from datasets import load_dataset, Audio
    from transformers import (
        Wav2Vec2ForCTC,
        Wav2Vec2CTCTokenizer,
        Wav2Vec2FeatureExtractor,
        Wav2Vec2Processor,
        TrainingArguments,
        Trainer,
    )
    import evaluate

    OUTPUT_DIR = Path("/training/ctc-model")
    CHECKPOINT_DIR = Path("/training/ctc-checkpoints")

    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Build vocab + tokenizer ──
    vocab_path = Path("/tmp/vocab.json")
    vocab_path.write_text(json.dumps(VOCAB))

    tokenizer = Wav2Vec2CTCTokenizer(
        str(vocab_path),
        unk_token="<unk>",
        pad_token="<pad>",
        word_delimiter_token="|",
    )

    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=16000,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=True,
    )

    processor = Wav2Vec2Processor(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
    )

    # ── Load model ──
    print("Loading wav2vec2-xls-r-300m...")
    model = Wav2Vec2ForCTC.from_pretrained(
        "facebook/wav2vec2-xls-r-300m",
        attention_dropout=0.1,
        hidden_dropout=0.1,
        feat_proj_dropout=0.0,
        mask_time_prob=0.05,
        layerdrop=0.1,
        ctc_loss_reduction="mean",
        pad_token_id=tokenizer.pad_token_id,
        vocab_size=len(tokenizer),
        ctc_zero_infinity=True,
    )

    # Freeze feature extractor (CNN layers)
    model.freeze_feature_extractor()

    # ── Arabic normalization (matches shared.normalizer) ──
    _DIACRITICS = re.compile(
        '[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06DC'
        '\u06DF-\u06E4\u06E7\u06E8\u06EA-\u06ED\u0640]'
    )
    _NORM_MAP = str.maketrans({
        '\u0623': '\u0627',  # أ -> ا
        '\u0625': '\u0627',  # إ -> ا
        '\u0622': '\u0627',  # آ -> ا
        '\u0671': '\u0627',  # ٱ -> ا
        '\u0629': '\u0647',  # ة -> ه
        '\u0649': '\u064A',  # ى -> ي
    })

    def normalize_arabic(text):
        text = _DIACRITICS.sub('', text)
        text = text.translate(_NORM_MAP)
        text = ' '.join(text.split())
        return text

    # ── Load dataset ──
    print("Loading Buraaq/quran-md-ayahs...")
    ds = load_dataset("Buraaq/quran-md-ayahs", split="train", streaming=True)
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))

    # Hold out 3 reciters for validation
    VAL_RECITERS = {"saood_ash_shuraym", "abdulbaset_abdulsamad_mojawwad", "hani_ar_rifai"}

    def is_train(example):
        return example["reciter_id"] not in VAL_RECITERS

    def is_val(example):
        return example["reciter_id"] in VAL_RECITERS

    train_ds = ds.filter(is_train)
    # For val, take a finite subset (streaming dataset)
    val_ds = ds.filter(is_val).take(2000)

    def prepare(batch):
        audio = batch["audio"]
        batch["input_values"] = processor(
            audio["array"],
            sampling_rate=16000,
            return_tensors=None,
        ).input_values[0]

        # Normalize Arabic text (strip diacritics, normalize chars)
        text = normalize_arabic(batch["ayah_ar"])
        # Replace spaces with word delimiter for CTC
        text_with_delim = text.replace(" ", "|")
        batch["labels"] = processor.tokenizer(text_with_delim).input_ids
        return batch

    train_ds = train_ds.map(prepare, remove_columns=[
        "surah_id", "ayah_id", "surah_name_ar", "surah_name_en",
        "surah_name_tr", "ayah_count", "ayah_ar", "ayah_en", "ayah_tr",
        "reciter_id", "reciter_name", "audio",
    ])

    # ── Data collator ──
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

    # ── Training ──
    training_args = TrainingArguments(
        output_dir=str(CHECKPOINT_DIR),
        per_device_train_batch_size=16,
        gradient_accumulation_steps=2,
        learning_rate=3e-5,
        warmup_steps=500,
        max_steps=5000,
        fp16=True,
        logging_steps=50,
        save_steps=1000,
        group_by_length=False,  # streaming dataset
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
    print("  Starting CTC training (5000 steps on A10G)")
    print("  Base: wav2vec2-xls-r-300m")
    print("  Dataset: Buraaq/quran-md-ayahs (30 reciters)")
    print("=" * 60 + "\n")

    trainer.train()

    # ── Save ──
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving model to {OUTPUT_DIR}...")
    model.save_pretrained(str(OUTPUT_DIR))
    processor.save_pretrained(str(OUTPUT_DIR))
    vol.commit()
    print("Done!")


@app.function(image=image, volumes={"/training": vol})
def download_model():
    from pathlib import Path
    model_dir = Path("/training/ctc-model")
    if not model_dir.exists():
        print("No model found! Run training first.")
        return {}

    files = {}
    for f in model_dir.rglob("*"):
        if f.is_file():
            rel = str(f.relative_to(model_dir))
            files[rel] = f.read_bytes()
            size_mb = len(files[rel]) / (1024 * 1024)
            print(f"  {rel}: {size_mb:.1f} MB")
    return files


@app.local_entrypoint()
def main():
    from pathlib import Path

    print("Starting CTC training on Modal GPU...")
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
```

**Step 2: Verify script parses**

```bash
.venv/bin/python -c "import ast; ast.parse(open('scripts/train_ctc_modal.py').read()); print('OK')"
```

**Step 3: Commit training script**

```bash
git add scripts/train_ctc_modal.py
git commit -m "feat: add Modal training script for wav2vec2 CTC on Quran"
```

---

### Task 5: Run training on Modal

**Step 1: Launch training**

```bash
cd /Users/rock/ai/projects/offline-tarteel
modal run scripts/train_ctc_modal.py
```

This trains on A10G, downloads model to `data/ctc-model/`. Takes ~1-2 hours.

**Step 2: Verify model downloaded**

```bash
ls -la data/ctc-model/
```

Expected: `config.json`, `model.safetensors` (or `pytorch_model.bin`), `preprocessor_config.json`, `tokenizer_config.json`, `vocab.json`, etc.

---

### Task 6: Update experiment to use fine-tuned model + re-benchmark

**Files:**
- Modify: `experiments/ctc-alignment/run.py`

**Step 1: Add model path switching**

Update `run.py` to try local fine-tuned model first, fall back to pre-trained:

```python
LOCAL_MODEL = PROJECT_ROOT / "data" / "ctc-model"
PRETRAINED_MODEL = "jonatasgrosman/wav2vec2-large-xlsr-53-arabic"

def _ensure_loaded():
    global _model, _processor, _db, _device
    if _model is not None:
        return
    model_path = str(LOCAL_MODEL) if LOCAL_MODEL.exists() else PRETRAINED_MODEL
    print(f"Loading CTC model from {model_path}...")
    _processor = Wav2Vec2Processor.from_pretrained(model_path)
    _model = Wav2Vec2ForCTC.from_pretrained(model_path)
    ...
```

**Step 2: Run benchmark**

```bash
.venv/bin/python -m benchmark.runner --experiment ctc-alignment
```

**Step 3: Compare against whisper-lora**

```bash
.venv/bin/python -m benchmark.runner --experiment whisper-lora
```

Compare accuracy. Target: beat 78% (whisper-lora's current score on 37 samples).

**Step 4: Commit results**

```bash
git add experiments/ctc-alignment/run.py
git commit -m "feat: ctc-alignment experiment with fine-tuned Quran CTC model"
```

---

## Summary

| Task | What | Depends On |
|------|------|-----------|
| 1 | CTC scorer utility | — |
| 2 | Experiment run.py (pre-trained baseline) | Task 1 |
| 3 | Register + benchmark baseline | Task 2 |
| 4 | Modal training script | — |
| 5 | Run training on Modal | Task 4 |
| 6 | Switch to fine-tuned model + benchmark | Tasks 3, 5 |

Tasks 1-3 give us a working baseline with no training. Tasks 4-6 improve accuracy with Quran-specific fine-tuning. Tasks 1+4 can be done in parallel.
