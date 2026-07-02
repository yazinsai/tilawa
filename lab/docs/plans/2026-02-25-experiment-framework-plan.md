# Experiment Framework Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Restructure offline-tarteel so all approaches live under `experiments/` with a standardized `run.py` contract, and a single benchmark runner compares them.

**Architecture:** Convention-based experiment folders, each with `predict(audio_path) -> dict` and `model_size() -> int`. A `benchmark/runner.py` discovers experiments, runs them against a shared test corpus, and outputs a comparison table + JSON.

**Tech Stack:** Python 3.11+, HuggingFace datasets (for RetaSy corpus curation), tabulate (for CLI table output)

---

### Task 1: Create shared/ utilities

**Files:**
- Create: `shared/__init__.py`
- Create: `shared/audio.py` (copy from `src/offline_tarteel/audio.py`)
- Create: `shared/normalizer.py` (copy from `src/offline_tarteel/normalizer.py`)
- Create: `shared/quran_db.py` (adapted from `src/offline_tarteel/quran_db.py`)

**Step 1: Create shared directory and copy files**

```bash
mkdir -p shared
```

`shared/__init__.py`:
```python
from .audio import load_audio
from .normalizer import normalize_arabic
from .quran_db import QuranDB
```

`shared/audio.py` — identical to `src/offline_tarteel/audio.py`:
```python
import numpy as np
import librosa

TARGET_SR = 16000


def load_audio(path: str, sr: int = TARGET_SR) -> np.ndarray:
    audio, _ = librosa.load(path, sr=sr, mono=True)
    return audio.astype(np.float32)
```

`shared/normalizer.py` — identical to `src/offline_tarteel/normalizer.py`:
```python
import re

_DIACRITICS = re.compile(
    '[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06DC'
    '\u06DF-\u06E4\u06E7\u06E8\u06EA-\u06ED\u0640]'
)

_NORM_MAP = str.maketrans({
    '\u0623': '\u0627',
    '\u0625': '\u0627',
    '\u0622': '\u0627',
    '\u0671': '\u0627',
    '\u0629': '\u0647',
    '\u0649': '\u064A',
})


def normalize_arabic(text: str) -> str:
    text = _DIACRITICS.sub('', text)
    text = text.translate(_NORM_MAP)
    text = ' '.join(text.split())
    return text
```

`shared/quran_db.py` — adapted with project-root-relative DATA_PATH:
```python
import json
from pathlib import Path
from Levenshtein import ratio
from shared.normalizer import normalize_arabic

# Resolve to project root / data / quran.json
DATA_PATH = Path(__file__).parent.parent / "data" / "quran.json"


class QuranDB:
    # ... (rest identical to src/offline_tarteel/quran_db.py)
```

**Step 2: Verify shared imports work**

```bash
cd /Users/rock/ai/projects/offline-tarteel
python -c "import sys; sys.path.insert(0, '.'); from shared import load_audio, normalize_arabic, QuranDB; db = QuranDB(); print(f'Loaded {db.total_verses} verses')"
```

Expected: `Loaded 6236 verses`

**Step 3: Commit**

```bash
git add shared/__init__.py shared/audio.py shared/normalizer.py shared/quran_db.py
git commit -m "feat: create shared/ utilities extracted from src/offline_tarteel"
```

---

### Task 2: Create experiments/whisper-lora/run.py

**Files:**
- Create: `experiments/whisper-lora/run.py`

**Context:** The base approach uses Whisper-small + LoRA adapter. Model loading is in `scripts/eval_lora.py` (lines 47-79) and `scripts/benchmark.py` (lines 80-133). The LoRA adapter lives at `data/lora-adapter-small/`.

**Step 1: Create run.py**

```python
import sys
from pathlib import Path

# Add project root to path for shared imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel
from shared.audio import load_audio
from shared.quran_db import QuranDB

MODEL_ID = "openai/whisper-small"
ADAPTER_PATH = PROJECT_ROOT / "data" / "lora-adapter-small"

_model = None
_processor = None
_db = None


def _ensure_loaded():
    global _model, _processor, _db
    if _model is not None:
        return
    _processor = WhisperProcessor.from_pretrained(MODEL_ID, language="arabic", task="transcribe")
    base = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)
    _model = PeftModel.from_pretrained(base, str(ADAPTER_PATH))
    _model.eval()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    _model.to(device)
    _db = QuranDB()


def predict(audio_path: str) -> dict:
    _ensure_loaded()
    audio = load_audio(audio_path)
    inputs = _processor(audio, sampling_rate=16000, return_tensors="pt")
    inputs = {k: v.to(_model.device) for k, v in inputs.items()}

    with torch.no_grad():
        ids = _model.generate(
            inputs["input_features"],
            max_new_tokens=225,
            repetition_penalty=1.2,
            language="ar",
            task="transcribe",
        )
    transcript = _processor.batch_decode(ids, skip_special_tokens=True)[0]
    match = _db.match_verse(transcript)

    if match is None:
        return {"surah": 0, "ayah": 0, "ayah_end": None, "score": 0.0, "transcript": transcript}

    return {
        "surah": match["surah"],
        "ayah": match["ayah"],
        "ayah_end": match.get("ayah_end"),
        "score": match["score"],
        "transcript": transcript,
    }


def model_size() -> int:
    """Whisper-small base + LoRA adapter."""
    total = 0
    # HF cache for whisper-small (~461MB) — estimate from known size
    total += 461 * 1024 * 1024
    # LoRA adapter on disk
    adapter = PROJECT_ROOT / "data" / "lora-adapter-small"
    if adapter.exists():
        total += sum(f.stat().st_size for f in adapter.rglob("*") if f.is_file())
    return total
```

**Step 2: Verify it loads and runs**

```bash
cd /Users/rock/ai/projects/offline-tarteel
python -c "
import sys; sys.path.insert(0, '.')
from experiments.whisper_lora import run
result = run.predict('data/reference_audio/001001.mp3')
print(result)
"
```

Expected: dict with surah=1, ayah=1

**Step 3: Commit**

```bash
git add experiments/whisper-lora/run.py
git commit -m "feat: add whisper-lora experiment with standardized run.py"
```

---

### Task 3: Create experiments/embedding-search/run.py

**Files:**
- Create: `experiments/embedding-search/run.py`

**Context:** Existing `search.py` has `search_verse()` (line 89) which takes audio_path and returns results. Wrap it to match the contract.

**Step 1: Create run.py**

```python
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.audio import load_audio

# Import from existing search module in same directory
sys.path.insert(0, str(Path(__file__).parent))
from search import load_search_model, load_index, search_verse

EXPERIMENT_DIR = Path(__file__).parent
DEFAULT_MODEL = "hubert"
DEFAULT_RECITER = "alafasy"

_loaded = False


def _ensure_loaded():
    global _loaded
    if _loaded:
        return
    load_search_model(DEFAULT_MODEL)
    load_index(DEFAULT_MODEL, DEFAULT_RECITER)
    _loaded = True


def predict(audio_path: str) -> dict:
    _ensure_loaded()
    results, _, _ = search_verse(audio_path, model_name=DEFAULT_MODEL, reciter=DEFAULT_RECITER, top_k=1)

    if not results:
        return {"surah": 0, "ayah": 0, "ayah_end": None, "score": 0.0, "transcript": ""}

    best = results[0]
    return {
        "surah": best["surah"],
        "ayah": best["ayah"],
        "ayah_end": None,
        "score": best["score"],
        "transcript": "",
    }


def model_size() -> int:
    """HuBERT-base model + FAISS index."""
    total = 360 * 1024 * 1024  # HuBERT-base ~360MB
    # FAISS index on disk
    index_dir = EXPERIMENT_DIR / "indices"
    if index_dir.exists():
        total += sum(f.stat().st_size for f in index_dir.rglob("*") if f.is_file())
    return total
```

**Step 2: Verify**

```bash
cd /Users/rock/ai/projects/offline-tarteel
python -c "
import sys; sys.path.insert(0, '.')
from experiments.embedding_search import run
print(run.model_size())
"
```

**Step 3: Commit**

```bash
git add experiments/embedding-search/run.py
git commit -m "feat: add embedding-search run.py wrapper"
```

---

### Task 4: Create experiments/contrastive/run.py

**Files:**
- Create: `experiments/contrastive/run.py`

**Context:** Existing `evaluate.py` has `encode_audio_file()` (line 127) and `find_nearest_verses()` (line 151). The model is QuranCLAP from `model.py`. Needs a trained checkpoint in `checkpoints/`.

**Step 1: Create run.py**

```python
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

import torch
from model import QuranCLAP
from evaluate import precompute_text_embeddings, encode_audio_file, find_nearest_verses

EXPERIMENT_DIR = Path(__file__).parent
CHECKPOINT = EXPERIMENT_DIR / "checkpoints" / "best_model.pt"

_model = None
_text_embeddings = None
_verse_refs = None


def _ensure_loaded():
    global _model, _text_embeddings, _verse_refs
    if _model is not None:
        return

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    _model = QuranCLAP()

    if CHECKPOINT.exists():
        state = torch.load(str(CHECKPOINT), map_location=device)
        _model.load_state_dict(state["model_state_dict"] if "model_state_dict" in state else state)

    _model.to(device)
    _model.eval()
    _text_embeddings, _verse_refs = precompute_text_embeddings(_model, device)


def predict(audio_path: str) -> dict:
    _ensure_loaded()
    device = next(_model.parameters()).device
    audio_emb = encode_audio_file(_model, audio_path, device)
    results = find_nearest_verses(audio_emb, _text_embeddings, _verse_refs, top_k=1)

    if not results:
        return {"surah": 0, "ayah": 0, "ayah_end": None, "score": 0.0, "transcript": ""}

    best = results[0]
    return {
        "surah": best["surah"],
        "ayah": best["ayah"],
        "ayah_end": None,
        "score": best["score"],
        "transcript": "",
    }


def model_size() -> int:
    """HuBERT-base + AraBERT + projection heads + checkpoint."""
    total = 360 * 1024 * 1024  # HuBERT-base
    total += 540 * 1024 * 1024  # AraBERT
    if CHECKPOINT.exists():
        total += CHECKPOINT.stat().st_size
    return total
```

**Step 2: Commit**

```bash
git add experiments/contrastive/run.py
git commit -m "feat: add contrastive run.py wrapper"
```

---

### Task 5: Create experiments/streaming-asr/run.py

**Files:**
- Create: `experiments/streaming-asr/run.py`

**Context:** Existing `streaming_transcriber.py` has `StreamingTranscriber` class with `transcribe_batch()` method (line 229). For the batch-mode contract, we use batch transcription and then match.

**Step 1: Create run.py**

```python
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from streaming_transcriber import StreamingTranscriber
from shared.quran_db import QuranDB

_transcriber = None
_db = None


def _ensure_loaded():
    global _transcriber, _db
    if _transcriber is not None:
        return
    _transcriber = StreamingTranscriber(chunk_seconds=3.0, overlap_seconds=0.0)
    _db = QuranDB()


def predict(audio_path: str) -> dict:
    _ensure_loaded()
    result = _transcriber.transcribe_batch(audio_path)
    transcript = result.get("accumulated_text", "") if isinstance(result, dict) else ""

    match = _db.match_verse(transcript)
    if match is None:
        return {"surah": 0, "ayah": 0, "ayah_end": None, "score": 0.0, "transcript": transcript}

    return {
        "surah": match["surah"],
        "ayah": match["ayah"],
        "ayah_end": match.get("ayah_end"),
        "score": match["score"],
        "transcript": transcript,
    }


def model_size() -> int:
    """mlx-whisper base model."""
    return 461 * 1024 * 1024  # whisper-base ~461MB
```

**Step 2: Commit**

```bash
git add experiments/streaming-asr/run.py
git commit -m "feat: add streaming-asr run.py wrapper"
```

---

### Task 6: Create experiments/new-models/run.py

**Files:**
- Create: `experiments/new-models/run.py`

**Context:** Existing `experiments/new-models-benchmark/benchmark.py` has separate test functions per model. We need to rename the folder to `new-models/` and create a unified `run.py` with `list_models()`, `predict(audio_path, model_name=...)`, and `model_size(model_name=...)`.

**Step 1: Rename directory**

```bash
cd /Users/rock/ai/projects/offline-tarteel
mv experiments/new-models-benchmark experiments/new-models
```

**Step 2: Create run.py**

This wraps the model-specific logic from `benchmark.py`. Each model has its own loader and predict path.

```python
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
from shared.audio import load_audio
from shared.quran_db import QuranDB

MODELS = {
    "whisper-large-v3-turbo": {
        "hf_id": "openai/whisper-large-v3-turbo",
        "type": "whisper",
        "size_mb": 3000,
    },
    "tarteel-whisper-base": {
        "hf_id": "tarteel-ai/whisper-base-ar-quran",
        "type": "whisper",
        "size_mb": 277,
    },
    "moonshine-tiny-arabic": {
        "hf_id": "UsefulSensors/moonshine-tiny-ar",
        "type": "moonshine",
        "size_mb": 103,
    },
    "mms-1b-all": {
        "hf_id": "facebook/mms-1b-all",
        "type": "mms",
        "size_mb": 3700,
    },
    "wav2vec2-quran-hamza": {
        "hf_id": "HamzaSidhu786/wav2vec2-base-word-by-word-quran-asr",
        "type": "wav2vec2",
        "size_mb": 360,
    },
}

_loaded_models = {}
_db = None


def list_models() -> list[str]:
    return list(MODELS.keys())


def _ensure_db():
    global _db
    if _db is None:
        _db = QuranDB()


def _load_model(model_name: str):
    if model_name in _loaded_models:
        return _loaded_models[model_name]

    cfg = MODELS[model_name]
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    if cfg["type"] == "whisper":
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        processor = WhisperProcessor.from_pretrained(cfg["hf_id"])
        model = WhisperForConditionalGeneration.from_pretrained(cfg["hf_id"]).to(device)
        model.eval()
        _loaded_models[model_name] = ("whisper", processor, model, device)

    elif cfg["type"] == "moonshine":
        from transformers import AutoProcessor, MoonshineForConditionalGeneration
        processor = AutoProcessor.from_pretrained(cfg["hf_id"])
        model = MoonshineForConditionalGeneration.from_pretrained(cfg["hf_id"]).to(device)
        model.eval()
        _loaded_models[model_name] = ("moonshine", processor, model, device)

    elif cfg["type"] == "mms":
        from transformers import Wav2Vec2ForCTC, AutoProcessor
        processor = AutoProcessor.from_pretrained(cfg["hf_id"], target_lang="ara")
        model = Wav2Vec2ForCTC.from_pretrained(cfg["hf_id"], target_lang="ara").to(device)
        model.eval()
        _loaded_models[model_name] = ("mms", processor, model, device)

    elif cfg["type"] == "wav2vec2":
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
        processor = Wav2Vec2Processor.from_pretrained(cfg["hf_id"])
        model = Wav2Vec2ForCTC.from_pretrained(cfg["hf_id"]).to(device)
        model.eval()
        _loaded_models[model_name] = ("wav2vec2", processor, model, device)

    return _loaded_models[model_name]


def predict(audio_path: str, model_name: str = "moonshine-tiny-arabic") -> dict:
    _ensure_db()
    model_type, processor, model, device = _load_model(model_name)
    audio = load_audio(audio_path)

    with torch.no_grad():
        if model_type in ("whisper", "moonshine"):
            inputs = processor(audio, sampling_rate=16000, return_tensors="pt").to(device)
            ids = model.generate(inputs["input_features"], max_new_tokens=225)
            transcript = processor.batch_decode(ids, skip_special_tokens=True)[0]

        elif model_type in ("mms", "wav2vec2"):
            inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True).to(device)
            logits = model(**inputs).logits
            ids = torch.argmax(logits, dim=-1)
            transcript = processor.batch_decode(ids)[0]

    match = _db.match_verse(transcript)
    if match is None:
        return {"surah": 0, "ayah": 0, "ayah_end": None, "score": 0.0, "transcript": transcript}

    return {
        "surah": match["surah"],
        "ayah": match["ayah"],
        "ayah_end": match.get("ayah_end"),
        "score": match["score"],
        "transcript": transcript,
    }


def model_size(model_name: str = "moonshine-tiny-arabic") -> int:
    return MODELS[model_name]["size_mb"] * 1024 * 1024
```

**Step 3: Commit**

```bash
git add experiments/new-models/run.py
git commit -m "feat: add new-models run.py with multi-model support"
```

---

### Task 7: Set up benchmark infrastructure

**Files:**
- Create: `benchmark/__init__.py`
- Create: `benchmark/runner.py`
- Create: `benchmark/test_corpus/.gitkeep`
- Create: `benchmark/results/.gitkeep`

**Step 1: Create directories**

```bash
mkdir -p benchmark/test_corpus benchmark/results
```

**Step 2: Create benchmark/runner.py**

```python
"""
Benchmark runner for all experiments.

Usage:
    python -m benchmark.runner                           # all experiments
    python -m benchmark.runner --experiment whisper-lora  # one experiment
    python -m benchmark.runner --category short           # filter by category
"""

import sys
import json
import time
import importlib
import argparse
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
CORPUS_DIR = Path(__file__).parent / "test_corpus"
RESULTS_DIR = Path(__file__).parent / "results"

# Experiments and their module paths (relative to project root)
EXPERIMENT_REGISTRY = {
    "whisper-lora": "experiments.whisper_lora.run",
    "embedding-search": "experiments.embedding_search.run",
    "contrastive": "experiments.contrastive.run",
    "streaming-asr": "experiments.streaming_asr.run",
}

# new-models gets expanded into one entry per model
NEW_MODELS_MODULE = "experiments.new_models.run"


def load_manifest() -> list[dict]:
    manifest_path = CORPUS_DIR / "manifest.json"
    with open(manifest_path) as f:
        data = json.load(f)
    return data["samples"]


def discover_experiments(filter_name: str | None = None) -> list[dict]:
    """Return list of {name, module_path, model_name (optional)}."""
    experiments = []

    for name, module_path in EXPERIMENT_REGISTRY.items():
        if filter_name and filter_name != name:
            continue
        experiments.append({"name": name, "module_path": module_path, "model_name": None})

    # Expand new-models
    if not filter_name or filter_name.startswith("new-models"):
        try:
            # Add project root for imports
            mod = importlib.import_module(NEW_MODELS_MODULE)
            for model_name in mod.list_models():
                entry_name = f"new-models/{model_name}"
                if filter_name and filter_name != entry_name and filter_name != "new-models":
                    continue
                experiments.append({
                    "name": entry_name,
                    "module_path": NEW_MODELS_MODULE,
                    "model_name": model_name,
                })
        except Exception as e:
            print(f"Warning: could not load new-models: {e}")

    return experiments


def run_experiment(exp: dict, samples: list[dict]) -> dict:
    """Run one experiment against all samples. Returns results dict."""
    mod = importlib.import_module(exp["module_path"])

    # Warmup call
    warmup_sample = samples[0]
    audio_path = str(CORPUS_DIR / warmup_sample["file"])
    try:
        if exp["model_name"]:
            mod.predict(audio_path, model_name=exp["model_name"])
        else:
            mod.predict(audio_path)
    except Exception as e:
        print(f"  Warmup failed for {exp['name']}: {e}")

    # Get model size
    try:
        if exp["model_name"]:
            size = mod.model_size(model_name=exp["model_name"])
        else:
            size = mod.model_size()
    except Exception:
        size = 0

    correct = 0
    total = len(samples)
    latencies = []
    per_sample = []

    for sample in samples:
        audio_path = str(CORPUS_DIR / sample["file"])
        try:
            start = time.perf_counter()
            if exp["model_name"]:
                result = mod.predict(audio_path, model_name=exp["model_name"])
            else:
                result = mod.predict(audio_path)
            elapsed = time.perf_counter() - start
        except Exception as e:
            result = {"surah": 0, "ayah": 0, "ayah_end": None, "score": 0.0, "transcript": f"ERROR: {e}"}
            elapsed = 0.0

        expected = (sample["surah"], sample["ayah"], sample.get("ayah_end"))
        predicted = (result["surah"], result["ayah"], result.get("ayah_end"))
        is_correct = expected == predicted

        if is_correct:
            correct += 1
        latencies.append(elapsed)

        per_sample.append({
            "id": sample["id"],
            "expected": {"surah": sample["surah"], "ayah": sample["ayah"], "ayah_end": sample.get("ayah_end")},
            "predicted": result,
            "correct": is_correct,
            "latency": elapsed,
        })

    avg_latency = sum(latencies) / len(latencies) if latencies else 0

    return {
        "name": exp["name"],
        "accuracy": correct / total if total > 0 else 0,
        "correct": correct,
        "total": total,
        "avg_latency": avg_latency,
        "model_size": size,
        "per_sample": per_sample,
    }


def format_size(size_bytes: int) -> str:
    if size_bytes >= 1024 * 1024 * 1024:
        return f"{size_bytes / (1024**3):.1f} GB"
    return f"{size_bytes / (1024**2):.0f} MB"


def print_table(results: list[dict]):
    # Header
    print()
    print(f"{'Experiment':<30} {'Accuracy':>12} {'Latency':>10} {'Model Size':>12}")
    print("-" * 66)
    for r in results:
        acc = f"{r['accuracy']:.0%} ({r['correct']}/{r['total']})"
        lat = f"{r['avg_latency']:.2f}s"
        size = format_size(r['model_size'])
        print(f"{r['name']:<30} {acc:>12} {lat:>10} {size:>12}")
    print()


def save_results(results: list[dict]):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    path = RESULTS_DIR / f"{timestamp}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to {path}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark all experiments")
    parser.add_argument("--experiment", type=str, help="Run only this experiment")
    parser.add_argument("--category", type=str, help="Filter samples by category")
    args = parser.parse_args()

    samples = load_manifest()
    if args.category:
        samples = [s for s in samples if s["category"] == args.category]
        print(f"Filtered to {len(samples)} samples in category '{args.category}'")

    experiments = discover_experiments(args.experiment)
    if not experiments:
        print(f"No experiments found matching '{args.experiment}'")
        return

    print(f"Running {len(experiments)} experiment(s) on {len(samples)} sample(s)...")

    results = []
    for exp in experiments:
        print(f"\n>>> {exp['name']}")
        result = run_experiment(exp, samples)
        results.append(result)
        print(f"    Accuracy: {result['accuracy']:.0%} ({result['correct']}/{result['total']})")

    print_table(results)
    save_results(results)


if __name__ == "__main__":
    main()
```

**Step 3: Create benchmark/__init__.py (empty)**

**Step 4: Commit**

```bash
git add benchmark/__init__.py benchmark/runner.py benchmark/test_corpus/.gitkeep benchmark/results/.gitkeep
git commit -m "feat: add benchmark runner with CLI and JSON output"
```

---

### Task 8: Build test corpus — move baseline files + create manifest

**Files:**
- Move: `data/test_audio/*.m4a` -> `benchmark/test_corpus/`
- Copy: `data/reference_audio/{001001,001002,002255,112001,036001}.mp3` -> `benchmark/test_corpus/`
- Create: `benchmark/test_corpus/manifest.json` (baseline entries only)

**Step 1: Copy existing test files**

```bash
cp data/test_audio/nisa_51.m4a benchmark/test_corpus/
cp data/test_audio/ikhlas_2_3.m4a benchmark/test_corpus/
cp data/reference_audio/001001.mp3 benchmark/test_corpus/
cp data/reference_audio/001002.mp3 benchmark/test_corpus/
cp data/reference_audio/002255.mp3 benchmark/test_corpus/
cp data/reference_audio/112001.mp3 benchmark/test_corpus/
cp data/reference_audio/036001.mp3 benchmark/test_corpus/
```

**Step 2: Create manifest.json with baseline entries**

```json
{
  "samples": [
    {"id": "user_nisa_51", "file": "nisa_51.m4a", "surah": 4, "ayah": 51, "ayah_end": null, "category": "medium", "source": "user"},
    {"id": "user_ikhlas_2_3", "file": "ikhlas_2_3.m4a", "surah": 112, "ayah": 2, "ayah_end": 3, "category": "multi", "source": "user"},
    {"id": "ref_001001", "file": "001001.mp3", "surah": 1, "ayah": 1, "ayah_end": null, "category": "medium", "source": "everyayah"},
    {"id": "ref_001002", "file": "001002.mp3", "surah": 1, "ayah": 2, "ayah_end": null, "category": "medium", "source": "everyayah"},
    {"id": "ref_002255", "file": "002255.mp3", "surah": 2, "ayah": 255, "ayah_end": null, "category": "long", "source": "everyayah"},
    {"id": "ref_112001", "file": "112001.mp3", "surah": 112, "ayah": 1, "ayah_end": null, "category": "short", "source": "everyayah"},
    {"id": "ref_036001", "file": "036001.mp3", "surah": 36, "ayah": 1, "ayah_end": null, "category": "short", "source": "everyayah"}
  ]
}
```

**Step 3: Commit**

```bash
git add benchmark/test_corpus/manifest.json benchmark/test_corpus/*.m4a benchmark/test_corpus/*.mp3
git commit -m "feat: add baseline test corpus with 7 samples and manifest"
```

---

### Task 9: Create curate_corpus.py for RetaSy samples

**Files:**
- Create: `benchmark/curate_corpus.py`

**Step 1: Create curation script**

```python
"""
One-time script to curate ~50 test samples from RetaSy/quranic_audio.

Usage:
    python benchmark/curate_corpus.py

Downloads samples, categorizes by verse length, picks diverse speakers,
copies audio to benchmark/test_corpus/, and updates manifest.json.
"""

import json
import random
import soundfile as sf
from pathlib import Path
from datasets import load_dataset

PROJECT_ROOT = Path(__file__).parent.parent
CORPUS_DIR = Path(__file__).parent / "test_corpus"
MANIFEST_PATH = CORPUS_DIR / "manifest.json"
QURAN_PATH = PROJECT_ROOT / "data" / "quran.json"

# Target counts per category
TARGETS = {
    "short": 15,   # 1-5 words
    "medium": 15,  # 6-15 words
    "long": 10,    # 15+ words
}
# Plus ~10 longest available as "multi" stand-ins

random.seed(42)


def load_quran_verse_lengths() -> dict[tuple[int, int], int]:
    """Returns {(surah, ayah): word_count}."""
    with open(QURAN_PATH) as f:
        verses = json.load(f)
    return {(v["surah"], v["ayah"]): len(v["text_clean"].split()) for v in verses}


def categorize(word_count: int) -> str:
    if word_count <= 5:
        return "short"
    elif word_count <= 15:
        return "medium"
    else:
        return "long"


def main():
    print("Loading verse lengths...")
    verse_lengths = load_quran_verse_lengths()

    print("Loading RetaSy/quranic_audio dataset...")
    ds = load_dataset("RetaSy/quranic_audio", split="train", streaming=True)

    # Collect candidates grouped by category
    candidates = {"short": [], "medium": [], "long": []}
    seen_speakers = {"short": set(), "medium": set(), "long": set()}

    for i, example in enumerate(ds):
        surah = example.get("surah") or example.get("sura")
        ayah = example.get("ayah") or example.get("aya")
        if surah is None or ayah is None:
            continue

        key = (int(surah), int(ayah))
        if key not in verse_lengths:
            continue

        wc = verse_lengths[key]
        cat = categorize(wc)

        # Extract speaker ID if available
        speaker = example.get("speaker_id", example.get("client_id", str(i)))

        # Prefer diverse speakers
        if speaker in seen_speakers[cat] and len(candidates[cat]) >= TARGETS[cat]:
            continue

        candidates[cat].append({
            "example": example,
            "surah": key[0],
            "ayah": key[1],
            "word_count": wc,
            "speaker": speaker,
            "idx": i,
        })
        seen_speakers[cat].add(speaker)

        # Stop early if we have enough
        total = sum(len(v) for v in candidates.values())
        if total >= 80:  # collect extra for selection
            break

        if i % 500 == 0:
            print(f"  Scanned {i} examples, collected {total} candidates...")

    # Select samples
    selected = []
    for cat, target in TARGETS.items():
        pool = candidates[cat]
        random.shuffle(pool)
        selected.extend(pool[:target])

    # Add ~10 longest as "long" extras (they serve as multi stand-ins)
    remaining_long = [c for c in candidates["long"] if c not in selected]
    remaining_long.sort(key=lambda x: x["word_count"], reverse=True)
    selected.extend(remaining_long[:10])

    print(f"\nSelected {len(selected)} samples:")
    for cat in ["short", "medium", "long"]:
        count = sum(1 for s in selected if categorize(s["word_count"]) == cat)
        print(f"  {cat}: {count}")

    # Load existing manifest
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)

    # Save audio files and update manifest
    for i, sample in enumerate(selected):
        audio = sample["example"]["audio"]
        filename = f"retasy_{i:03d}.wav"
        filepath = CORPUS_DIR / filename

        # Save audio
        sf.write(str(filepath), audio["array"], audio["sampling_rate"])

        cat = categorize(sample["word_count"])
        manifest["samples"].append({
            "id": f"retasy_{i:03d}",
            "file": filename,
            "surah": sample["surah"],
            "ayah": sample["ayah"],
            "ayah_end": None,
            "category": cat,
            "source": "retasy",
        })

    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nDone. Manifest now has {len(manifest['samples'])} total samples.")
    print(f"Audio saved to {CORPUS_DIR}")


if __name__ == "__main__":
    main()
```

**Step 2: Run the curation script**

```bash
cd /Users/rock/ai/projects/offline-tarteel
python benchmark/curate_corpus.py
```

Expected: ~50 new .wav files in `benchmark/test_corpus/`, manifest updated.

**Step 3: Commit**

```bash
git add benchmark/curate_corpus.py benchmark/test_corpus/manifest.json benchmark/test_corpus/retasy_*.wav
git commit -m "feat: curate 50 RetaSy test samples for benchmark corpus"
```

---

### Task 10: Update imports across existing experiment files

**Files:**
- Modify: `experiments/new-models/benchmark.py` — change `from offline_tarteel.X` to `from shared.X`
- Modify: `experiments/embedding-search/benchmark.py` — same
- Modify: `experiments/embedding-search/search.py` — same
- Modify: `experiments/embedding-search/build_index.py` — same
- Modify: `experiments/streaming-asr/benchmark.py` — same
- Modify: `experiments/contrastive/evaluate.py` — same
- Modify: `experiments/contrastive/dataset.py` — same
- Modify: `web/server.py` — same

**Step 1: Find all files importing from offline_tarteel**

```bash
cd /Users/rock/ai/projects/offline-tarteel
grep -r "from offline_tarteel" --include="*.py" -l
```

**Step 2: Replace imports in each file**

For each file found, replace:
- `from offline_tarteel.audio import load_audio` -> `from shared.audio import load_audio`
- `from offline_tarteel.quran_db import QuranDB` -> `from shared.quran_db import QuranDB`
- `from offline_tarteel.normalizer import normalize_arabic` -> `from shared.normalizer import normalize_arabic`

Also ensure each file has `sys.path.insert(0, str(PROJECT_ROOT))` if not already present.

**Step 3: Verify no remaining old imports**

```bash
grep -r "from offline_tarteel" --include="*.py"
```

Expected: no results (or only in `src/` which we keep as legacy).

**Step 4: Commit**

```bash
git add -u  # only modified files
git commit -m "refactor: update all imports from offline_tarteel to shared"
```

---

### Task 11: End-to-end smoke test

**Step 1: Run benchmark on baseline corpus with one experiment**

```bash
cd /Users/rock/ai/projects/offline-tarteel
python -m benchmark.runner --experiment whisper-lora
```

Expected: table with accuracy, latency, model size for whisper-lora across 7 baseline samples.

**Step 2: If it works, run full benchmark**

```bash
python -m benchmark.runner
```

Expected: table with all experiments (may error on some if models not downloaded — that's fine, runner handles errors).

**Step 3: Verify results JSON was created**

```bash
ls benchmark/results/
```

Expected: one timestamped JSON file.

**Step 4: Final commit**

```bash
git add benchmark/results/.gitkeep
git commit -m "chore: verify benchmark framework works end-to-end"
```
