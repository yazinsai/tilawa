# Offline Tarteel

Offline Quran verse recognition — given an audio clip of someone reciting, identify the surah and ayah.

## Project Structure

```
shared/              # Common utilities (audio, normalizer, quran_db)
experiments/         # Each approach gets its own directory with a run.py
benchmark/           # Runner, test corpus, results
  runner.py          # CLI: python -m benchmark.runner
  test_corpus/       # Audio files + manifest.json (54 samples)
  results/           # Timestamped JSON output
data/                # quran.json, reference audio, LoRA adapters
src/offline_tarteel/ # Legacy package (kept for compatibility)
scripts/             # One-off training/eval scripts
web/                 # FastAPI + React frontend for live demo
```

## Experiment Convention

Every experiment lives in `experiments/<name>/` with a `run.py` that exports:

### Required functions

```python
def predict(audio_path: str) -> dict:
    """Run inference on an audio file.

    Returns:
        {"surah": int, "ayah": int, "ayah_end": int|None, "score": float, "transcript": str}

    On failure/no match: surah=0, ayah=0, score=0.0
    transcript="" is fine for non-ASR approaches (embedding search, contrastive).
    """

def model_size() -> int:
    """Total model size in bytes (estimate is fine)."""
```

### Pattern

```python
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.audio import load_audio
from shared.quran_db import QuranDB

# Lazy-load globals
_model = None

def _ensure_loaded():
    global _model
    if _model is not None:
        return
    # ... load model, processor, etc.

def predict(audio_path: str) -> dict:
    _ensure_loaded()
    # ... inference logic
    # For ASR approaches: transcribe -> match against QuranDB
    # For embedding approaches: encode -> find nearest verse

def model_size() -> int:
    return 461 * 1024 * 1024  # estimate in bytes
```

### Multi-model experiments

If one experiment wraps multiple models (like `new-models/`), also export:

```python
def list_models() -> list[str]: ...
def predict(audio_path: str, model_name: str = "default") -> dict: ...
def model_size(model_name: str = "default") -> int: ...
```

The benchmark runner expands these into separate entries automatically.

### Registering a new experiment

Add it to `EXPERIMENT_REGISTRY` in `benchmark/runner.py`:

```python
EXPERIMENT_REGISTRY = {
    "whisper-lora": EXPERIMENTS_DIR / "whisper-lora" / "run.py",
    "your-new-experiment": EXPERIMENTS_DIR / "your-new-experiment" / "run.py",
    # ...
}
```

Directory names use hyphens (e.g. `whisper-lora`). The runner uses `importlib.util.spec_from_file_location` so hyphens work fine.

## Shared Utilities

Use `from shared.X import ...` (not `from offline_tarteel.X`):

- `shared.audio.load_audio(path, sr=16000)` — returns float32 numpy array at 16kHz
- `shared.normalizer.normalize_arabic(text)` — strips diacritics, normalizes alef/taa marbuta
- `shared.quran_db.QuranDB` — loads `data/quran.json` (6,236 verses), provides:
  - `match_verse(text)` — fuzzy match with multi-ayah span support
  - `search(text, top_k=5)` — top-k Levenshtein matches
  - `get_verse(surah, ayah)`, `get_surah(surah)`, `get_next_verse(surah, ayah)`
- `shared.verse_tracker.VerseTracker` — streaming verse detection with continuation bias
- `shared.streaming.StreamingPipeline` — connects ASR backends to verse tracker

## Running Benchmarks

```bash
.venv/bin/python -m benchmark.runner                          # all experiments
.venv/bin/python -m benchmark.runner --experiment whisper-lora # single experiment
.venv/bin/python -m benchmark.runner --category short          # filter by category
```

Results go to `benchmark/results/<timestamp>.json`.

### Scoring

The benchmark uses **sequence evaluation**. Experiments with `predict()` are called directly; others use `transcribe()` + `VerseTracker`. Metrics:

- **Recall**: fraction of expected verses detected in the correct order
- **Precision**: fraction of predicted verses that are correct
- **Sequence Accuracy**: 1.0 only if the full ordered sequence matches exactly

### Experiment Interface

Each experiment must export:

```python
def transcribe(audio_path: str) -> str:  # raw transcript (required for benchmark)
def predict(audio_path: str) -> dict:     # full prediction with verse match
def model_size() -> int:                  # model size in bytes
```

## Test Corpus

`benchmark/test_corpus/manifest.json` — 54 samples:
- 2 user recordings (.m4a)
- 23 EveryAyah reference (Alafasy, includes 8 long single-ayah + 9 multi-ayah concatenated)
- 29 RetaSy crowdsourced (curated via `benchmark/curate_corpus.py`)
- Categories: short (17), medium (19), long (9), multi (9)

## Current Experiments

| Experiment | Approach | SeqAcc | Recall | Precision | Latency | Size |
|---|---|---|---|---|---|---|
| ctc-alignment | wav2vec2 CTC forced alignment (fine-tuned) | **81%** | 83% | 83% | ~5s | 1.2 GB |
| tarteel-whisper-base | tarteel-ai/whisper-base-ar-quran | 67% | 72% | 75% | ~3s | 290 MB |
| whisper-lora | Whisper-small + LoRA | 58% | 64% | 65% | ~1.3s | 485 MB |
| embedding-search | HuBERT → FAISS index | — | — | — | — | — |
| contrastive | QuranCLAP (HuBERT+AraBERT) | — | — | — | — | — |
| streaming-asr | mlx-whisper chunked | — | — | — | — | — |
| new-models/* | Various HF models | — | — | — | — | — |

**Dead ends:** Moonshine Tiny AR LoRA (character-level tokenizer makes it un-fine-tunable, see README "What we tried").

## Training

Use **Modal** (modal.com) for GPU training jobs. Training scripts live in `scripts/` and are named `*_modal.py`. Always spawn Modal jobs detached (`modal run --detach`) so they keep running if the terminal closes or internet disconnects.

```bash
modal run --detach scripts/train_xxx_modal.py
```

### Modal Training Lessons

- **Set `cpu=8` on GPU functions.** Modal defaults to minimal CPUs. Without enough cores, `num_workers` data loaders can't keep the GPU fed and utilization drops to ~20%.
- **Size batch to fill GPU memory.** A100-80GB can handle `batch_size=32+` for FastConformer. `batch_size=16` only uses ~20GB VRAM — wasteful.
- **`modal.Mount` was removed in Modal 1.0+.** Use `image.add_local_file()` / `image.add_local_dir()` instead.
- **Never `add_local_dir` on large directories.** The `data/` dir is 23GB of models/checkpoints. Mount only the specific files needed (e.g. two 1.4MB JSON files).
- **Use `--detach` for long runs.** Local connections drop on jobs longer than a few minutes.

## Git Worktrees

Use `./.worktrees/` as the worktree directory for all git worktrees in this project.

**New experiments MUST be developed in a worktree.** Create a worktree, build and validate the experiment there, then merge back to main once it's proven out. This keeps the main branch clean and lets multiple experiments run in parallel without conflicts.

```bash
git worktree add .worktrees/experiment-name -b experiment-name
# ... develop and test in .worktrees/experiment-name/ ...
git merge experiment-name   # merge back once done
git worktree remove .worktrees/experiment-name
```

## Python Environment

Always use `.venv/bin/python` — the system Python lacks project dependencies.
