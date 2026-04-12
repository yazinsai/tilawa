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

## Experiment workflow (end-to-end)

Every experiment — a new model, a tracker/matcher change, a training run, a data-mix tweak — follows this protocol. Do not skip steps. The definition-of-done checklist at the bottom is the enforcement contract.

### 1. Branch into a worktree

```bash
git worktree add .worktrees/<name> -b <name>
cd .worktrees/<name>
```

New experiments MUST be developed in a worktree (see "Git Worktrees" below). Keeps main clean and lets multiple experiments run in parallel without conflicts.

### 2. Implement

- New ASR/matching experiment → `experiments/<name>/run.py` per the "Experiment Convention" section. Register in `benchmark/runner.py`.
- Inference / tracker / matcher change → edit `web/frontend/src/lib/` (TS) or `shared/` (Python). Add a vitest (`test/*.test.ts`) that deterministically exercises the change without ONNX.
- Training run → script in `scripts/*_modal.py` launched detached (see "Training" below). One data-side change per run — never combine.

### 3. Measure with discipline

**ONNX streaming has ±3–6 sample variance per run on v1.** Run 3 times (max) and report the median. A single-run improvement inside the variance envelope is not an improvement.

- **Browser/RN streaming (shipped pipeline):**
  ```bash
  cd web/frontend
  npx tsx test/stability-report.ts --repeats=3 --json=test/<name>-stability.json
  npx tsx test/stability-report.ts --repeats=3 --corpus=test_corpus_v2 --json=test/<name>-v2-stability.json
  ```
  Produces per-sample pass-rate classification + aggregate medians. Compare against baseline JSON from the prior commit.
- **Python batch / Python streaming:**
  ```bash
  .venv/bin/python -m benchmark.runner --experiment <name>
  ```
  Results land in `benchmark/results/<timestamp>.json`.
- **Unit tests:** `npx vitest run` (TS) or `pytest` (Python) must stay green.

If the shipped-model streaming numbers changed, also run a **v2 blind check** — same pipeline, `--corpus=test_corpus_v2`. Same-direction movement on v2 is the sanity gate. If v2 regresses while v1 improves, treat it as overfit to v1 and stop.

### 4. Document in EXPERIMENTS.md

EXPERIMENTS.md is the single source of truth for benchmark numbers and writeups. README only carries the shipped-model headline table.

Decide which of these patterns applies (not mutually exclusive):

**Inference/tracker/matcher change to the shipped pipeline:**
- Add an entry to `## Streaming changelog` under `## Shipped model`. Use this exact format:
  ```md
  **YYYY-MM-DD — short title** (commit `<hash>`)
  One paragraph: what changed, why, the invariant or structural property it adds.
  Numbers: precision <before>% → <after>% (<±Δ>pp), SeqAcc <before>% → <after>% (<±Δ>pp), recall <before>% → <after>% (<±Δ>pp) on v1. Same pattern on v2 blind check.
  <Optional> Measurement command, edge cases worth remembering.
  ```
- Update the shipped-model headline table if its cells changed.
- Update the one-line summary in `README.md` (Goal section) if the headline metric changed.

**New experiment (new `experiments/<name>/`):**
- Add a row to `## All experiments — streaming` and/or `## All experiments — batch`, sorted by v1 recall. Use ✓/— in the FT column, MB for size, s for latency.
- Add a `**<name>**` line to `## Per-experiment notes` — 1–3 sentences on what it does and the key finding.
- If it fits an existing deep dive (Rabah pruning, TLOG mix), add a row there instead of a new section.
- 0% recall or missing deps → `### 0% recall — broken or inapplicable` with a one-sentence reason.

**Finding that generalizes across experiments** → add to `## Key findings` as a numbered bullet.

### 5. Save raw artifacts

- `benchmark/results/<timestamp>.json` (runner writes automatically) — do not delete.
- `web/frontend/test/<name>-stability.json` for streaming stability runs.
- Training checkpoints stay on Modal volumes (see "Training"); document the volume name + checkpoint step in the changelog entry so it's reproducible.

### 6. Commit + merge

- Commit subject: `<area>: <what changed>` (≤72 chars). Area is `tracker`, `matcher`, `train`, `docs`, `experiment/<name>`, etc.
- Commit body: the why, the before→after deltas, the measurement methodology (repeats, variance).
- Never skip hooks or bypass signing.
- Merge with `git merge <branch> --no-ff -m "Merge branch '<name>': ..."` so the experiment is a discoverable merge commit.
- Remove the worktree: `git worktree remove .worktrees/<name>`.

### Definition of done

Merge is blocked until every box is checked:

- [ ] Developed in a worktree under `.worktrees/<name>/`
- [ ] `run.py` exports the required interface (if a new experiment) and is in `EXPERIMENT_REGISTRY`
- [ ] Measured over 3 runs; median reported, not cherry-picked best
- [ ] v2 blind check run if the shipped pipeline changed; v2 did not regress
- [ ] Unit tests pass (`npx vitest run` or `pytest`) with deterministic coverage of the change
- [ ] EXPERIMENTS.md updated: table row / changelog entry / per-experiment note / key finding, as applicable
- [ ] README.md updated only if the shipped-model headline table or Goal-line metric changed — otherwise README stays untouched
- [ ] Raw result JSON committed to `benchmark/results/` or `web/frontend/test/`
- [ ] Commit body cites before→after deltas with variance context (e.g. "v1 median across 5 runs")
- [ ] Merged to `main` with `--no-ff`; worktree removed

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

### Pre-Launch Discipline

- **Verify arg defaults.** Read the entrypoint signature before launching — don't assume what `0` or `""` means.
- **Check external deps first.** Verify models/APIs still exist before writing code that depends on them.
- **Estimate throughput upfront.** Do the math (samples × time/sample ÷ workers) before committing to a design.
- **Diagnose before iterating.** When something fails, understand WHY before trying a different config. Research first, not after 5 failed attempts.
- **Design for failure.** Workers get preempted. Write results to volume incrementally, not in a final merge step. Clean up stale apps after each failed attempt (`modal app list`).

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
