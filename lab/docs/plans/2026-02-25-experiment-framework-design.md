# Experiment Framework Design

## Goal

Restructure the offline-tarteel project so every approach lives under a unified `experiments/` folder with a standardized interface, enabling single-command comparison benchmarks across all approaches.

## Directory Structure

```
offline-tarteel/
├── experiments/
│   ├── whisper-lora/          # base approach (moved from src/ + scripts/)
│   ├── embedding-search/      # HuBERT + FAISS
│   ├── contrastive/           # QuranCLAP
│   ├── streaming-asr/         # chunked Whisper (batch mode for benchmark)
│   └── new-models/            # 8 ASR models behind one run.py
├── benchmark/
│   ├── runner.py              # CLI entry point
│   ├── curate_corpus.py       # one-time script to build test corpus
│   ├── test_corpus/           # audio files + manifest.json
│   └── results/               # timestamped JSON output
├── shared/
│   ├── audio.py               # load_audio()
│   ├── normalizer.py          # normalize_arabic()
│   └── quran_db.py            # QuranDB + fuzzy matching
├── data/                      # quran.json, lora adapters, FAISS index, etc.
├── web/                       # web UI (untouched)
├── docs/
├── REPORT.md
└── README.md
```

## Experiment Contract

Each experiment folder contains a `run.py` with two required functions:

```python
def predict(audio_path: str) -> dict:
    """
    Takes a path to an audio file, returns a verse match.

    Returns:
        {
            "surah": int,           # 1-114
            "ayah": int,            # start ayah
            "ayah_end": int | None, # end ayah if multi-verse, else None
            "score": float,         # 0.0-1.0 confidence
            "transcript": str,      # intermediate transcription, or "" if N/A
        }
    """

def model_size() -> int:
    """Returns total model size in bytes (files on disk needed for inference)."""
```

### Notes

- `predict()` handles its own model loading. The benchmark runner does a warmup call before timing.
- `score` meaning varies by approach (Levenshtein ratio, cosine similarity, etc.) — informational, not compared across approaches.
- `transcript` is `""` for approaches that don't produce text (embedding search, contrastive).
- For `new-models/`, `run.py` adds a `model_name` parameter: `predict(audio_path, model_name="moonshine-tiny-arabic")` and `model_size(model_name=...)`. The runner discovers available models via `list_models() -> list[str]`.

## Test Corpus

### Manifest format

`benchmark/test_corpus/manifest.json`:

```json
{
  "samples": [
    {
      "id": "retasy_001",
      "file": "retasy_001.wav",
      "surah": 2,
      "ayah": 255,
      "ayah_end": null,
      "category": "long",
      "source": "retasy"
    },
    {
      "id": "baseline_ikhlas_2_3",
      "file": "ikhlas_2_3.m4a",
      "surah": 112,
      "ayah": 2,
      "ayah_end": 3,
      "category": "multi",
      "source": "user"
    }
  ]
}
```

- `category`: `short` (1-5 words), `medium` (6-15 words), `long` (15+ words), `multi` (multi-ayah span)
- `source`: `retasy`, `everyayah`, or `user`
- Adding a new test file = drop audio in folder + add entry to manifest

### Composition (~57 samples total)

- ~15 short verses (1-5 words) from RetaSy — hardest for all approaches
- ~15 medium verses (6-15 words) from RetaSy
- ~10 long verses (15+ words) from RetaSy
- ~10 multi-ayah or longest available from RetaSy
- 7 existing baseline files (2 user recordings + 5 EveryAyah reference)

All RetaSy samples from different speakers for accent/noise diversity.

### Curation

`benchmark/curate_corpus.py` runs once to:
1. Load `RetaSy/quranic_audio` dataset from HuggingFace
2. Stratify by verse length, pick diverse speakers
3. Copy audio files into `benchmark/test_corpus/`
4. Generate `manifest.json`

The resulting files get committed. The curation script is not part of the benchmark itself.

## Benchmark Runner

### CLI

```
python -m benchmark.runner                           # all experiments, all samples
python -m benchmark.runner --experiment whisper-lora  # one experiment
python -m benchmark.runner --category short           # filter by category
```

### Output

Prints a table to stdout:

```
┌────────────────────────┬──────────┬──────────┬────────────┐
│ Experiment             │ Accuracy │ Latency  │ Model Size │
├────────────────────────┼──────────┼──────────┼────────────┤
│ whisper-lora           │ 72% (41) │ 1.23s    │ 482 MB     │
│ embedding-search       │ 45% (26) │ 0.38s    │ 378 MB     │
│ streaming-asr          │ 68% (39) │ 1.87s    │ 461 MB     │
│ new-models/moonshine   │ 65% (37) │ 1.26s    │ 103 MB     │
└────────────────────────┴──────────┴──────────┴────────────┘
```

Saves JSON to `benchmark/results/YYYY-MM-DD_HHMMSS.json` with per-sample detail.

### Metrics

- **Accuracy**: predicted `(surah, ayah, ayah_end)` exactly matches expected. Parens show correct count.
- **Latency**: average time per prediction (excluding warmup).
- **Model size**: total disk footprint reported by `model_size()`.

## Key Decisions

- Batch mode only (audio file in, verse match out) — streaming measured on final output
- Convention-based contract (functions in `run.py`), no abstract base classes
- Shared utilities in `shared/` (not a Python package, just importable modules)
- `new-models` treats its 8 models as configs within one experiment, not 8 separate folders
