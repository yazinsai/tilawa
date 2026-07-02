# Plan: FastConformer N-Best + CTC Brute-Force Fallback

## Context

The `fastconformer-ctc-rescore` experiment showed that CTC re-scoring doesn't recover any of the 8 failures because **both models receive the same (wrong) candidates** — the candidates come from a single FastConformer transcript matched against QuranDB. If the transcript is wrong, the candidate set is wrong, and CTC can't help.

This experiment attacks the problem from two angles:
1. **N-best transcripts** — beam search produces 3-5 diverse transcripts, multiplying the candidate pool. If even one hypothesis is close, QuranDB matching can find the right verse.
2. **CTC brute-force** — when ALL N-best matches have low confidence, bypass transcript-based retrieval entirely. CTC-score all verses in the top candidate surahs directly against the audio. This removes the transcript quality bottleneck.

## The 8 failures to target

| Sample | Expected | Current result | Failure mode |
|---|---|---|---|
| ref_112001 | 112:1 | 89:26 (score 0.61) | Bad transcript (Al-Ikhlas) |
| ref_036001 | 36:1 | 42:2 (score 0.40) | Bad transcript (Ya-Sin isolated letters) |
| retasy_019 | 3:2 | 92:6 (score 0.53) | Bad transcript |
| multi_036_001_005 | 36:1-5 | 36:5 (score 0.76) | Span matching failure + MAX_SPAN=4 |
| multi_113_001_005 | 113:1-5 | 31:12 (score 0.49) | Bad transcript + multi-verse |
| multi_114_001_006 | 114:1-6 | 10:104 (score 0.53) | Bad transcript + multi-verse |
| multi_055_001_004 | 55:1-4 | 96:1-2 (score 0.64) | Bad transcript |
| multi_103_001_003 | 103:1-3 | 103:2-3 (score 0.91) | Span matching off-by-one |

## Design

### Pipeline

```
Audio
  │
  ├──► FastConformer beam search (beam_size=5)
  │      → 5 transcript hypotheses with scores
  │
  ├──► For each hypothesis:
  │      QuranDB.match_verse(text, max_span=6)
  │      → best verse match with score
  │
  ├──► If any match score >= CONFIDENCE_THRESHOLD:
  │      RETURN best match (fast path, ~0.4s)
  │
  ├──► Collect top-K surahs across ALL N-best matches
  │      (union of candidate surahs from all hypotheses)
  │
  ├──► CTC brute-force: score ALL verses in those surahs
  │      - Single verses + multi-verse spans (up to MAX_SPAN=6)
  │      - ~500-2000 candidates depending on surah sizes
  │
  └──► Return best CTC-scored result
```

### Why N-best helps

CTC beam search explores multiple paths through the token lattice. For ambiguous audio (short verses, isolated letters), different beams may produce different transcripts. Example: Ya-Sin "يس" might appear as "ياسين" in beam 1 and "يس" in beam 3. Only one needs to be close enough for QuranDB to find surah 36.

**Caveat:** CTC N-best diversity is limited without a language model — beams often collapse to similar texts. If diversity is poor, the N-best step may not help much. The CTC brute-force fallback covers this case.

### Why CTC brute-force helps

The current pipeline: transcript → QuranDB search (top-50) → CTC re-score 50 candidates.

The failure mode: when the transcript is "ان ربك هو القصب" instead of "قل هو الله احد", QuranDB search returns 50 candidates that don't include surah 112 at all. CTC re-scoring 50 wrong candidates can't produce the right answer.

CTC brute-force: skip QuranDB search entirely. Take the top 10 surahs from the N-best QuranDB matches (which gives us a wider net), then CTC-score ALL verses in those surahs. Even if no single transcript is correct, as long as the correct surah appears in ANY of the 5 × match results, the CTC model can find the right verse by direct acoustic matching.

For the 8 failures, the question is whether the correct surah appears in the top-10 across 5 hypotheses. If yes, CTC brute-force can find it.

### Surah-level candidate math

- Average surah: ~40 verses, max: 286 (Al-Baqarah)
- Top 10 surahs × ~40 verses = ~400 single candidates
- With spans (up to 6): ~400 + ~1200 spans = ~1600 total candidates
- CTC scoring 1600 candidates takes ~2-4s (batched ctc_loss)
- Only runs on low-confidence samples (~15% of corpus)

### Configuration

```python
N_BEST = 5                    # Beam search hypotheses
CONFIDENCE_THRESHOLD = 0.7    # Skip CTC if any N-best match exceeds this
TOP_SURAHS_BRUTEFORCE = 10    # Surahs for CTC brute-force
MAX_SPAN = 6                  # Support up to 6-verse spans
```

## Tasks

### Task 1: Create `experiments/fastconformer-nbest-bruteforce/run.py`

**Key functions:**

```python
def _fastconformer_transcribe_nbest(audio_path: str) -> list[dict]:
    """Return N-best transcriptions: [{"text": str, "score": float}, ...]"""
    # Uses beam search with return_best_hypothesis=False

def _collect_candidate_surahs(nbest_matches: list[dict]) -> list[int]:
    """Union of top surahs from all N-best QuranDB matches."""

def _bruteforce_ctc_score(audio_path: str, surahs: list[int]) -> dict:
    """CTC-score ALL verses + spans in the given surahs."""

def predict(audio_path: str) -> dict:
    """
    1. N-best transcribe
    2. QuranDB match each → if confident, return
    3. CTC brute-force top surahs → return best
    """
```

**FastConformer N-best setup:**

```python
from omegaconf import open_dict

decoding_cfg = _model.cfg.decoding
with open_dict(decoding_cfg):
    decoding_cfg.strategy = "beam"
    decoding_cfg.beam.beam_size = N_BEST
    decoding_cfg.beam.return_best_hypothesis = False
    decoding_cfg.beam.search_type = "default"
_model.change_decoding_strategy(decoding_cfg, decoder_type="ctc")
```

**CTC brute-force candidate building:**

```python
def _build_bruteforce_candidates(surahs: list[int]) -> list[dict]:
    candidates = []
    for surah_num in surahs:
        verses = _db.get_surah(surah_num)
        # Single verses
        for v in verses:
            candidates.append({
                "surah": surah_num,
                "ayah": v["ayah"],
                "ayah_end": None,
                "text_clean": v["text_clean"],
                "text_uthmani": v["text_uthmani"],
                "text": v["text_uthmani"],
            })
        # Multi-verse spans
        for start_idx in range(len(verses)):
            for span_len in range(2, MAX_SPAN + 1):
                end_idx = start_idx + span_len
                if end_idx > len(verses):
                    break
                chunk = verses[start_idx:end_idx]
                candidates.append({
                    "surah": surah_num,
                    "ayah": chunk[0]["ayah"],
                    "ayah_end": chunk[-1]["ayah"],
                    "text_clean": " ".join(v["text_clean"] for v in chunk),
                    "text_uthmani": " ".join(v["text_uthmani"] for v in chunk),
                    "text": " ".join(v["text_uthmani"] for v in chunk),
                })
    return candidates
```

**Reuse from existing code:**
- FastConformer loading + kaldialign fallback from `experiments/nvidia-fastconformer/run.py`
- CTC model loading from `experiments/fastconformer-ctc-rescore/run.py`
- `score_candidates()` from `experiments/ctc-alignment/ctc_scorer.py`
- `QuranDB` from `shared/quran_db.py`

**Stage 2 CTC model fallback chain** (same as fastconformer-ctc-rescore):

```python
STAGE2_CANDIDATES = [
    PROJECT_ROOT / "data" / "rabah-ctc-pruned-8l-first_n-finetuned",
    PROJECT_ROOT / "data" / "rabah-ctc-pruned-8l-evenly_spaced-finetuned",
    PROJECT_ROOT / "data" / "rabah-ctc-pruned-12l-evenly_spaced-finetuned",
    PROJECT_ROOT / "data" / "rabah-ctc-pruned-8",
    PROJECT_ROOT / "data" / "ctc-base-distilled",
    PROJECT_ROOT / "data" / "ctc-base-finetuned",
    PROJECT_ROOT / "data" / "ctc-model",
]
```

### Task 2: Register in benchmark runner

Add to `EXPERIMENT_REGISTRY` in `benchmark/runner.py`:
```python
"fastconformer-nbest-bruteforce": EXPERIMENTS_DIR / "fastconformer-nbest-bruteforce" / "run.py",
```

### Task 3: Benchmark and analyze

```bash
.venv/bin/python -m benchmark.runner --experiment fastconformer-nbest-bruteforce
```

Check:
1. Do any of the 8 failures get recovered?
2. What is the N-best diversity like? (print the 5 transcripts for failed samples)
3. Does the correct surah appear in the brute-force candidate set?
4. What's the latency impact? (fast path vs brute-force path)

### Task 4: Update README with results

## Files to create/modify

| Action | File |
|--------|------|
| Create | `experiments/fastconformer-nbest-bruteforce/run.py` |
| Modify | `benchmark/runner.py` (add registry entry) |
| Modify | `README.md` (add results after benchmarking) |

## Risks

1. **Low N-best diversity from CTC beam search.** Without a language model, CTC beams often collapse to near-identical texts. Mitigation: the CTC brute-force fallback doesn't depend on transcript diversity at all.

2. **Brute-force candidate count could be large.** If top surahs include Al-Baqarah (286 verses), candidates could be 3000+. Mitigation: chunk the CTC scoring into batches of 50, or cap per-surah verse count.

3. **Beam search is slower.** beam_size=5 is ~3-5x slower than greedy for FastConformer. Mitigation: only matters for the overall experiment latency, not the fast path (since the fast path confidence gate still applies).

4. **NeMo beam search API may differ between versions.** The `change_decoding_strategy` API changed across NeMo versions. The installed version may need different kwargs. Mitigation: wrap in try/except and fall back to greedy + CTC brute-force only.
