# Spec: Improve Streaming Accuracy from 41% to 70%+

## Problem

The Quran recitation recognition app uses a streaming pipeline: audio arrives in 300ms chunks, gets transcribed via ONNX model, and matched against 6236 Quran verses. The streaming pipeline scores **22/53 (41.5%)** on the test corpus, while non-streaming (full-file transcription) scores **37/53 (69.8%)**. The gap is caused by the tracker's windowing, thresholds, and state transitions — not the model or mel spectrogram.

## Architecture Overview

```
Audio chunks (300ms) → RecitationTracker.feed()
  ├── Discovery mode: accumulate 2s → transcribe → matchVerse()
  │   └── Match found → emit verse_match → enter tracking mode
  └── Tracking mode: accumulate 0.5s → transcribe → alignPosition()
      ├── Progress → emit word_progress
      ├── Verse complete → emit next verse_match → track next verse
      └── Stale (4 cycles) → exit tracking → back to discovery
```

Key files (all in `web/frontend/`):
- `src/lib/tracker.ts` — `RecitationTracker` class (discovery + tracking logic)
- `src/lib/quran-db.ts` — `QuranDB.matchVerse()` (verse matching with Levenshtein)
- `src/lib/types.ts` — all tunable constants
- `src/worker/mel.ts` — mel spectrogram (don't modify)
- `src/worker/ctc-decode.ts` — CTC decoder (don't modify)
- `test/validate-streaming.ts` — test harness
- `test/diagnose-sample.ts` — diagnostic tool for empty-result failures
- `test/diagnose-longverse.ts` — diagnostic tool for long-verse failures (ratio vs partialRatio vs bestSpanRatio)
- `test/diagnose-multiverse.ts` — diagnostic tool for multi-verse cascading

## Phase 3 Results (completed)

**After Phase 2:** 30/53 (56.6%) streaming
**After Phase 3:** 29-33/53 (54.7-62.3%) streaming, 37/53 non-streaming (no regression)
**Note:** ONNX runtime non-determinism causes ±2-3 sample variance per run.

### Cross-validation (overfitting check)

| | v1 corpus (53 samples) | v2 corpus (43 samples) | Average |
|---|---|---|---|
| **Phase 2** (be8ab60) | 30/53 (56.6%) | 21/43 (48.8%) | 52.7% |
| **Phase 3** (336d713) | 29/53 (54.7%) | 25/43 (58.1%) | 56.4% |
| **Delta** | -1 (-1.9%) | +4 (+9.3%) | +3.7% |

Phase 3 generalizes better — gained +4 on unseen v2 corpus while Phase 2 dropped 8 points from v1→v2. No overfitting detected.

### Phase 3 changes made

1. **Extended partial scoring to 15+ word verses** (quran-db.ts) — lowered gate from 30→15 words (40→20 without hint). Fixes medium-length verses where ratio() length-mismatch penalty was blocking correct matches.

2. **Anti-cascade hysteresis** (tracker.ts) — after verse emit, for 2 discovery cycles require higher threshold (0.65) for non-continuation jumps. Reduces false positive cascading into wrong verses.

3. **No-space char-level comparison** (tracker.ts) — `_charLevelProgress` strips spaces before sliding-window comparison, improving match quality for spaceless model output.

### What was tried and reverted

| Approach | Result | Why it failed |
|---|---|---|
| Remove 10-word char-level gate (transcript-shape only) | +7 gains, -3 regressions (34/53 but unstable) | Cascading through short wrong verses; char-level completes 3-5 word verses instantly |
| Lower partial scoring to 8+ words | -7 regressions (26/53) | Wrong short verses get boosted partial scores that beat correct long verses |
| Top-100 rescoreLimit for partial scoring | Catastrophic (-7 long verse regressions) | Long verses score 0.25-0.33 on ratio(); not in top 100, never get partial scoring |
| OR logic gate (10+ always OR 6+ when spaceless) | Unstable, ±3 variance | ONNX runtime non-determinism makes measurement unreliable |

### Remaining gap analysis

9 streaming-specific failures remain (pass non-streaming, fail streaming). 7 are fundamentally blocked by `FIRST_MATCH_THRESHOLD=0.75` where the correct verse isn't even top-3 in any streaming cycle. Further progress requires a different approach (model upgrade or non-Levenshtein scoring).

## Phase 2 Results (completed)

**After Phase 1:** 25/53 (47.2%) streaming
**After Phase 2:** 31-32/53 (58-60%) streaming, 37/53 non-streaming (no regression)
**Net gain:** +6-7 samples over Phase 1, zero non-streaming regressions

### Phase 2 changes made

1. **`fastPartialRatio` for long-verse discovery** (quran-db.ts) — Sliding window partial matching with coarse-then-refine for performance. Re-scores 30+ word verses when continuation hint exists, using spaceless text comparison (model outputs have no word boundaries). Discount factor 0.85 prevents false positives.

2. **Character-level tracking progress** (tracker.ts) — `_charLevelProgress()` method slides the transcript across the verse's joined phonemes to estimate word position. Only enabled for verses with 10+ words (short verses use word-level alignment). Threshold 0.55 prevents spurious progress in wrong verses. This fixed tracking for multi-verse cascading (67:1→67:2→...→67:13).

3. **Tuned interactions** — The partial scoring discount (0.85), char-level threshold (0.55), and 10+ word gate work together: partial scoring discovers correct long verses without boosting wrong short ones, and char-level tracking enables proper verse-by-verse progression.

## Phase 1 Results (completed)

**Baseline:** 21/53 (39.6%) streaming, 37/53 non-streaming
**After Phase 1:** 25/53 (47.2%) streaming, 37/53 non-streaming (no regression)
**Net gain:** +4 samples, zero regressions

### Changes made (tracker.ts only)

1. **Cumulative completion fix** — tracking used per-cycle `matchedIndices.length / totalWords` which underreported coverage from rolling 5s windows. Changed to cumulative `(trackingLastWordIdx + 1) / totalWords`. Unlocked multi-verse tracking completion.

2. **`hasEverMatched` flag** — decoupled cold-start threshold from `lastEmittedRef` nullity. Prevents `FIRST_MATCH_THRESHOLD` (0.75) from re-applying after stale rollback resets `lastEmittedRef` to null. Set eagerly on first discovery match and reinforced in `_exitTracking` on successful tracking.

### What was tried and reverted

| Approach | Result | Why it failed |
|---|---|---|
| Lower FIRST_MATCH_THRESHOLD (0.55-0.65) | Net zero: +6 gains, -6 regressions | Wrong verses also score 0.55-0.65; false positives cascade |
| Prefix scoring in matchVerse | Wrong direction | Rolling window keeps LAST N seconds, not first N; transcript is from verse middle/end |
| Progressive window growth (10s→30s) | More cascading wrong matches | Longer window ≠ better transcript; just more time to match wrong verses |
| First-match confirmation (2 cycles) | No effect on empty results | Noisy transcripts produce different top matches each cycle |
| Score margin check (top1 - top2 ≥ 0.05) | No effect | Margins too small in 2s partial audio |
| Residual TTL (2 cycles) | -2 regression | Removing the check allows MORE wrong cascading |
| Fast rediscovery (0.5s trigger after stale) | -2 regression | Less audio = noisier transcripts = worse matches |

## Diagnostic Findings (critical — read before implementing)

### Finding 1: `ratio()` fundamentally cannot rank correct verses for partial audio

The Levenshtein `ratio()` formula `2 * matches / (len1 + len2)` inherently penalizes length mismatches. A 20-character streaming transcript compared against a 34-character verse (correct) scores ~0.60, while a wrong 22-character verse scores ~0.71 simply because it's closer in length.

| Sample | Correct verse ratio (streaming) | Best wrong verse ratio | Correct verse ratio (full-file) |
|---|---|---|---|
| retasy_003 (1:2) | 0.600 | 0.705 (26:101) | 0.852 |
| retasy_017 (1:7) | 0.588 | 0.710 | 0.788 |
| ref_003191 (3:191) | 0.548 | 0.685 | 0.829 |

**The correct verse never appears in the top 3 candidates during any streaming cycle.** The problem is NOT the threshold — it's that `ratio()` ranks wrong verses higher than the correct one for partial transcripts.

### Finding 2: `partialRatio` and `bestSpanRatio` fix long-verse scoring completely

For long verses (20+ seconds), `ratio()` is catastrophically bad because the 10s rolling window produces a transcript covering ~15% of the verse text. But partial/span scoring works:

| Sample | Best `ratio()` | Best `partialRatio()` | Best `bestSpanRatio()` |
|---|---|---|---|
| ref_002255 (52s, Ayat al-Kursi) | 0.328 | 0.889 | 0.920 |
| ref_048029 (77s) | 0.254 | 0.863 | 0.914 |

Both `partialRatio` and `bestSpanRatio` score the correct verse at 0.86-0.92 — well above any threshold. The fix is clear: use span-based scoring in `matchVerse()`.

`bestSpanRatio` implementation (slide a window of ~transcript length across the verse):
```ts
function bestSpanRatio(text: string, verseText: string): number {
  const textWords = text.split(" ");
  const verseWords = verseText.split(" ");
  if (textWords.length < 2) return 0;
  let best = 0;
  const spanLen = Math.min(textWords.length + 3, verseWords.length);
  for (let i = 0; i <= verseWords.length - spanLen; i++) {
    const span = verseWords.slice(i, i + spanLen).join(" ");
    best = Math.max(best, ratio(text, span));
  }
  return best;
}
```

### Finding 3: Tracking mode is broken — model outputs phonemes without word boundaries

**This is the biggest discovery.** The ONNX phoneme model outputs continuous phoneme strings like `"tabyadhilmulkuwahwaEalaakulli$ay<"` without spaces between words. The tracker's `alignPosition()` does `text.split(" ")` to get recognized words, producing 1-2 giant strings. These never match individual entries in `phoneme_words` like `["bismi", "allahi", ...]`.

Evidence from multi_067_001_004:
- Tracker correctly discovers 67:1 via discovery mode (full-text Levenshtein works)
- Enters tracking mode for 67:1
- `alignPosition` always returns `matchedIndices = []` because the spaceless transcript can't match individual words
- After 4 stale cycles (0% progress), tracking exits
- Stale rollback (progress < 0.5) resets `lastEmittedRef`, losing the continuation hint
- Discovery cascades into wrong verses without the continuation bonus

**Tracking mode never actually tracks word progress for most verses.** The cumulative completion fix helped only because some transcripts coincidentally had partial space separation.

## Root Causes (revised, ranked by impact)

### 1. CRITICAL: `ratio()` cannot match partial transcripts against full verses

`matchVerse()` uses `ratio(text, verse.phonemes_joined)` which penalizes length mismatches. For streaming's short windows, this means:
- Correct verse scores 0.25-0.60 (transcript much shorter than verse)
- Wrong shorter verses score 0.65-0.89 (closer in length to transcript)
- The correct verse is not even in the top 3 candidates

**Fix:** Use `bestSpanRatio()` (or `partialRatio()`) as supplementary scorer: `max(ratio, discounted_span_ratio)`. Apply to top N candidates from initial `ratio()` pass for performance.

**Affects:** ALL failure types — empty results, long verses, and cascading wrong matches all stem from bad ranking.

### 2. CRITICAL: Tracking mode's word alignment is incompatible with model output

The model outputs continuous phoneme strings without word boundaries. `alignPosition()` splits on spaces and tries to match individual words, which always fails. Tracking always stale-exits at 0% progress.

**Fix:** Either:
- Use character-level alignment instead of word-level in tracking
- Use `partialRatio()` or sliding-window alignment to measure tracking progress
- Pre-process the transcript to insert word boundaries by aligning against the verse's known phoneme words

**Affects:** All multi-verse tracking, verse completion detection, word_progress emissions.

### 3. HIGH: Post-verse stale rollback loses continuation context

When tracking stales at 0% (which is ALWAYS due to #2), `lastEmittedRef` is rolled back, losing the continuation bonus for the next verse. This removes the +0.22 bonus that would correctly bias discovery toward the next verse in sequence.

**Fix:** Fixing #2 would fix this automatically. Alternatively, don't roll back `lastEmittedRef` when tracking duration was very short (stale exit within 2-3 cycles suggests the verse identification was wrong, but if tracking lasted longer, the verse was likely correct even if word alignment failed).

### 4. MEDIUM: `FIRST_MATCH_THRESHOLD = 0.75` is too high but can't be lowered safely

The threshold blocks correct matches (which score 0.55-0.70) AND wrong matches (which also score 0.55-0.70). Lowering it alone causes net-zero: equal gains and regressions. Only fixing the RANKING (#1) would make lowering the threshold safe.

### 5. LOW: Residual check is both helpful and harmful

The `partialRatio > 0.7` residual check prevents cascading wrong matches (good) but also blocks correct subsequent verses that share phonemes (bad). With better scoring (#1), this check becomes less necessary because correct verses would rank higher and pass above the wrong ones.

## Constraints

1. **Do NOT modify** `mel.ts`, `ctc-decode.ts`, `levenshtein.ts`, `phoneme-aligner.ts`, or `correction.ts` — these are correct
2. **Do NOT change** `validate-streaming.ts` test logic (chunk size, silence tail, pass/fail criteria) — you need a stable test to measure improvements
3. **You CAN modify**: `tracker.ts`, `quran-db.ts`, `types.ts`
4. **You CAN add** new utility files if needed (e.g., a partial matching function)
5. Changes must not break the non-streaming path or the browser-side usage (the tracker is also used in the web worker via `inference.ts`)
6. The `WorkerOutbound` message types in `types.ts` are the public API — don't remove or rename existing message types

## Next Steps: Phase 2 Implementation Plan

### Phase 2A: Replace `ratio()` with span-based scoring in `matchVerse()` (highest impact)

In `quran-db.ts`, change the scoring in `matchVerse()`:

1. First pass: `ratio()` on all 6236 verses (fast, existing)
2. Second pass: `bestSpanRatio()` on top 50 candidates from first pass
3. Use `max(ratio, bestSpanRatio * 0.90)` as the effective raw score
4. The 0.90 discount prevents false positives from coincidental substring matches

This should fix:
- Long-verse failures (ratio 0.25 → span 0.92)
- Some empty-result failures (correct verse moves from outside top 3 into top 1)
- Cascading wrong matches (correct verse ranks higher, continuation bonus compounds)

**Test first with:** `ref_002255`, `ref_048029`, `ref_003191`, `retasy_003`

### Phase 2B: Fix tracking word alignment (second highest impact)

The model's spaceless phoneme output breaks `alignPosition()`. Options:

**Option 1 (simplest):** Use character-level `partialRatio` for tracking progress instead of word-level alignment. Compare the full transcript against the tracking verse's `phonemes_joined` and measure how far into the verse the best-matching window extends.

**Option 2 (more accurate):** Pre-segment the transcript by aligning it against the verse's known word boundaries. For each phoneme_word in the verse, find the best matching substring in the transcript using dynamic programming.

**Option 3 (Codex suggestion):** Accumulate evidence across cycles — track whether alignment progress is monotonic rather than requiring exact word matches.

Start with Option 1 as it's simplest and directly addresses the diagnostic finding.

### Phase 2C: Anti-cascade hysteresis (after 2A and 2B)

After a verse emit, for 1-2 cycles:
- Continuation candidates (next verse in sequence): normal threshold (0.45)
- Jumps to unrelated verses: require higher threshold (0.65+)

This preserves the safety valve while allowing legitimate transitions. Only implement after 2A/2B since better scoring may make this unnecessary.

### Phase 2D: Evidence accumulation for first match (after 2A)

If span scoring alone doesn't fix empty results:
- Let medium-confidence candidates (0.50-0.75) enter a "pending" state
- Accumulate evidence across 2-4 discovery cycles
- Emit when confidence spikes or candidate shows monotonic alignment progress
- This provides a lower effective first-match threshold without false-positive explosion

## How to Test

Run from `web/frontend/`:

```bash
# Streaming mode (the one you're improving)
npm run test:streaming

# Non-streaming baseline (should stay at 37/53 or improve — never regress)
npm run test:streaming -- --no-streaming

# Single sample for quick iteration
npm run test:streaming -- retasy_017
npm run test:streaming -- ref_002255
npm run test:streaming -- multi_036_001_005

# Diagnostic tools (for understanding specific failures)
npx tsx test/diagnose-sample.ts retasy_003
npx tsx test/diagnose-longverse.ts ref_002255
npx tsx test/diagnose-multiverse.ts multi_067_001_004
```

The test takes ~15-20 minutes for all 53 samples. For faster iteration, test against specific failure buckets:

```bash
# Short verses that return empty (scoring/threshold issue)
npm run test:streaming -- retasy_003
npm run test:streaming -- retasy_012
npm run test:streaming -- ref_036001

# Long verses (span scoring target)
npm run test:streaming -- ref_002255
npm run test:streaming -- ref_024035

# Multi-verse transitions (tracking alignment target)
npm run test:streaming -- multi_036_001_005
npm run test:streaming -- multi_067_001_004
```

## Success Criteria

- Streaming accuracy: **35/53 (66%)** or higher (currently 25/53 = 47.2%)
- Non-streaming must not regress below 37/53
- No new message types or breaking changes to `WorkerOutbound`
- The browser app (`npm run dev`) must still work — tracker is used identically

## Current Constants Reference (`types.ts`)

```
SAMPLE_RATE = 16000
TRIGGER_SECONDS = 2.0          → TRIGGER_SAMPLES = 32000
MAX_WINDOW_SECONDS = 10.0      → MAX_WINDOW_SAMPLES = 160000
SILENCE_RMS_THRESHOLD = 0.005
VERSE_MATCH_THRESHOLD = 0.45
FIRST_MATCH_THRESHOLD = 0.75
RAW_TRANSCRIPT_THRESHOLD = 0.25
SURROUNDING_CONTEXT = 2
TRACKING_TRIGGER_SECONDS = 0.5 → TRACKING_TRIGGER_SAMPLES = 8000
TRACKING_SILENCE_TIMEOUT = 4.0 → TRACKING_SILENCE_SAMPLES = 64000
TRACKING_MAX_WINDOW_SECONDS = 5.0 → TRACKING_MAX_WINDOW_SAMPLES = 80000
STALE_CYCLE_LIMIT = 4
LOOKAHEAD = 5
```
