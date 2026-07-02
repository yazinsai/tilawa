# Phase 3: Streaming Accuracy 58-60% to 66%+

**Baseline:** 31-32/53 (58-60%) streaming, 37/53 non-streaming
**Target:** 35/53 (66%+) streaming, no non-streaming regression
**Gap:** ~5-6 streaming-specific failures (16 fail in both modes = model limit)

## Changes

### #1: Transcript-shape tracking fallback (tracker.ts)

**Problem:** Char-level fallback is gated to `>=10 word` verses. The real issue is the MODEL outputs spaceless phonemes — word-level alignment fails for ANY verse when the transcript has no spaces.

**Fix:**
- Trigger char-level fallback based on transcript shape (`<=1 spaces`), not verse length
- In `_charLevelProgress`, compare no-space text to no-space verse
- Add monotonic guard: max word jump per cycle (+5) to prevent false leaps
- Remove the `trackingVerseWords.length >= 10` gate

### #2: Extend partial scoring to 15+ word verses (quran-db.ts)

**Problem:** `fastPartialRatio` only runs for 30+ word verses (or 40+ without hint). Medium-length verses (15-29 words) still suffer from `ratio()` length-mismatch penalty.

**Fix:**
- Lower gate from `wc < 30` to `wc < 15` (and `wc < 20` without hint)
- Keep 0.85 discount factor
- Keep top-N candidate approach for performance
- NOT extending to all verses — short verses (3-14 words) don't have length-mismatch issues

### #3: Anti-cascade hysteresis (tracker.ts)

**Problem:** After verse emit, residual block (`partialRatio > 0.7`) is global — blocks correct continuations too. No protection against jumping to unrelated verses.

**Fix:**
- After verse emit, for 2 cycles: exempt continuation candidates from residual check
- For non-continuation jumps in those 2 cycles: require higher threshold (0.65+)
- Continuation = next 1-3 verses in sequence from lastEmittedRef

### #4: Soft pending first-match (tracker.ts) — ONLY IF NEEDED

**Problem:** `FIRST_MATCH_THRESHOLD = 0.75` blocks correct first matches that score 0.55-0.70.

**Fix:** Accumulate candidate support over 2-4 discovery cycles. Emit when aggregate support crosses threshold even if top-1 identity varies per cycle.

**Risk:** Phase 1 tried "first-match confirmation (2 cycles)" with no effect. This "aggregate" approach is slightly different but uncertain. Implement only if still below 35/53 after #1-#3.

## Execution Order

1. Run fresh baseline test
2. Implement #1, test targeted short-verse + multi-verse samples
3. Implement #2, test empty-result + long-verse samples
4. Implement #3, run full suite
5. Implement #4 only if still below target
6. Final full suite + non-streaming regression check

## Constraints

Same as Phase 1-2: don't modify mel.ts, ctc-decode.ts, levenshtein.ts, phoneme-aligner.ts, correction.ts. Don't change validate-streaming.ts test logic.
