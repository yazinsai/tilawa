# Phase 4: Minimal Beam + Continuation Scoring

**Baseline:** 29-33/53 (55-62%) streaming, 37/53 non-streaming
**Target:** 35/53 (66%+) streaming, no non-streaming regression
**Approach:** Two independent experiments, tested separately, stackable

## Experiment A: Minimal Beam (tracker.ts)

### Problem
The tracker commits to a verse on the first discovery cycle where score >= threshold. Noisy 2-second transcripts produce different top matches each cycle, so a single noisy spike can commit to the wrong verse. Wrong commits cascade: enter tracking -> stale out -> lose continuation hint -> cascade into more wrong verses.

### Solution
Track top-5 candidates across discovery cycles. Delay commitment until the leader is stable. Feed the leading hypothesis as hint to matchVerse before committing (gets continuation bonuses flowing earlier).

### New state
```ts
private beamHistory: Array<{surah: number, ayah: number, score: number}[]> = [];
private beamLeader: {surah: number, ayah: number, streak: number} | null = null;
```

### Commit rules
1. **Continuation** (next 1-3 verses from lastEmittedRef): commit immediately if score >= threshold. No delay needed.
2. **Stable leader**: commit when streak >= 2 AND score >= threshold.
3. **Consistent presence**: appeared in top-5 for 3+ of last 4 cycles AND currently #1 -> commit.
4. **Fallback**: after 5 cycles with no commit, commit current #1 if score >= threshold.

### Early hint
While beam has an uncommitted leader, pass it as hint to matchVerse on next cycle. Gets +0.22 continuation bonus flowing before commitment.

### Constants
```ts
BEAM_HISTORY_SIZE = 5   // cycles of history to keep
BEAM_STREAK_MIN = 2     // consecutive #1 to commit
BEAM_PRESENCE_MIN = 3   // appearances in top-5 (of last 4) to commit
```

### Files changed
- `web/frontend/src/lib/tracker.ts` (~40 lines)
- `web/frontend/src/lib/types.ts` (3 constants)

---

## Experiment B: Continuation-Only Partial Scoring (quran-db.ts)

### Problem
Partial scoring (fastPartialRatio) only runs for 15+ word verses (20+ without hint). Short continuation verses (3-14 words) don't get partial scoring, even though the prior probability of them being correct is high when they're the next verse in sequence.

### Solution
Apply partial scoring to continuation candidates (hint+1/+2/+3) regardless of word count, with a stricter 0.80 discount (vs 0.85 for long verses).

### Change
In matchVerse() Pass 1.5, replace the word-count gate:
```ts
// Before:
if (wc < 15 || (!hint && wc < 20)) continue;

// After:
const isContinuation = hint && bonuses.has(`${v.surah}:${v.ayah}`);
if (!isContinuation && (wc < 15 || (!hint && wc < 20))) continue;
const discount = (isContinuation && wc < 15) ? 0.80 : 0.85;
```

### Risk assessment
Very low. Only affects ~3 verses per cycle (continuation candidates). The 0.80 discount + existing length guard (transcript must be significantly shorter than verse) prevent false positives.

### Files changed
- `web/frontend/src/lib/quran-db.ts` (~10 lines)

---

## Testing Plan

1. Implement A on branch `phase4-beam`, run streaming suite 3x, record medians
2. Implement B on branch `phase4-scoring`, run streaming suite 3x, record medians
3. If both improve: stack on `phase4-combined`, run 3x
4. Merge winner(s) into main
5. Non-streaming regression check on final result

### Acceptance criteria
- Median streaming >= 35/53
- Min streaming >= 33/53
- Non-streaming >= 37/53 (all runs)

## Constraints
- Do NOT modify mel.ts, ctc-decode.ts, levenshtein.ts, phoneme-aligner.ts, correction.ts
- Do NOT change validate-streaming.ts test logic
- Keep WorkerOutbound message types unchanged
