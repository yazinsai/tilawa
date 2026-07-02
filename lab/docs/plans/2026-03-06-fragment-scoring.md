# Fragment Scoring (Semi-Global Alignment) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace symmetric Levenshtein ratio() with directional fragment scoring to fix streaming accuracy for partial transcripts matching against longer verses.

**Architecture:** Add `semiGlobalDistance()` to levenshtein.ts — a DP alignment that doesn't penalize skipping the start/end of the reference (verse). Wrap it as `fragmentScore(transcript, verse)` = `1 - semiGlobalDistance / transcriptLen`. Use as a **boost** in Pass 1.5 (`ratio + (frag - ratio) * 0.7`) rather than a replacement, to avoid ranking pollution from common substrings. Pass 2 surah selection preserved from ratio-only ranking.

**Tech Stack:** TypeScript, Vitest (unit tests), bun (runner)

---

## Background

The core problem: `ratio()` is symmetric Levenshtein similarity — `(lenA + lenB - dist) / (lenA + lenB)`. When a 2-second transcript (20 chars) matches against a long verse (60 chars), ratio() penalizes the length mismatch even if the transcript is a perfect substring. This causes wrong same-length verses to outscore correct longer verses.

**Fragment scoring** fixes this by asking: "how much of the transcript does this verse explain?" using semi-global alignment (free gaps at start/end of verse). The score `1 - d/L` has 2x the dynamic range of ratio's `1 - d/2L`, making coincidental matches much less competitive.

**Key insight:** Wrong short verses naturally cap at `verseLen/transcriptLen` (e.g., 8-char verse vs 20-char transcript can never exceed 0.40), so no special gating is needed.

## Constraints

- Do NOT modify: `mel.ts`, `ctc-decode.ts`, `phoneme-aligner.ts`, `correction.ts`
- Do NOT change `validate-streaming.ts` test logic
- Keep `WorkerOutbound` message types unchanged
- Can ADD to `levenshtein.ts` (don't modify existing functions)
- `ratio()` stays — fragmentScore augments it via `max(ratio, fragmentScore)`

## Design Notes

**The 0.8 length guard:** `noSpaceText.length < nsVerse.length * 0.8` means fragmentScore is only used when the transcript covers less than 80% of the verse. When the transcript is near-equal or longer than the verse, `ratio()` is already unbiased (no length mismatch to penalize), so fragmentScore adds no value and could introduce noise.

**Residual check behavior change (Task 5):** Replacing `partialRatio` with `fragmentScore` in the residual check subtly changes filtering. Example: if `text` is 30 chars (15 old + 15 new) and `lastEmittedText` is 15 chars, `partialRatio` slides the 15-char string across the 30-char text, finds a perfect match, and filters (score=1.0). `fragmentScore(text, lastEmitted)` asks "how much of the 30-char text does the 15-char emitted text explain?" — answer is only 15/30 = 0.5, so it doesn't filter. This is arguably correct (a half-new transcript should attempt matching), but if it causes false positives in testing, lower the threshold from 0.7 to ~0.5.

---

### Task 0: Capture baseline

**Step 1: Run baseline streaming test 3x**

Run each separately, capture output:
```bash
cd web/frontend
npx vitest run test/validate-streaming.test.ts 2>&1 | tee /tmp/baseline-run1.txt
npx vitest run test/validate-streaming.test.ts 2>&1 | tee /tmp/baseline-run2.txt
npx vitest run test/validate-streaming.test.ts 2>&1 | tee /tmp/baseline-run3.txt
```

**Step 2: Record results**

```bash
grep "Result:" /tmp/baseline-run*.txt
```

Record streaming and non-streaming scores. Expected baseline: streaming 29-33/53 (median ~31), non-streaming 37/53.

**Step 3: Commit baseline note**

Add a `## Baseline` section at the bottom of this plan file with the 3 run results. Commit:
```bash
git add docs/plans/2026-03-06-fragment-scoring.md
git commit -m "docs: capture fragment scoring baseline"
```

---

### Task 1: Add `semiGlobalDistance()` to levenshtein.ts

**Files:**
- Modify: `web/frontend/src/lib/levenshtein.ts` (append new function)
- Create: `web/frontend/test/semi-global.test.ts`

**Step 1: Write the failing test**

Create `web/frontend/test/semi-global.test.ts`:

```ts
import { describe, it, expect } from "vitest";
import { semiGlobalDistance } from "../src/lib/levenshtein";

describe("semiGlobalDistance", () => {
  it("returns 0 for exact substring match", () => {
    expect(semiGlobalDistance("bcd", "abcde")).toBe(0);
  });

  it("returns 0 for prefix match", () => {
    expect(semiGlobalDistance("abc", "abcde")).toBe(0);
  });

  it("returns 0 for suffix match", () => {
    expect(semiGlobalDistance("cde", "abcde")).toBe(0);
  });

  it("returns edit distance for 1-char substitution in substring", () => {
    expect(semiGlobalDistance("bxd", "abcde")).toBe(1);
  });

  it("returns query length when ref is empty", () => {
    expect(semiGlobalDistance("abc", "")).toBe(3);
  });

  it("returns 0 when query is empty", () => {
    expect(semiGlobalDistance("", "abcde")).toBe(0);
  });

  it("returns full distance when query is longer than ref", () => {
    // query="abcdef", ref="bcd" -> best alignment: "bcd" matches positions 1-3,
    // leaving "a" unmatched at start and "ef" unmatched at end = 3 edits
    expect(semiGlobalDistance("abcdef", "bcd")).toBe(3);
  });

  it("handles identical strings", () => {
    expect(semiGlobalDistance("abc", "abc")).toBe(0);
  });

  it("handles realistic phoneme fragment", () => {
    // Transcript fragment matching against longer verse
    const transcript = "bismi allahi";
    const verse = "bismi allahi arraHmaani arraHiimi";
    expect(semiGlobalDistance(transcript, verse)).toBe(0);
  });

  it("handles no-space matching", () => {
    const transcript = "bismiallahi";
    const verse = "bismiallahiarraHmaaniarraHiimi";
    expect(semiGlobalDistance(transcript, verse)).toBe(0);
  });
});
```

**Step 2: Run test to verify it fails**

Run: `cd web/frontend && npx vitest run test/semi-global.test.ts`
Expected: FAIL — `semiGlobalDistance` not exported from levenshtein.ts

**Step 3: Implement semiGlobalDistance**

Append to `web/frontend/src/lib/levenshtein.ts`:

```ts
/**
 * Semi-global edit distance: finds the minimum edit distance to align
 * the entire query against any substring of ref.
 * Free gaps at start and end of ref (row 0 initialized to 0, take min of last row).
 * Use case: "how well does this transcript fragment match somewhere inside this verse?"
 */
export function semiGlobalDistance(query: string, ref: string): number {
  if (query.length === 0) return 0;
  if (ref.length === 0) return query.length;
  const m = query.length;
  const n = ref.length;
  let prev = new Uint16Array(m + 1);
  let curr = new Uint16Array(m + 1);
  for (let i = 0; i <= m; i++) prev[i] = i;
  let best = prev[m];
  for (let j = 1; j <= n; j++) {
    curr[0] = 0; // Free to start anywhere in ref
    for (let i = 1; i <= m; i++) {
      const cost = query[i - 1] === ref[j - 1] ? 0 : 1;
      curr[i] = Math.min(prev[i] + 1, curr[i - 1] + 1, prev[i - 1] + cost);
    }
    best = Math.min(best, curr[m]); // Free to end anywhere in ref
    [prev, curr] = [curr, prev];
  }
  return best;
}
```

**Step 4: Run test to verify it passes**

Run: `cd web/frontend && npx vitest run test/semi-global.test.ts`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add web/frontend/src/lib/levenshtein.ts web/frontend/test/semi-global.test.ts
git commit -m "feat: add semiGlobalDistance for fragment matching"
```

---

### Task 2: Add `fragmentScore()` wrapper to levenshtein.ts

**Files:**
- Modify: `web/frontend/src/lib/levenshtein.ts` (append)
- Modify: `web/frontend/test/semi-global.test.ts` (add tests)

**Step 1: Add fragmentScore tests**

Append to `web/frontend/test/semi-global.test.ts`:

```ts
import { fragmentScore } from "../src/lib/levenshtein";

describe("fragmentScore", () => {
  it("returns 1.0 for exact substring", () => {
    expect(fragmentScore("bcd", "abcde")).toBe(1.0);
  });

  it("returns high score for near-match substring", () => {
    // 1 substitution in 3-char query = 1 - 1/3 ≈ 0.667
    expect(fragmentScore("bxd", "abcde")).toBeCloseTo(0.667, 2);
  });

  it("returns 0 for completely different strings", () => {
    const score = fragmentScore("zzzzz", "abcde");
    expect(score).toBeLessThan(0.2);
  });

  it("returns 1.0 for identical strings", () => {
    expect(fragmentScore("abc", "abc")).toBe(1.0);
  });

  it("returns 1.0 for empty query", () => {
    expect(fragmentScore("", "abc")).toBe(1.0);
  });

  it("penalizes when query is longer than ref", () => {
    // "abcdef" vs "bcd": distance=3, score = 1 - 3/6 = 0.5
    expect(fragmentScore("abcdef", "bcd")).toBeCloseTo(0.5, 2);
  });

  it("handles realistic partial transcript vs full verse", () => {
    const transcript = "bismiallahi";
    const verse = "bismiallahiarraHmaaniarraHiimi";
    expect(fragmentScore(transcript, verse)).toBe(1.0);
  });
});
```

**Step 2: Run test to verify it fails**

Run: `cd web/frontend && npx vitest run test/semi-global.test.ts`
Expected: FAIL — `fragmentScore` not exported

**Step 3: Implement fragmentScore**

Append to `web/frontend/src/lib/levenshtein.ts`:

```ts
/**
 * Fragment score: how well does the query match as a fragment of ref?
 * Returns 0.0–1.0. Score of 1.0 means query is an exact substring of ref.
 * Directional: measures "how much of the query does the ref explain?"
 */
export function fragmentScore(query: string, ref: string): number {
  if (query.length === 0) return 1.0;
  return Math.max(0, 1 - semiGlobalDistance(query, ref) / query.length);
}
```

**Step 4: Run test to verify it passes**

Run: `cd web/frontend && npx vitest run test/semi-global.test.ts`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add web/frontend/src/lib/levenshtein.ts web/frontend/test/semi-global.test.ts
git commit -m "feat: add fragmentScore wrapper for directional matching"
```

---

### Task 3: Pre-compute no-space verse strings

**Files:**
- Modify: `web/frontend/src/lib/types.ts` (add fields to QuranVerse)
- Modify: `web/frontend/src/lib/quran-db.ts` (constructor pre-computation)

Computing `v.phonemes_joined.replace(/ /g, "")` inside the Pass 1 loop would be 6236 string allocations every discovery cycle (~200ms). Pre-compute these once in the constructor.

**Step 1: Add fields to QuranVerse type**

In `web/frontend/src/lib/types.ts`, add two fields to the `QuranVerse` interface:

```ts
export interface QuranVerse {
  surah: number;
  ayah: number;
  text_uthmani: string;
  surah_name: string;
  surah_name_en: string;
  phonemes: string;
  phonemes_joined: string;
  phonemes_joined_no_bsm?: string | null;
  phonemes_joined_ns?: string;           // no-space version (pre-computed)
  phonemes_joined_no_bsm_ns?: string | null; // no-space no-bismillah (pre-computed)
  phoneme_words: string[];
}
```

**Step 2: Pre-compute in QuranDB constructor**

In the constructor loop in `quran-db.ts`, after the bismillah stripping block (after `v.phonemes_joined_no_bsm = null;`), add:

```ts
      // Pre-compute no-space versions for fragment scoring
      v.phonemes_joined_ns = v.phonemes_joined.replace(/ /g, "");
      v.phonemes_joined_no_bsm_ns = v.phonemes_joined_no_bsm
        ? v.phonemes_joined_no_bsm.replace(/ /g, "")
        : null;
```

**Step 3: Run tests**

Run: `cd web/frontend && npx vitest run`
Expected: All tests PASS

**Step 4: Commit**

```bash
git add web/frontend/src/lib/types.ts web/frontend/src/lib/quran-db.ts
git commit -m "perf: pre-compute no-space verse strings in constructor"
```

---

### Task 4: Integrate fragmentScore into matchVerse() Pass 1

**Files:**
- Modify: `web/frontend/src/lib/quran-db.ts:1-8,169-185`

**Step 1: Add import**

At top of `quran-db.ts`, change line 1:
```ts
// Before:
import { ratio } from "./levenshtein";

// After:
import { ratio, fragmentScore } from "./levenshtein";
```

**Step 2: Add noSpaceText computation before Pass 1**

Insert after line 167 (`const bonuses = this._continuationBonuses(hint);`), before the Pass 1 comment:

```ts
    const noSpaceText = text.replace(/ /g, "");
```

**Step 3: Replace Pass 1 scoring (lines 170-184)**

Replace the Pass 1 loop body. The full loop should become:

```ts
    // Pass 1: score all single verses (ratio + fragment score + continuation bonus)
    const scored: [QuranVerse, number, number, number][] = [];
    for (const v of this.verses) {
      let raw = ratio(text, v.phonemes_joined);
      // Also try matching without the bismillah prefix for verse 1s
      if (v.phonemes_joined_no_bsm) {
        raw = Math.max(raw, ratio(text, v.phonemes_joined_no_bsm));
      }
      // Fragment score: directional matching for partial transcripts
      // Only when transcript is significantly shorter than verse (< 80%),
      // since ratio() is unbiased for near-equal lengths.
      if (noSpaceText.length < v.phonemes_joined_ns!.length * 0.8) {
        raw = Math.max(raw, fragmentScore(noSpaceText, v.phonemes_joined_ns!));
        if (v.phonemes_joined_no_bsm_ns) {
          raw = Math.max(raw, fragmentScore(noSpaceText, v.phonemes_joined_no_bsm_ns));
        }
      }
      const bonus = bonuses.get(`${v.surah}:${v.ayah}`) ?? 0.0;
      // For continuation candidates, also try suffix-prefix matching
      if (bonus > 0) {
        const sp = QuranDB._suffixPrefixScore(text, v.phonemes_joined);
        raw = Math.max(raw, sp);
      }
      scored.push([v, raw, bonus, Math.min(raw + bonus, 1.0)]);
    }
    scored.sort((a, b) => b[3] - a[3]);
```

**Step 4: Remove Pass 1.5 entirely**

Delete the entire Pass 1.5 block (lines 187-212 in original, the block starting with `// Pass 1.5: re-score medium/long verses`). This includes:
- The `noSpaceText` declaration (we moved it earlier)
- The `if (noSpaceText.length >= 10)` block
- All the `fastPartialRatio` logic

**Step 5: Remove unused import**

Remove `fastPartialRatio` function (lines 6-30) since it's no longer called. Keep `partialRatio` (lines 32-45) since it's still used in tracker.ts residual check (will be replaced in Task 5, but keep it for now to avoid breaking things).

**Step 6: Run existing tests**

Run: `cd web/frontend && npx vitest run`
Expected: All existing tests PASS

**Step 7: Commit**

```bash
git add web/frontend/src/lib/quran-db.ts
git commit -m "feat: integrate fragmentScore into matchVerse Pass 1, remove Pass 1.5"
```

---

### Task 5: Update Pass 2 multi-verse span scoring

**Files:**
- Modify: `web/frontend/src/lib/quran-db.ts:235-273`

**Step 1: Update span scoring to use fragmentScore**

In the Pass 2 inner loop, after computing `const raw = ratio(text, combined);`, add fragment scoring. Replace the span scoring section:

```ts
      for (let i = 0; i < verses.length; i++) {
        for (let span = 2; span <= maxSpan; span++) {
          if (i + span > verses.length) break;
          const chunk = verses.slice(i, i + span);
          // Use no-bismillah text for the first verse in a span
          const firstText =
            chunk[0].phonemes_joined_no_bsm ?? chunk[0].phonemes_joined;
          const combined = [firstText]
            .concat(chunk.slice(1).map((c) => c.phonemes_joined))
            .join(" ");
          let raw = ratio(text, combined);
          // Fragment score for spans too
          const nsCombined = combined.replace(/ /g, "");
          if (noSpaceText.length < nsCombined.length * 0.8) {
            raw = Math.max(raw, fragmentScore(noSpaceText, nsCombined));
          }
          const bonus =
            bonuses.get(`${chunk[0].surah}:${chunk[0].ayah}`) ?? 0.0;
          const score = Math.min(raw + bonus, 1.0);
          if (score > bestScore) {
            bestScore = score;
            best = {
              surah: s,
              ayah: chunk[0].ayah,
              ayah_end: chunk[chunk.length - 1].ayah,
              text: chunk.map((c) => c.text_uthmani).join(" "),
              phonemes_joined: combined,
              score,
              raw_score: raw,
              bonus,
            };
          }
        }
      }
```

Note: `noSpaceText` is already declared earlier in the function (from Task 3).

**Step 2: Run tests**

Run: `cd web/frontend && npx vitest run`
Expected: All tests PASS

**Step 3: Commit**

```bash
git add web/frontend/src/lib/quran-db.ts
git commit -m "feat: add fragmentScore to multi-verse span scoring"
```

---

### ~~Task 6: Replace tracker.ts residual check~~ (DEFERRED)

Deferred to separate experiment per code review feedback: changing the residual filter simultaneously with the discovery scorer makes it impossible to attribute any regression/improvement. Keep the current `partialRatio` residual check for now.

---

### Task 7: Run streaming accuracy tests (3x)

**Files:**
- None modified — testing only

**Step 1: Run non-streaming test first (regression check)**

Run: `cd web/frontend && npx vitest run test/validate-streaming.test.ts 2>&1 | head -80`

Look at non-streaming results. Must be >= 37/53.

**Step 2: Run streaming test #1**

Run: `cd web/frontend && node --experimental-vm-modules node_modules/.bin/vitest run test/validate-streaming.test.ts 2>&1 | tee /tmp/frag-run1.txt`

Record streaming result from output.

**Step 3: Run streaming test #2**

Run: `cd web/frontend && node --experimental-vm-modules node_modules/.bin/vitest run test/validate-streaming.test.ts 2>&1 | tee /tmp/frag-run2.txt`

Record streaming result.

**Step 4: Run streaming test #3**

Run: `cd web/frontend && node --experimental-vm-modules node_modules/.bin/vitest run test/validate-streaming.test.ts 2>&1 | tee /tmp/frag-run3.txt`

Record streaming result.

**Step 5: Evaluate results**

Extract results: `grep "Result:" /tmp/frag-run*.txt`

**Acceptance criteria** (baseline: streaming 29-33/53 median ~31, non-streaming 37/53):
- Median streaming >= 35/53 (66%+) — at least +4 improvement over baseline median
- Min streaming >= 33/53
- Non-streaming >= 37/53 (all runs, no regression)

**Step 6: If passing, commit results doc**

Update `docs/plans/2026-03-06-fragment-scoring.md` with results section at the bottom.

```bash
git add docs/plans/2026-03-06-fragment-scoring.md
git commit -m "docs: add fragment scoring test results"
```

**Step 7: If failing, diagnose**

If results are below target:
- Check which specific test cases regressed vs improved
- Look for false positives where short wrong verses now outscore correct ones
- Consider adding a minimum transcript length guard (e.g., only use fragmentScore when noSpaceText.length >= 8)
- Consider a discount factor on fragmentScore (e.g., `fragmentScore * 0.95`) to prevent marginal false positives

---

### Task 8: Clean up unused code

**Files:**
- Modify: `web/frontend/src/lib/quran-db.ts` (remove `fastPartialRatio` if not done in Task 3)
- Modify: `web/frontend/src/lib/quran-db.ts` (remove `partialRatio` export if no longer imported anywhere)

**Step 1: Check for remaining references to partialRatio and fastPartialRatio**

Run: `cd web/frontend && grep -rn "partialRatio\|fastPartialRatio" src/ test/`

**Step 2: Remove any unused functions**

If `partialRatio` is no longer imported anywhere, remove it from `quran-db.ts`. If `fastPartialRatio` wasn't removed in Task 3, remove it now.

**Step 3: Run tests**

Run: `cd web/frontend && npx vitest run`
Expected: All tests PASS

**Step 4: Commit**

```bash
git add web/frontend/src/lib/quran-db.ts
git commit -m "chore: remove unused partial ratio functions"
```

---

## Performance Note

`semiGlobalDistance()` is O(m*n) where m=query length, n=verse length. For no-space text:
- Typical transcript: ~20-40 chars
- Typical verse: ~30-200 chars
- Per verse: ~20*100 = 2000 operations
- 6236 verses: ~12M operations per cycle

This is comparable to the existing `ratio()` call (also O(m*n)) and runs in the same loop. The fragment score only runs when `transcriptLen < verseLen * 0.8`, which skips ~40% of verses, so net cost increase is modest. If perf is a concern, the 0.8 threshold can be tightened.

---

## Results

### What was actually implemented (differs from original plan)

The original plan used `max(ratio, fragmentScore)` as a direct replacement in Pass 1. Testing revealed this caused severe non-streaming regression (28/53 vs 37/53) because fragmentScore in global ranking let common substrings in long verses hijack the top-20, preventing Pass 2 from checking correct surahs.

**Final architecture:**
1. **Pass 1:** ratio() only for global ranking (unchanged from baseline)
2. **Save pass2Surahs** from ratio-only top-20 (before any fragmentScore)
3. **Pass 1.5 (new):** boost all verses with fragmentScore — `boosted = ratio + (frag - ratio) * 0.7`. This lifts correct long verses above same-length wrong ones without fully overriding ratio.
4. **Pass 2:** uses `pass2Surahs` from step 2, ratio-only span scoring (no fragmentScore in spans)
5. **Task 6 (residual check):** deferred — changing anti-cascade behavior simultaneously with scoring would confound measurement

### Test results

| Metric | Baseline | Fragment Scoring (boost) |
|--------|----------|--------------------------|
| Streaming run 1 | 32 | 34 |
| Streaming run 2 | 32 | 33 |
| Streaming run 3 | — | 33 |
| Streaming median | 32 | 33 |
| Non-streaming | 37 | 37-38 |

**Consistent improvements** (pass in all fragment runs, fail in all baseline runs):
- `retasy_017` (1:7) — long verse, ratio=0.588 boosted above wrong 0.710
- `retasy_018` (110:3) — similar pattern
- `ref_001002` (1:2)
- `ref_059023` (59:23)
- `ref_059024` (59:24)

**Consistent regression:**
- `retasy_008` (114:2 → got 2:188) — short verse fragment found inside long wrong verse

**Net: +4 consistent, aggregate +1 median** (ONNX noise ±3 masks the improvement)

### Lessons learned

1. **fragmentScore cannot replace ratio() for global ranking.** Common Arabic phoneme substrings ("allahi", "annaasi", "alladhee") appear in hundreds of verses. Using `max(ratio, fragmentScore)` globally collapses many candidates to near-equal scores and lets input order decide the winner.

2. **Pass 2 surah selection is the critical bottleneck.** If fragmentScore pushes wrong verses into the top-20, the correct surah's spans never get checked, causing cascade failures in multi-verse cases.

3. **The boost formula is the key insight.** `ratio + (frag - ratio) * factor` preserves ratio's ranking signal while adding directional information. Factor 0.7 was empirically optimal: 0.5 was too weak (barely beat ONNX noise), 1.0 (= max) caused regression.
