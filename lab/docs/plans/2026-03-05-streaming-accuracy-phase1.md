# Phase 1: Streaming Accuracy Improvement

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Raise streaming accuracy from 22/53 (41.5%) to 35+/53 (66%+) by fixing three root causes.

**Architecture:** Three independent fixes targeting the highest-impact failure modes: (1) tracking completion uses per-cycle coverage instead of cumulative — blocks long-verse completion, (2) FIRST_MATCH_THRESHOLD re-applies after stale rollback via null lastEmittedRef — blocks cold re-starts, (3) matchVerse full-verse ratio() penalizes short transcripts against long verses — blocks discovery of long verses.

**Tech Stack:** TypeScript, ONNX runtime, tsx test runner

---

### Task 1: Fix tracking completion — use cumulative progress

**Files:**
- Modify: `web/frontend/src/lib/tracker.ts:245-252`

**Step 1: Run a targeted sample that fails due to long-verse tracking**

Run: `cd web/frontend && npx tsx test/validate-streaming.ts ref_002255`
Expected: FAIL (this is a 52s verse that can't complete with per-cycle coverage)

**Step 2: Fix the completion check**

In `tracker.ts`, replace the per-cycle coverage check (lines 245-252):

```ts
// BEFORE:
const coverage = matchedIndices.length / this.trackingVerseWords.length;
const nearEnd = matchedIndices[matchedIndices.length - 1] >= this.trackingVerseWords.length - 2;
if (coverage >= 0.8 && nearEnd) {

// AFTER:
const cumulativeCoverage = (this.trackingLastWordIdx + 1) / this.trackingVerseWords.length;
const nearEnd = this.trackingLastWordIdx >= this.trackingVerseWords.length - 2;
if (cumulativeCoverage >= 0.8 && nearEnd) {
```

Key insight: `trackingLastWordIdx` is the high-water mark updated earlier in this function. It reflects cumulative progress across all tracking cycles, not just the current window.

**Step 3: Re-run the targeted sample**

Run: `cd web/frontend && npx tsx test/validate-streaming.ts ref_002255`
Expected: Improvement (may still fail due to other root causes, but should get further)

---

### Task 2: Add hasEverMatched to decouple cold-start threshold

**Files:**
- Modify: `web/frontend/src/lib/tracker.ts` (add field + modify threshold logic)

**Step 1: Run a targeted sample that returns empty due to high first-match threshold**

Run: `cd web/frontend && npx tsx test/validate-streaming.ts retasy_003`
Expected: FAIL (empty result — threshold too high for partial audio)

**Step 2: Add hasEverMatched field and update threshold logic**

Add private field:
```ts
private hasEverMatched = false;
```

In `_handleDiscovery`, change the threshold selection (line 328-331):
```ts
// BEFORE:
const effectiveThreshold = this.lastEmittedRef === null
  ? FIRST_MATCH_THRESHOLD
  : VERSE_MATCH_THRESHOLD;

// AFTER:
const effectiveThreshold = this.hasEverMatched
  ? VERSE_MATCH_THRESHOLD
  : FIRST_MATCH_THRESHOLD;
```

Set the flag when a verse is matched (after the verse_match push, around line 413):
```ts
this.hasEverMatched = true;
```

**Step 3: Re-run targeted samples**

Run: `cd web/frontend && npx tsx test/validate-streaming.ts retasy_003`
Expected: Should now find the verse (lower threshold after first match persists through stale rollbacks)

Note: This alone won't fix first-match failures — FIRST_MATCH_THRESHOLD is still 0.75 for the actual first match. But it prevents the threshold from re-applying after stale rollback.

---

### Task 3: Lower FIRST_MATCH_THRESHOLD

**Files:**
- Modify: `web/frontend/src/lib/types.ts:100`

**Step 1: Lower the threshold**

```ts
// BEFORE:
export const FIRST_MATCH_THRESHOLD = 0.75;

// AFTER:
export const FIRST_MATCH_THRESHOLD = 0.55;
```

Rationale: Non-streaming uses effective threshold of 0.45. 0.55 is still conservative but allows 2s partial transcripts (which score ~0.4-0.6) to match.

**Step 2: Test empty-result samples**

Run: `cd web/frontend && npx tsx test/validate-streaming.ts retasy_003 && npx tsx test/validate-streaming.ts retasy_012 && npx tsx test/validate-streaming.ts ref_036001`
Expected: Previously empty results should now find verses.

---

### Task 4: Add prefix-aware scoring in matchVerse

**Files:**
- Modify: `web/frontend/src/lib/quran-db.ts:145-158` (scoring loop in matchVerse)

**Step 1: Run a long-verse sample**

Run: `cd web/frontend && npx tsx test/validate-streaming.ts ref_024035`
Expected: FAIL (80s verse, 10s window can never match full verse via ratio())

**Step 2: Add prefix scoring in the single-verse scoring loop**

In `matchVerse`, after the existing `ratio()` call (line 146), add prefix scoring:

```ts
// After: let raw = ratio(text, v.phonemes_joined);
const textWords = text.split(" ");
const verseWords = v.phonemes_joined.split(" ");
// If transcript is much shorter than verse, try prefix match
if (textWords.length >= 3 && textWords.length < verseWords.length * 0.7) {
  const prefixLen = Math.min(textWords.length + 3, verseWords.length);
  const prefix = verseWords.slice(0, prefixLen).join(" ");
  const prefixScore = ratio(text, prefix);
  // Slight discount to avoid false positives on short shared openings
  raw = Math.max(raw, prefixScore * 0.85);
}
```

Also apply to no-bsm variant:
```ts
if (v.phonemes_joined_no_bsm) {
  let noBsmRaw = ratio(text, v.phonemes_joined_no_bsm);
  const noBsmWords = v.phonemes_joined_no_bsm.split(" ");
  if (textWords.length >= 3 && textWords.length < noBsmWords.length * 0.7) {
    const prefixLen = Math.min(textWords.length + 3, noBsmWords.length);
    const prefix = noBsmWords.slice(0, prefixLen).join(" ");
    const prefixScore = ratio(text, prefix);
    noBsmRaw = Math.max(noBsmRaw, prefixScore * 0.85);
  }
  raw = Math.max(raw, noBsmRaw);
}
```

**Step 3: Test long-verse samples**

Run: `cd web/frontend && npx tsx test/validate-streaming.ts ref_024035 && npx tsx test/validate-streaming.ts ref_002255`
Expected: Should now discover these long verses.

---

### Task 5: Run full test suite and validate

**Step 1: Run full streaming suite**

Run: `cd web/frontend && npm run test:streaming`
Expected: 35+/53 (66%+)

**Step 2: Run non-streaming baseline to confirm no regression**

Run: `cd web/frontend && npm run test:streaming -- --no-streaming`
Expected: 37/53 or better (must not regress)

**Step 3: If targets met, commit**

```bash
git add web/frontend/src/lib/tracker.ts web/frontend/src/lib/quran-db.ts web/frontend/src/lib/types.ts
git commit -m "feat: improve streaming accuracy — cumulative tracking, prefix scoring, threshold fixes"
```

---

## Failure Recovery

If streaming score doesn't reach 35/53 after all Phase 1 changes:
- Check which failure bucket improved least
- Consider Phase 2 fixes: fast rediscovery after stale exits, residual TTL
- Tune the prefix discount factor (0.85) and FIRST_MATCH_THRESHOLD (0.55)
