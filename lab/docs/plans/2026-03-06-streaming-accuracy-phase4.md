# Phase 4: Minimal Beam + Continuation Scoring

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Raise streaming accuracy from 29-33/53 to 35+/53 by delaying commitment until stable and extending partial scoring to short continuation verses.

**Architecture:** Two independent experiments on separate branches. Experiment A adds a beam to discovery mode in `tracker.ts` — tracks top-5 candidates across cycles and delays commitment until the leader is stable. Experiment B extends partial scoring in `quran-db.ts` to continuation candidates regardless of word count. Both are tested independently, then stacked if both improve results.

**Tech Stack:** TypeScript, ONNX runtime, tsx test runner

---

### Task 1: Create branch and run baseline

**Files:**
- None (baseline measurement only)

**Step 1: Create experiment A branch**

Run: `cd /Users/rock/ai/projects/offline-tarteel && git checkout -b phase4-beam main`
Expected: Branch created

**Step 2: Run streaming baseline 3 times to establish current median**

Run: `cd /Users/rock/ai/projects/offline-tarteel/web/frontend && npx tsx test/validate-streaming.ts 2>&1 | tail -5`
Expected: Score in 29-33/53 range. Record the number. Repeat 2 more times.

**Step 3: Record baseline**

Note the 3 scores and median. This is the comparison point for Experiment A.

---

### Task 2: Add beam constants to types.ts

**Files:**
- Modify: `web/frontend/src/lib/types.ts:111-112`

**Step 1: Add beam constants after LOOKAHEAD**

Add at the end of `web/frontend/src/lib/types.ts` (after line 112):

```ts
// Beam (discovery-mode hypothesis tracking)
export const BEAM_HISTORY_SIZE = 5;
export const BEAM_STREAK_MIN = 2;
export const BEAM_PRESENCE_MIN = 3;
```

**Step 2: Commit**

```bash
git add web/frontend/src/lib/types.ts
git commit -m "feat: add beam constants for Phase 4 streaming accuracy"
```

---

### Task 3: Add beam state and helper to tracker.ts

**Files:**
- Modify: `web/frontend/src/lib/tracker.ts:1-19` (imports), `103-123` (class fields + constructor)

**Step 1: Add beam constant imports**

In `web/frontend/src/lib/tracker.ts`, add to the import block (line 5-19):

```ts
import {
  SAMPLE_RATE,
  TRIGGER_SAMPLES,
  MAX_WINDOW_SAMPLES,
  SILENCE_RMS_THRESHOLD,
  VERSE_MATCH_THRESHOLD,
  FIRST_MATCH_THRESHOLD,
  RAW_TRANSCRIPT_THRESHOLD,
  SURROUNDING_CONTEXT,
  TRACKING_TRIGGER_SAMPLES,
  TRACKING_SILENCE_SAMPLES,
  TRACKING_MAX_WINDOW_SAMPLES,
  STALE_CYCLE_LIMIT,
  LOOKAHEAD,
  BEAM_HISTORY_SIZE,
  BEAM_STREAK_MIN,
  BEAM_PRESENCE_MIN,
} from "./types";
```

**Step 2: Add beam state fields**

In the `RecitationTracker` class, after the `cyclesSinceEmit` field (line 111), add:

```ts
  // Beam state — tracks top candidates across discovery cycles
  private beamHistory: {surah: number, ayah: number, score: number}[][] = [];
  private beamLeader: {surah: number, ayah: number, streak: number} | null = null;
```

**Step 3: Add beam helper methods**

After `_exitTracking()` (after line 555), add:

```ts
  private _isContinuation(surah: number, ayah: number): boolean {
    if (!this.lastEmittedRef) return false;
    return (
      surah === this.lastEmittedRef[0] &&
      ayah >= this.lastEmittedRef[1] + 1 &&
      ayah <= this.lastEmittedRef[1] + 3
    );
  }

  private _updateBeam(
    candidates: {surah: number, ayah: number, score: number}[],
  ): void {
    this.beamHistory.push(candidates);
    if (this.beamHistory.length > BEAM_HISTORY_SIZE) {
      this.beamHistory.shift();
    }

    const top = candidates[0];
    if (!top) {
      this.beamLeader = null;
      return;
    }

    if (
      this.beamLeader &&
      this.beamLeader.surah === top.surah &&
      this.beamLeader.ayah === top.ayah
    ) {
      this.beamLeader.streak++;
    } else {
      this.beamLeader = { surah: top.surah, ayah: top.ayah, streak: 1 };
    }
  }

  private _shouldCommitBeam(
    match: Record<string, any>,
    effectiveThreshold: number,
  ): boolean {
    if (match.score < effectiveThreshold) return false;

    // Continuations commit immediately — strong prior
    if (this._isContinuation(match.surah, match.ayah)) return true;

    // First match ever — no beam history to use, commit immediately
    if (!this.hasEverMatched) return true;

    const leader = this.beamLeader;
    if (!leader || leader.surah !== match.surah || leader.ayah !== match.ayah) {
      return false;
    }

    // Stable leader: top-1 for BEAM_STREAK_MIN consecutive cycles
    if (leader.streak >= BEAM_STREAK_MIN) return true;

    // Consistent presence: appeared in top-5 for BEAM_PRESENCE_MIN of last 4 cycles
    if (this.beamHistory.length >= 4) {
      const last4 = this.beamHistory.slice(-4);
      let appearances = 0;
      for (const cycle of last4) {
        if (cycle.some(c => c.surah === match.surah && c.ayah === match.ayah)) {
          appearances++;
        }
      }
      if (appearances >= BEAM_PRESENCE_MIN) return true;
    }

    // Fallback: after BEAM_HISTORY_SIZE cycles with no commit, just commit
    if (this.beamHistory.length >= BEAM_HISTORY_SIZE) return true;

    return false;
  }

  private _resetBeam(): void {
    this.beamHistory = [];
    this.beamLeader = null;
  }
```

**Step 4: Reset beam on tracking exit and entry**

In `_enterTracking()` (line 517), add `this._resetBeam();` at the end of the method.

In `_exitTracking()` (line 525), add `this._resetBeam();` at the end, before the closing brace.

**Step 5: Commit**

```bash
git add web/frontend/src/lib/tracker.ts
git commit -m "feat: add beam state and helpers to tracker"
```

---

### Task 4: Integrate beam into _handleDiscovery

**Files:**
- Modify: `web/frontend/src/lib/tracker.ts:312-464` (`_handleDiscovery` method)

**Step 1: Replace the discovery logic**

Replace the entire `_handleDiscovery` method with:

```ts
  private async _handleDiscovery(): Promise<WorkerOutbound[]> {
    const messages: WorkerOutbound[] = [];

    if (this.newAudioCount < TRIGGER_SAMPLES) return messages;
    this.newAudioCount = 0;
    this.cyclesSinceEmit++;

    // Skip silent chunks
    const tail = this.fullAudio.slice(-TRIGGER_SAMPLES);
    if (isSilence(tail)) return messages;

    // Transcribe
    const { text } = await this.transcribe(this.fullAudio.slice());
    if (!text || text.trim().length < 5) return messages;

    // Skip if transcription is mostly residual from last emitted verse
    if (this.lastEmittedText) {
      const residual = partialRatio(text, this.lastEmittedText);
      if (residual > 0.7) return messages;
    }

    // Use beam leader as hint if we have one (gets continuation bonuses flowing early)
    const hint = this.beamLeader
      ? [this.beamLeader.surah, this.beamLeader.ayah] as [number, number]
      : this.lastEmittedRef;

    // Match against QuranDB with top-5 candidates
    const match = this.db.matchVerse(
      text,
      RAW_TRANSCRIPT_THRESHOLD,
      4,
      hint,
      5,
    );

    if (!match) {
      this._updateBeam([]);
      messages.push({ type: "raw_transcript", text, confidence: 0 });
      return messages;
    }

    // Build candidate list from match + runners_up
    const candidates: {surah: number, ayah: number, score: number}[] = [
      { surah: match.surah, ayah: match.ayah, score: match.score },
    ];
    for (const ru of (match.runners_up ?? [])) {
      if (ru.surah !== match.surah || ru.ayah !== match.ayah) {
        candidates.push({ surah: ru.surah, ayah: ru.ayah, score: ru.score });
      }
      if (candidates.length >= 5) break;
    }
    this._updateBeam(candidates);

    // Anti-cascade: shortly after an emit, require higher threshold for
    // non-continuation jumps to prevent false positives from cascading
    let effectiveThreshold = this.hasEverMatched
      ? VERSE_MATCH_THRESHOLD
      : FIRST_MATCH_THRESHOLD;

    if (this.hasEverMatched && this.cyclesSinceEmit <= 2 && this.lastEmittedRef) {
      if (!this._isContinuation(match.surah, match.ayah)) {
        effectiveThreshold = Math.max(effectiveThreshold, 0.65);
      }
    }

    // Beam commit decision
    if (!this._shouldCommitBeam(match, effectiveThreshold)) {
      const score = Math.round(match.score * 100) / 100;
      messages.push({ type: "raw_transcript", text, confidence: score });
      return messages;
    }

    const ref: [number, number] = [match.surah, match.ayah];

    // Ambiguity guard: only suppress when scores are nearly identical
    // and the transcript hasn't clearly differentiated the verses.
    const runnersUp: Record<string, any>[] = match.runners_up ?? [];
    if (runnersUp.length >= 2) {
      const matchVerse = this.db.getVerse(match.surah, match.ayah);
      let altRunner: Record<string, any> | null = null;
      for (const ru of runnersUp) {
        if (ru.surah !== match.surah || ru.ayah !== match.ayah) {
          altRunner = ru;
          break;
        }
      }
      if (altRunner && altRunner.score >= runnersUp[0].score * 0.97) {
        const altVerse = this.db.getVerse(altRunner.surah, altRunner.ayah);
        if (matchVerse && altVerse) {
          const w1 = matchVerse.phonemes_joined.split(" ");
          const w2 = altVerse.phonemes_joined.split(" ");
          let sharedPrefix = 0;
          for (let i = 0; i < Math.min(w1.length, w2.length); i++) {
            if (w1[i] === w2[i]) sharedPrefix++;
            else break;
          }
          if (sharedPrefix >= 8) {
            const textWords = text.split(" ").length;
            if (textWords <= sharedPrefix + 2) {
              messages.push({
                type: "raw_transcript",
                text,
                confidence: Math.round(match.score * 100) / 100,
              });
              return messages;
            }
          }
        }
      }
    }

    // Dedup: skip if same verse was just sent
    if (
      this.lastEmittedRef &&
      this.lastEmittedRef[0] === ref[0] &&
      this.lastEmittedRef[1] === ref[1]
    ) {
      return messages;
    }

    const verse = this.db.getVerse(match.surah, match.ayah);
    const surrounding = getSurroundingVerses(
      this.db,
      match.surah,
      match.ayah,
    );

    messages.push({
      type: "verse_match",
      surah: match.surah,
      ayah: match.ayah,
      verse_text: verse?.text_uthmani ?? match.text ?? "",
      surah_name: verse?.surah_name ?? "",
      confidence: Math.round(match.score * 100) / 100,
      surrounding_verses: surrounding,
    });

    this.hasEverMatched = true;
    this.cyclesSinceEmit = 0;

    const ayahEnd = match.ayah_end;
    const effectiveRef: [number, number] = ayahEnd
      ? [match.surah, ayahEnd]
      : ref;
    this.prevEmittedRef = this.lastEmittedRef;
    this.prevEmittedText = this.lastEmittedText;
    this.lastEmittedRef = effectiveRef;
    this.lastEmittedText =
      match.phonemes_joined ?? verse?.phonemes_joined ?? "";

    if (verse) {
      this._enterTracking(verse, ref);
    } else {
      this.fullAudio = tail.slice();
    }

    return messages;
  }
```

**Key differences from current code:**
1. Uses beam leader as hint (early continuation bonuses)
2. Builds candidate list and feeds to `_updateBeam`
3. Uses `_shouldCommitBeam` instead of raw threshold check
4. Extracted `_isContinuation` for reuse
5. All other logic (ambiguity guard, dedup, tracking entry) unchanged

**Step 2: Commit**

```bash
git add web/frontend/src/lib/tracker.ts
git commit -m "feat: integrate beam into discovery mode — delay commitment until stable"
```

---

### Task 5: Test Experiment A

**Files:**
- None (testing only)

**Step 1: Run targeted bucket tests**

Run each individually and check for improvement vs baseline:

```bash
cd /Users/rock/ai/projects/offline-tarteel/web/frontend

# Multi-verse cascade samples (beam should help most here)
npx tsx test/validate-streaming.ts multi_036_001_005
npx tsx test/validate-streaming.ts multi_067_001_004
npx tsx test/validate-streaming.ts multi_113_001_005

# Empty-first-match samples
npx tsx test/validate-streaming.ts retasy_003
npx tsx test/validate-streaming.ts retasy_012
npx tsx test/validate-streaming.ts ref_036001
```

**Step 2: Run full streaming suite 3 times**

Run: `cd /Users/rock/ai/projects/offline-tarteel/web/frontend && npx tsx test/validate-streaming.ts 2>&1 | tail -5`
Repeat 3 times. Record all scores and median.

**Step 3: Run non-streaming regression check**

Run: `cd /Users/rock/ai/projects/offline-tarteel/web/frontend && npx tsx test/validate-streaming.ts --no-streaming 2>&1 | tail -5`
Expected: >= 37/53

**Step 4: Record results**

Note median streaming score, min/max, and non-streaming score. Compare to Task 1 baseline.

---

### Task 6: Create Experiment B branch and implement continuation scoring

**Files:**
- Modify: `web/frontend/src/lib/quran-db.ts:194-209`

**Step 1: Create branch from main**

Run: `cd /Users/rock/ai/projects/offline-tarteel && git checkout -b phase4-scoring main`
Expected: Branch created from main (NOT from phase4-beam)

**Step 2: Modify the Pass 1.5 partial scoring gate**

In `web/frontend/src/lib/quran-db.ts`, replace lines 194-209 (the inner loop of Pass 1.5):

```ts
      for (let i = 0; i < scored.length; i++) {
        const [v, raw, bonus] = scored[i];
        const wc = v.phoneme_words.length;
        const isContinuation = hint && bonuses.has(`${v.surah}:${v.ayah}`);
        if (!isContinuation && (wc < 15 || (!hint && wc < 20))) continue;
        const nsVerse = v.phonemes_joined.replace(/ /g, "");
        if (noSpaceText.length >= nsVerse.length * 0.8) continue;
        let spanRaw = fastPartialRatio(noSpaceText, nsVerse);
        if (v.phonemes_joined_no_bsm) {
          const nsNoBsm = v.phonemes_joined_no_bsm.replace(/ /g, "");
          spanRaw = Math.max(spanRaw, fastPartialRatio(noSpaceText, nsNoBsm));
        }
        const discount = (isContinuation && wc < 15) ? 0.80 : 0.85;
        const effectiveRaw = Math.max(raw, spanRaw * discount);
        if (effectiveRaw > raw) {
          scored[i] = [v, effectiveRaw, bonus, Math.min(effectiveRaw + bonus, 1.0)];
          resorted = true;
        }
      }
```

**Step 3: Commit**

```bash
git add web/frontend/src/lib/quran-db.ts
git commit -m "feat: extend partial scoring to short continuation verses with 0.80 discount"
```

---

### Task 7: Test Experiment B

**Files:**
- None (testing only)

**Step 1: Run targeted bucket tests**

```bash
cd /Users/rock/ai/projects/offline-tarteel/web/frontend

# Multi-verse samples (continuation scoring should help here)
npx tsx test/validate-streaming.ts multi_036_001_005
npx tsx test/validate-streaming.ts multi_067_001_004
npx tsx test/validate-streaming.ts multi_113_001_005
```

**Step 2: Run full streaming suite 3 times**

Run: `cd /Users/rock/ai/projects/offline-tarteel/web/frontend && npx tsx test/validate-streaming.ts 2>&1 | tail -5`
Repeat 3 times. Record all scores and median.

**Step 3: Run non-streaming regression check**

Run: `cd /Users/rock/ai/projects/offline-tarteel/web/frontend && npx tsx test/validate-streaming.ts --no-streaming 2>&1 | tail -5`
Expected: >= 37/53

**Step 4: Record results**

Note median streaming score, min/max, and non-streaming score. Compare to Task 1 baseline.

---

### Task 8: Evaluate and merge winners

**Step 1: Compare results**

| Experiment | Median | Min | Max | Non-streaming |
|---|---|---|---|---|
| Baseline | ? | ? | ? | 37/53 |
| A (beam) | ? | ? | ? | ? |
| B (scoring) | ? | ? | ? | ? |

**Step 2: If both improve, test combined**

```bash
cd /Users/rock/ai/projects/offline-tarteel
git checkout -b phase4-combined main
git merge phase4-beam --no-edit
git merge phase4-scoring --no-edit
```

Run streaming suite 3 times. Record results.

**Step 3: Merge winner(s) into main**

If only A improved:
```bash
git checkout main && git merge phase4-beam --no-edit
```

If only B improved:
```bash
git checkout main && git merge phase4-scoring --no-edit
```

If combined is best:
```bash
git checkout main && git merge phase4-combined --no-edit
```

**Step 4: Update streaming accuracy spec**

Add Phase 4 results section to `tasks/streaming-accuracy-spec.md` with scores and changes made.

**Step 5: Final commit**

```bash
git add tasks/streaming-accuracy-spec.md
git commit -m "docs: update streaming accuracy spec with Phase 4 results"
```
