# Multi-Verse Benchmark Failures: Root Cause Analysis

## Problem

4 out of 53 benchmark samples fail consistently across all experiments. These are all multi-verse recordings (4-6 consecutive verses) where the ASR model (`nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0`) produces transcriptions too garbled for verse identification.

## Failures

### 1. `multi_036_001_005` — Surah Ya-Sin 36:1-5

**Transcription:**
```
م انكرسلينراط تنزيل العزيز الرحيم
```

**Expected (5 verses):**
```
36:1  بسم الله الرحمن الرحيم يس
36:2  والقرءان الحكيم
36:3  انك لمن المرسلين
36:4  علي صرط مستقيم
36:5  تنزيل العزيز الرحيم
```

**Diagnosis:** The model collapses most of the recording into garbled text. Only the tail ("تنزيل العزيز الرحيم" = verse 5) is recognizable. The opening "يس" (Ya-Sin) is completely lost, and verses 2-4 are fused into nonsense. This recording starts with "Yaseeen" (the letter name), which the model doesn't handle — it's not standard Arabic speech, it's a Quranic letter name.

**Result:** Only 36:5 matched. Score 0.73.

---

### 2. `multi_113_001_005` — Surah Al-Falaq 113:1-5

**Transcription:**
```
قل اعوذ برب الفللق من شر ما خلق ومن شر غاسق اذا وقب ومن شر النفااثات في العقد ومن شر حاسد اذا حسد
```

**Expected (5 verses):**
```
113:1  بسم الله الرحمن الرحيم قل اعوذ برب الفلق
113:2  من شر ما خلق
113:3  ومن شر غاسق اذا وقب
113:4  ومن شر النفثت في العقد
113:5  ومن شر حاسد اذا حسد
```

**Diagnosis:** The transcription is actually quite good! The main issues are minor:
- "الفللق" vs "الفلق" (doubled ل)
- "النفااثات" vs "النفثت" (extra vowels + different ending)
- Missing Bismillah prefix

The real problem is that `match_verse(max_span=8)` tries to match this as a single span. The entire 5-verse text is treated as one block. The span matching finds a poor single-verse match (31:12 at score 0.49) rather than splitting it into 5 verses.

**Root cause:** The span matching in `run_on_full_transcript` calls `match_verse()` which tries spans up to length 8 — but it tries them *within the same surah*. When the full text matches a 5-verse span of 113:1-5, it SHOULD score well. The issue is likely that the Bismillah is missing from the transcription, and verse 113:1 in the database starts with "بسم الله الرحمن الرحيم" — so the span text includes the Bismillah but the transcription doesn't, hurting the Levenshtein score.

**This is fixable.** The `match_verse` function already strips Bismillah for single-verse matching (`text_clean_no_bsm`), but for span matching it uses the FULL text including Bismillah for the first verse in a span. Applying `text_clean_no_bsm` to the first verse in span matching would likely fix this.

---

### 3. `multi_114_001_006` — Surah An-Nas 114:1-6

**Transcription:**
```
قل اعوذ برب الناس ملك الناس اله الناس الناس من ش الوسواس الخناس الذي يوسوسوس في صدور الناس منجنه والنا
```

**Expected (6 verses):**
```
114:1  بسم الله الرحمن الرحيم قل اعوذ برب الناس
114:2  ملك الناس
114:3  اله الناس
114:4  من شر الوسواس الخناس
114:5  الذي يوسوس في صدور الناس
114:6  من الجنه والناس
```

**Diagnosis:** Similar to 113 — the transcription is mostly correct but:
- Missing Bismillah prefix
- "الناس من ش" — the word boundary between 114:3 and 114:4 merges, and "شر" loses its ر
- "يوسوسوس" vs "يوسوس" (extra syllable)
- "منجنه والنا" vs "من الجنه والناس" (fused/truncated)

**Same root cause as #2:** Span matching is hurt by the missing Bismillah. Also, the truncated ending ("والنا" instead of "والناس") reduces the match quality.

**Partially fixable** with the Bismillah span fix.

---

### 4. `multi_055_001_004` — Surah Ar-Rahman 55:1-4

**Transcription:**
```
الرحمن الاسلام الرحمن الرحيم الرحمن خلق الانسان علمه البيان
```

**Expected (4 verses):**
```
55:1  بسم الله الرحمن الرحيم الرحمن
55:2  علم القرءان
55:3  خلق الانسن
55:4  علمه البيان
```

**Diagnosis:** The model hallucinates. The transcription has:
- "الرحمن الاسلام الرحمن الرحيم" — garbled Bismillah + first verse, with "الاسلام" appearing from nowhere
- "علم القرءان" (verse 2) is completely missing
- Verse 3 becomes "خلق الانسان" (close to "خلق الانسن")
- Verse 4 "علمه البيان" is correct

**Result:** The garbled text matches 12:1 (score 0.65) which also starts with "بسم الله الرحمن الرحيم" and has overlap. The span matching can't recover from the hallucinated "الاسلام" and missing verse 2.

**Hard to fix** without better transcription.

---

## Summary

| Sample | Root Cause | Fixable? |
|--------|-----------|----------|
| multi_036_001_005 | Model can't transcribe Quranic letter names ("يس") | No — needs model improvement |
| multi_113_001_005 | `match_verse` pass 1 can't rank surah 113 in top-20 for multi-verse input | **Yes** — prefix scoring in pass 1 |
| multi_114_001_006 | Same candidate filtering issue + some transcription errors | **Yes** — same fix |
| multi_055_001_004 | Hallucinated text + missing verse | No — needs model improvement |

## Investigation Findings (2026-03-04)

### Confirmed root cause: match_verse pass 1 candidate filtering

Pass 1 ranks verses by `ratio(full_text, verse_text)`. For multi-verse input (20 words) against short verses (4 words), this ratio is ~0.3 — too low for the correct surah to enter the top-20 candidate set. Pass 2 then never evaluates spans from the correct surah.

The Bismillah-stripped span matching in pass 2 is already correct (`text_clean_no_bsm`). The problem is strictly that the correct surah never reaches pass 2.

### Benchmark runner bug: predict() vs transcribe()

The benchmark runner (line 199) had the comment "Prefer transcribe() + streaming pipeline over predict()" but the code did the opposite:

```python
use_predict = hasattr(mod, "predict") and mode == "full"
```

Since nvidia-fastconformer has both `predict()` and `transcribe()`, it was using `predict()` → `match_verse()` directly, bypassing `run_on_full_transcript()` entirely. **Fixed** to:

```python
use_predict = hasattr(mod, "predict") and not hasattr(mod, "transcribe") and mode == "full"
```

### VerseTracker approach: tested and rejected for full transcripts

The plan proposed replacing the iterative `match_verse` loop in `run_on_full_transcript` with VerseTracker fed in 2-word chunks. This was tested extensively.

**Results by chunk size on the 5 key multi-verse samples:**

| chunk_size | multi_113 | multi_114 | multi_002 | multi_059 | multi_067 |
|-----------|-----------|-----------|-----------|-----------|-----------|
| 1 | FAIL (23 emissions, all noise) | — | — | — | — |
| 2 | OK | FAIL (10 emitted, 5 false) | FAIL (38 emitted, 36 false) | FAIL (16 emitted, 13 false) | FAIL (24 emitted, 22 false) |
| 3 | OK | FAIL (7, 3 false) | OK | FAIL (13, 12 false) | OK |
| 4 | OK | FAIL (5, 2 false) | OK | FAIL (9, 7 false) | OK |
| 5 | OK | OK | OK | FAIL (9, 6 false) | OK |
| all | FAIL (matches 46:9) | — | — | — | — |

**Problem:** No single chunk size works for all samples. Short chunks (2-word) are great for short-verse surahs (113) but produce massive false positive floods on longer texts (multi_002 = 38 emissions for 2 expected). Larger chunks reduce noise but don't eliminate it.

**Why it fails:** VerseTracker was designed for streaming where text arrives gradually from real ASR. For full transcripts, artificially splitting into chunks means:
1. Each short chunk matches many unrelated verses with moderate scores (0.5-0.8)
2. When the best match changes between chunks, the previous match is emitted as a "completed verse"
3. False positive scores overlap with true positive scores (no clean threshold exists)

**Score overlap between true and false positives (chunk_size=2):**
- multi_002 true hits: 0.697, 0.660
- multi_002 false positives: many at 0.600-0.756
- multi_067 true hits: 0.880, 0.709
- multi_067 false positive 87:2: 0.929 (HIGHER than a true hit)

### verse_tracker.py bismillah bug: found and fixed

`_find_best_match` always stored `v["text_clean"]` (with bismillah) in the match dict, even when `text_clean_no_bsm` gave the better score. This caused `_emit` to over-trim accumulated text by the bismillah length, consuming words that belonged to the next verse.

Example: 113:1 text_clean = "بسم الله الرحمن الرحيم قل اعوذ برب الفلق" (8 words). After matching the ASR output "قل اعوذ برب الفللق" (which matched the no_bsm variant), _emit trimmed 8 words instead of 4, swallowing the start of verse 113:2.

**Fixed** in `_find_best_match` to track which text variant produced the best score:
```python
matched_text = v["text_clean"]
if v.get("text_clean_no_bsm"):
    alt = self._score_verse(text, v["text_clean_no_bsm"], ...)
    if alt > score:
        score = alt
        matched_text = v["text_clean_no_bsm"]
```

This fix is valid regardless of the multi-verse detection approach and improves the streaming path too.

## Recommended Fix

**1. Fix `match_verse` pass 1 candidate selection** (in `shared/quran_db.py`):

Add prefix-aware scoring: for each verse, also compute `ratio(text_prefix, verse_text)` where `text_prefix` is the first N words of input (N = verse word count). This way, even when the full 20-word input scores poorly against a 4-word verse via full-text ratio, the first 4 words "قل اعوذ برب الفللق" would score ~0.95 against 113:1 "قل اعوذ برب الفلق" and the surah enters the top-20.

This is a minimal change (~5 lines) that directly addresses the bottleneck without architectural upheaval. The iterative match→trim→match loop in `run_on_full_transcript` already works well for 5/9 multi-verse samples; it only fails when the correct surah can't enter the candidate set.

**2. Keep the verse_tracker.py bismillah fix** — genuine bug fix that helps the streaming path.

**3. Keep the benchmark runner fix** — the code should match the documented intent.

## Files

- Audio files: `benchmark/test_corpus/multi_*.wav`
- Manifest: `benchmark/test_corpus/manifest.json`
- Verse matching: `shared/quran_db.py` (match_verse, lines 135-233)
- Verse tracker: `shared/verse_tracker.py` (_find_best_match bismillah fix)
- Streaming pipeline: `shared/streaming.py` (run_on_full_transcript)
- Benchmark runner: `benchmark/runner.py` (predict vs transcribe fix)
