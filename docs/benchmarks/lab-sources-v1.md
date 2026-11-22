# Lab sources v1: mp3quran surahs with teacher word timings

Lab ticket 02. Full-surah mp3quran recordings of the 32 held-out reciters (`lab-reciters-v1.json`) on the `tilawa-lab` Modal volume, with mp3quran ayah timings, teacher word timings, and a consistency verdict for every ayah. Ticket 03 builds the spliced sessions from them. Machine-readable: `lab-sources-v1.json` holds the pool and these numbers. Per-surah detail and every excluded ayah with its reasons are in `report-v1.json` on the volume. Everything ran on Modal:

    modal run modal/lab_fetch_align.py            # fetch, decode, align, report; finished items are skipped
    modal run modal/lab_fetch_align.py --pilot    # 2 reciters x 3 surahs
    modal run modal/lab_fetch_align.py --stage align --force-align   # re-align from stored evidence, CPU only

Logic and tests: `tools/lab_sources.py`, `tools/test_lab_sources.py`.

## Result

**640 surahs from 32 reciters: 192.6 h, 48,946 ayahs and 552,556 words. 6,811 ayahs (13.9%) are excluded, leaving 149.8 h in 933 clean runs of 3 minutes or more.**

| split | reciters | hours | clean hours | ayahs | excluded | clean runs ≥ 3 min |
|---|---|---|---|---|---|---|
| practice | 16 | 97.0 | 78.4 | 24,557 | 2,499 (10.2%) | 488 |
| exam | 16 | 95.6 | 71.3 | 24,389 | 4,312 (17.7%) | 445 |

A clean run is a stretch of consecutive included ayahs, measured from the first aligned start to the last aligned end. Exam reciter Abdulrahman Alsuwayid (read 135) is entirely excluded: his timings drift against the audio by up to 7 s in every surah. In practice the exam set has 15 usable reciters.

Per reciter (hours / clean hours / excluded ayahs): see `result.reciters` in the JSON. Ranges run from Abdulhadi Kanakeri (2% excluded) to Ahmad Al-Hawashi (33%), Neamah Al-Hassan (29%) and Ibrahim Al-Asiri (24%). The high rates come from reciters who often restart inside long ayahs.

## Pool

Each reciter gets 20 surahs. Seed `20261028`, rule in `lab_sources.pool`, timed-surah lists fetched 2026-10-28:

1. Required when timed: **55, 77, 54, 26**. These refrain and shared-run surahs feed the strict slice. All 32 reciters have all four.
2. Per stratum, a shuffle seeded with `seed:readId`. Each pick takes the candidate from the juz least covered so far:
   - 8 surahs of 160–450 words (about 3–8 min), the ayah-1 hosts for short sessions;
   - 5 of 450–1,500 words;
   - 3 of over 1,500 words.
3. The pool is filled to 20 from the timed surahs of at least 160 words.

This gives 63 distinct surahs, 16 or more juz per reciter, and about 18,000 words per reciter. Surahs 32 and 44 are in every pool because each is the only host-length surah in its juz.

## Method

1. **Fetch** (CPU, 32 containers). The ayah timing JSON comes from `ayat_timing?surah=&read=`. The mp3 comes from `<server><sss>.mp3`. The last four were fetched with curl over HTTP/1.1, with stall detection and retries, after the CDN stalled them over HTTP/2; the code now uses curl for all. ffmpeg decodes it to 16 kHz mono PCM16 with `+bitexact`. `source.json` records the URL, sha256, byte count, fetch time and wav hash, plus the raw timing JSON with its sha256 and fetch time. mp3quran has no API revision, so that hash and time stand in for one.
2. **Teacher** (`obadx/muaalem-model-v3_2`, L4 GPUs, 32 containers). The whole surah is decoded in 30 s windows every 25 s. Every 40 ms frame is taken from the window whose centre is nearer, so each frame has at least 2.5 s of context and global frame *f* is exactly sample 640·*f*. Output: top-5 ids and probabilities per frame, plus the greedy phonemes. The teacher's vocabulary is asserted equal to the ref-v2 inventory.
3. **Align** (CPU, from the stored evidence only, so it can be re-run without the GPU). Each ayah goes through CTC Viterbi forced alignment (`e1_report.viterbi`) on ref-v2 phonemes. Three choices make this robust:
   - **Neighbours in sequence.** The previous, current and next ayahs' text is aligned together over their mp3quran windows with 1.5 s padding. Without the neighbours, a rhyming ayah ending (fawasil) in the padding pulled half the ayahs one ayah over (pilot: 253 of 352 ayahs flagged).
   - **Length runs collapsed.** A run of one phoneme (madd, ghunna, shadda) becomes one CTC token, and its span is then stretched over the repeat spikes. Reciters' madd lengths differ from the ref-v2 profile, and otherwise the aligner hunts for a missing alef in the next ayah.
   - **Garbage blank.** The blank may absorb any speech at 10⁻³ of the frame's best token, which covers isti'adha, a repeated phrase or noise.
   - **Ayah 1** is also aligned with the basmala as a prefix, and the fit with the lower PER is kept. The basmala ends like 55:1 (ar-rahman) and stole it otherwise. The basmala was found in 470 of the 633 surahs that can have one.
4. **Word timings.** Each word's span runs from its first token's first frame to its last token's (stretched) last frame. Word offsets come from ref-v2 `o`; the one word with no phonemes of its own (10:15) gets a zero-length span. Each word also gets `cutAfterS`, the quietest 10 ms frame between it and the next word (±40 ms), and that frame's dB against the ayah's median level.
5. **Verdict.** An ayah is excluded when any of these holds. Nothing is patched by hand.
   - **window-mismatch**: its mp3quran window misses its own aligned speech, or holds a neighbour's, by more than 0.3 s. A bad boundary therefore excludes both ayahs on either side of it.
   - **per**: the length-insensitive PER (runs squeezed) of the greedy decode over its aligned span is above 0.15.
   - **word-score**: a word's mean token posterior is under 0.2.
   - **align-failed**: no alignment exists.
   - **timing** (surah level): ayahs missing or out of order, or the timing ends more than max(60 s, a quarter of the file) before the audio does, or after it.

## Exclusions

| reason (an ayah can have several) | ayahs |
|---|---|
| per | 5,436 |
| window-mismatch | 3,146 |
| word-score | 1,212 |
| timing (surah level) | 190 |
| align-failed | 30 |

- **Timing that doesn't fit the audio, 2,127 ayahs in 28 surahs.** Five surahs fail the surah-level timing check: 299/44, 55, 62 and 66 cover only the first 46–128 s of 243–519 s files, and 135/32's timing runs past its audio. In 23 more, at least half the ayahs are window-mismatched, a constant or drifting offset:
  - 3/7 is about 1 s early throughout;
  - read 135 is off in all 19 of its other surahs;
  - 25/5, 229/14 and 300/6 are the rest.

  These timings were made for another edit of the recording.
- **Repeats and restarts, most of the remaining PER exclusions.** 4,244 of the 5,436 PER-excluded ayahs decode at least 10% longer than their text. A spot check of decodes shows the reciter going back a few words inside a long ayah ("لتنذر لتنذر", 32:3, read 6) or reciting a refrain twice (55:67, read 3). The audio really differs from the text, so these must not be labelled as clean.
- **Remaining boundary errors, 1,287 ayahs outside the offset surahs.** These are local: one mp3quran boundary 0.3–5 s off, often next to a restart.
- **Ayah 1, 158 excluded outside the offset surahs.** About 130 are huruf muqatta'at openings (الم, حم, الر, طسم, ن), whose long letter names the teacher doesn't hear as written.

## Alignment quality

The pilot covered reads 3 and 22 on surahs 55 and 77 plus one host surah each (68 and 70). Its decodes were read one by one, and dB profiles were checked at 50 ms resolution around boundaries.

- **Ayah boundaries:**
  - Where the reciter pauses (the dip within 0.3 s of the boundary is 12 dB or more below the ayah level), the dip lies inside the aligned gap at 96.5% of the pooled pauses.
  - The aligned start is 20 ms before the energy onset (median of per-surah medians; p10–p90 −40 to 0 ms).
  - The aligned end is 35 ms before the 6 dB energy offset (p10–p90 −160 to +40 ms). The tail of that spread is reverb or the final madd fading, which reaches 0.5–1 s in mosque recordings (read 3).
- **mp3quran boundaries vs alignment:** 95.8% of mp3quran ayah boundaries (mean over surahs) lie in the aligned gap between the two ayahs, within 0.3 s. The ayah window mismatch is 0 at the median and p90, and 6.3 s at p99.
- **Word boundaries:** the quietest frame in the 80 ms around an in-ayah word boundary is −5.6 dB against the ayah level (median of surah medians). The same search at word middles, as a control, gives −2.1 dB. Murattal links most words, so most boundaries aren't pauses. Ticket 03 should cut at `cutAfterS` and prefer boundaries with a low `cutDb`.
- **Teacher fit:** the ayah PER (length-insensitive) is 0 at p10 and p50, 0.17 at p90 and 0.84 at p99. The minimum word score per ayah is 0.70 at p10 and 0.90 at the median.
- **Determinism:** re-aligning from the stored evidence gives byte-identical `align.json`. Re-runs skip every finished file, and an `align.json` is recomputed only when its `version` changes or `--force-align` is given.

## Outputs

Volume `tilawa-lab`, paths relative to `/data` (`LAB_DATA` in `modal/lab_run.py`). `<sss>` is the zero-padded surah number.

    sources/mp3quran/<readId>/<sss>.wav            16 kHz mono PCM16
    sources/mp3quran/<readId>/<sss>.source.json
    sources/mp3quran/<readId>/<sss>.evidence.npz
    sources/mp3quran/<readId>/<sss>.align.json
    sources/mp3quran/catalog-v1.json               one entry per surah, for ticket 03
    sources/mp3quran/report-v1.json                this report with per-surah detail and every exclusion

`source.json`:

    {"version": 1, "readId": 3, "surah": 55, "url": "https://cdn.mp3quran.net/audio/ibrahim-asiri/r1/055.mp3", "bytes": 9876543,
     "sha256": "…", "fetchedAt": "2026-10-28T07:16:02+00:00",
     "wav": {"path": "sources/mp3quran/3/055.wav", "sha256": "…", "samples": 9875200, "durationS": 617.2, "sampleRate": 16000, "ffmpeg": "ffmpeg version …"},
     "timingUrl": "https://www.mp3quran.net/api/v3/ayat_timing?surah=55&read=3", "timingFetchedAt": "…", "timingSha256": "…",
     "timing": [{"ayah": 1, "start_time": 380, "end_time": 2960, "polygon": "…", "x": "…", "y": "…", "page": "…"}, …]}

`evidence.npz` (numpy):

    ids          uint8  [T, 5]  top-5 teacher token ids per 40 ms frame (0 = blank, i = ref-v2 inventory.tokens[i-1])
    p            float16 [T, 5] their probabilities
    greedy       int32  [n, 3]  greedy CTC phonemes: token id, first frame, last frame (inclusive)
    greedy_conf  float16 [n]    max top-1 probability over each greedy phoneme
    windows      int64  [W, 3]  window plan: start sample, first kept local frame, end kept local frame (-1 = to the end)
    frame_ms     int32          40
    model        str            obadx/muaalem-model-v3_2

Frame *f* starts at *f*·0.04 s.

`align.json` (`version: "lab-align-v2"`; times in seconds from the start of the wav):

    {"version": "lab-align-v2", "readId": 22, "surah": 55, "reciter": "Khalid Abdulkafi", "split": "practice",
     "audio": "sources/mp3quran/22/055.wav", "evidence": "sources/mp3quran/22/055.evidence.npz", "durationS": 397.95, "frameMs": 40,
     "teacher": "obadx/muaalem-model-v3_2", "reference": "ref-v2", "timingSha256": "…", "timingProblems": [],
     "preRoll": null | {"startS", "endS"},                     // mp3quran ayah 0, when listed
     "basmala": null | {"startS": 0.12, "endS": 2.2, "per": 0.0, "present": true},
     "ayahs": [{"ayah": 2, "windowS": [4.76, 6.94], "startS": 4.92, "endS": 6.88, "per": 0.0, "lenRatio": 1.0, "score": 0.9435,
                "minWordScore": 0.8907, "mismatchS": 0.0, "levelDb": -26.2, "excluded": false, "reasons": []}, …],
     "words": [{"surah": 55, "ayah": 3, "word": 1, "startS": 7.12, "endS": 7.6, "score": 0.9875, "cutAfterS": 7.625, "cutDb": -2.0}, …],
     "checks": {"wordCutDb": [p10, p50, p90], "wordMidMinDb": […], "mp3quranBoundaryInGap": 1.0, "ayahPauses": 66, "ayahDipInGap": 0.939,
                "ayahOffsetLagS": […], "ayahOnsetLagS": […]}}

How to read these fields:

- `windowS` is the mp3quran window.
- `startS` and `endS` on an ayah are its aligned speech (first word start, last word end). They are null, with `align-failed`, when alignment failed.
- `excluded` is authoritative. `words` lists every aligned ayah's words, including excluded ones; filter by `ayahs[].excluded`. Under a surah-level timing problem every ayah has `"reasons": ["timing"]` and `words` is empty.
- `word` is 1-based (`WordRef`).
- `score` is the word's mean token posterior, null for a word with no phonemes of its own.
- `cutAfterS` and `cutDb` are null for the last word of the surah, and before an excluded or unaligned ayah gap. `cutDb` is measured against the ayah's `levelDb`.
- `lenRatio` is the decoded length over the text length after squeezing runs; ≥ 1.1 with a high PER means a repeat.

`catalog-v1.json`:

    {"version": 1, "alignVersion": "lab-align-v2", "root": "data/sources/mp3quran",
     "items": [{"readId": 3, "surah": 6, "split": "exam", "reciter": "Ibrahim Al-Asiri", "durationS": 4465.9,
                "audio": "sources/mp3quran/3/006.wav", "align": "sources/mp3quran/3/006.align.json", "evidence": "sources/mp3quran/3/006.evidence.npz",
                "ayahs": 165, "excludedAyahs": [7, 10, …], "cleanRunsS": [116.76, 30.32, 221.84, …], "basmala": {…} | null}, …]}

## Cost and time

These are estimates from the function timings. Modal's billing wasn't queried.

- **Teacher:** 2.9 L4-hours over 640 surahs (about 67× realtime), roughly $2–3.
- **Fetch:** about 45 min wall on 32 two-core containers; the CDN is the bottleneck.
- **Alignment:** about 10–14 min wall per full pass on up to 64 containers. Four passes were run while the method was tuned.

Total is on the order of $10–20. Storage is about 22 GB of wav plus 0.2 GB of evidence.

## Caveats

- **Teacher ceiling.** The word timings and the PER check are the teacher's view. An ayah the teacher mishears (muqatta'at, fast passages) is excluded rather than verified, so the clean set leans toward what the teacher hears well.
- **40 ms resolution.** CTC spikes are peaky, so word starts and ends are accurate to about ±40–80 ms. Cut at `cutAfterS`, not at `endS`.
- **Reverb.** The energy checks treat a reverb tail as sound. Ayah ends sit where the voiced part stops, so a splice that drops the next 0.5–1 s of tail will be audible in reverberant recordings. That is ticket 03's crossfade problem.
- **Natural restarts are excluded, not labelled.** About 4,200 ayahs contain a real repeat. They could become a natural non-error slice later, but they would need labels first.
- **No listening yet.** Quality evidence here is numeric. Ticket 07's listens are the acceptance check.
