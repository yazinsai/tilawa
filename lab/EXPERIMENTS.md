# Benchmark results

Three test corpora: **v1** (53 samples: user recordings, EveryAyah reference, RetaSy crowdsourced), **v2** (43 samples: RetaSy expanded + EveryAyah multi-verse), and **v3** (256 samples: EveryAyah Alafasy+Husary singles/multis + TLOG-clean crowd-sourced filtered through shipped ONNX + user recordings). v3 exists to reduce the per-sample noise floor: on v1 a one-sample swing is ±1.9pp recall, whereas on v3 the same swing is ±0.4pp.

Metrics: **Recall** = fraction of expected verses found. **Precision** = fraction of emitted verses that were expected. **ExactSetAcc** = deduped emitted set exactly matches expected set, order ignored. **OrderedSeqAcc** = deduped emitted sequence exactly equals the expected ordered sequence. Older changelog entries and tables that say "SeqAcc" used the pre-rename set metric, so read them as ExactSetAcc unless a row explicitly says ordered.

Browser streaming reports now separate durable raw `verse_match` commits from silence-time `final_sequence`. Raw commit metrics remain the engineering guardrail for bad visible emissions. Final-sequence metrics measure the product contract after accumulated streaming evidence has had a chance to smooth or repair early hypotheses.

ONNX inference is non-deterministic at **±3–6 samples per run** on v1 — streaming numbers below are medians over 3 runs (except the deferred-emission changelog entry, which was measured at 5 runs).

## Shipped model

Current browser/runtime model: Zipformer2-CTC `interp-gentle-a0.5` int8 (`web/frontend/public/models/zipformer_interp_gentle_a05.int8.onnx`, 66 MB). Default engine in the Vite demo and in `@tilawa/core` (`createRecognitionSession()`); FastConformer stays behind `?engine=fastconformer`. The acoustic model is NPL-1.2; the word-level tracker is the native MIT recitation engine, now in the SDK at `packages/core/src/recitation/`. Batch champion is unchanged: Cyberistic's full-mixed text CTC FastConformer (`fastconformer_full_mixed.onnx`, 88 MB).

| Mode | Corpus | Recall | Precision | ExactSetAcc | Notes |
|---|---|---|---|---|---|
| **Zipformer browser streaming** (300ms chunks, native MIT engine) | v1 | **100%** | **100%** | **100%** | 3-repeat median; OrderedSeqAcc also 100% (53/53 every run) |
| **Zipformer browser streaming** | v2 | **100%** | **100%** | **100%** | blind check; 43/43 every run |
| **`c2c-direct-mixed-tta` full-file batch** | v1 | **100%** | **100%** | **100%** | Cyberistic champion, median across 3 reproduced runs |
| **`c2c-direct-mixed` full-file batch** | v1 | 98% | 98% | 98% | Same ONNX without 0.9x/1.1x TTA |

Historical pre-Zipformer browser/RN streaming baseline, using `fastconformer-phoneme v4-tlog` (131 MB quantized ONNX):

| Mode | Corpus | Recall | Precision | ExactSetAcc | Correct |
|---|---|---|---|---|---|
| **Browser/RN streaming** (300ms chunks, `RecitationTracker`) | v2 | **87.9%** | **68.9%** | **55.8%** | 37/43 |
| **Browser/RN streaming** | v3 | **89.3%** | **73.4%** | **58.2%** | 223–225/256 |
| Non-streaming (full-file, single `matchVerse()`) | v1 | 84.1% | 84.9% | 81.1% | 43/53 |
| Non-streaming (full-file, single `matchVerse()`) | v2 | 78.1% | 79.1% | 74.4% | 32/43 |

### Streaming changelog

**2026-09-17 — recitation engine moves into `@tilawa/core`, Zipformer becomes the SDK default** (commits `b51887b`, `cdfa864`)
The streaming phoneme engine was only reachable through the web demo; it now lives in the SDK at `packages/core/src/recitation/` (2,600 lines moved with `git mv`, no logic edits) behind a public `ZipformerSession` / `createZipformerSession()` API, and a top-level `createRecognitionSession({ engine })` selector defaults to `"zipformer"`. The browser worker, the Node stability report, and the Python harness (`lab/experiments/zipformer-ctc/harness.ts`) are now three thin consumers of one implementation instead of one implementation plus two copies of the host loop — the invariant this buys is that the demo's numbers and the lab's numbers can no longer drift apart silently. ONNX stays injected: `{ ort, model }` for web/node, `{ session, Tensor }` for React Native, so the package still imports no runtime. `bridgeGapAyahs` and the emission gate moved out of the harness into `emission.ts`, which is what made the harness a consumer rather than a fork.

Numbers: precision 100.0% → **100.0%** (0pp), SeqAcc 100.0% → **100.0%** (0pp), recall 100.0% → **100.0%** (0pp) on v1 (53/53). Same on v2 blind check (43/43). A pure refactor should move nothing, and nothing moved — the point of the measurement is that three repeats on each corpus were identical, as expected from this deterministic engine.

Measurement commands:
```
npx tsx test/stability-report.ts --engine=zipformer --repeats=3 --json=test/sdk-zipformer-v1-stability.json
npx tsx test/stability-report.ts --engine=zipformer --repeats=3 --corpus=test_corpus_v2 --json=test/sdk-zipformer-v2-stability.json
cd lab && ZIPFORMER_ORT_DIR=… .venv/bin/python -m benchmark.runner --experiment zipformer-ctc --corpus test_corpus
```
Raw JSON at `web/frontend/test/sdk-zipformer-v{1,2}-stability.json`; Python harness at `lab/benchmark/results/2026-09-17_103550.json` (100% / 100% / 100%, 0.84s/sample). 110 vitest cases pass — 60 in `packages/core`, 50 in `web/frontend` — of which 6 are new: the session API end-to-end against a scripted ORT stub that replays an Al-Fatiha phoneme timeline, plus the engine selector's default and its FastConformer branch. Note for future runs: `onnxruntime-node` aborts with exit 134 during process teardown *after* the stability report writes its JSON and prints its summary — a known ORT teardown crash, not a recognition failure.

**2026-09-16 — native MIT recitation engine replaces vendored tracker** (commit `9fd87fd`)
The Zipformer worker now runs `packages/core/src/recitation/`, a clean-room MIT TypeScript engine written from the behavioural spec (`docs/specs/recitation-engine-spec.md`) plus 23 dump-vector oracles. The vendored reference engine is gone from the frontend. Nothing was copied from that tree — vectors stayed exact, and ZipformerHost scores did not move.

Numbers: precision 100.0% → **100.0%** (0pp), SeqAcc 100.0% → **100.0%** (0pp), recall 100.0% → **100.0%** (0pp) on v1. Same pattern on v2 blind check. v3 248/256 unchanged (same 8 misses).

Measurement commands:
```
npx tsx test/stability-report.ts --engine=zipformer --repeats=3 --json=test/track-c-v1-stability.json
npx tsx test/stability-report.ts --engine=zipformer --repeats=3 --corpus=test_corpus_v2 --json=test/track-c-v2-stability.json
npx tsx test/stability-report.ts --engine=zipformer --repeats=1 --corpus=test_corpus_v3 --json=test/track-c-v3-stability.json
```
Raw JSON at `web/frontend/test/track-c-v{1,2}-stability.json` and `track-c-v3-stability.json`. 3-repeat medians: v1 53/53 every run, v2 43/43 every run; v3 248/256; q-lab 572/583.

**2026-09-16 — Zipformer (interp-gentle-a0.5) is the default browser engine** (commit `9cfd295`)
The Vite demo now loads streaming Zipformer2-CTC (`interp-gentle-a0.5` int8, 66 MB) by default so the live UI matches the promoted acoustic model. FastConformer remains a complete fallback (`?engine=fastconformer` or `localStorage.tilawaEngine=fastconformer`); the status pill shows which engine is running. The ONNX and phoneme lexicon are NPL-1.2 Derivatives; the word-level tracker is the vendored reference engine pending a native port. Deploy pulls those two assets from GitHub release `yazinsai/tilawa` v0.3.0; `zipformer_interp_gentle_a05.io.json` is committed.

Numbers: precision 66.8% → **100.0%** (+33.2pp), SeqAcc 47.2% → **100.0%** (+52.8pp), recall 78.6% → **100.0%** (+21.4pp) on v1. Same pattern on v2 blind check: precision 68.9% → **100.0%** (+31.1pp), SeqAcc 55.8% → **100.0%** (+44.2pp), recall 87.9% → **100.0%** (+12.1pp). All three repeats were identical (v1 53/53, v2 43/43). v1 "before" is the last measured v1 streaming row (deferred-emission, 5-run); later tracker gates were scored on v2/v3 only. v2 "before" is the shipped phoneme FastConformer headline. `fastconformer_phoneme_q8.onnx` is not in this checkout, so the FastConformer path of `stability-report.ts` was not re-run.

Measurement commands:
```
npx tsx test/stability-report.ts --engine=zipformer --repeats=3 --json=test/zipformer-default-stability.json
npx tsx test/stability-report.ts --engine=zipformer --repeats=3 --corpus=test_corpus_v2 --json=test/zipformer-default-v2-stability.json
```
Raw JSON at `web/frontend/test/zipformer-default-stability.json` and `…-v2-stability.json`. 72 vitest cases pass (5 new: default engine is zipformer; `?engine=fastconformer` is honoured).

**2026-04-25 — decode-stability gate on single-cycle commits** (file: `web/frontend/src/lib/tracker.ts`)
A context-sweep diagnostic (`web/frontend/test/diagnose-context-sweep.ts`) measured how the model's CTC greedy decode of audio prefixes compares to its decode of the full audio. On v1 the result was striking: across prefix lengths from 1s to 5s, **~50% of every prefix-decode token gets revised** when full audio context arrives (median LCP / |prefix-decode| ≈ 0.50). Full-audio WER vs the expected phoneme reference is 14%, so the offline ceiling is fine — but every short-prefix decode sits in a regime where half its emissions are non-final because the FastConformer encoder uses bidirectional attention to refine early frames once more audio is in.

The browser's `RecitationTracker` was committing `verse_match` on single-cycle `clearMargin` paths — riding those unstable predictions. The fix gates that one path: track `lastRawPhonemes`, and require the current decode's Levenshtein ratio to the previous cycle's decode be ≥ 0.70 before allowing a single-cycle clearMargin commit. Repeated-leader and finalFlush commits are not gated (they have their own multi-cycle protection). Commit is denied with no diagnostic noise — the tracker either commits in this cycle, defers to the next, or eventually fires via the existing `repeatedLeader` path. Continuation jumps (the next ayah of the verse currently being tracked) are not gated either, since they're not the "early-frame instability" failure mode.

The gate is on by default; set `DECODE_STABILITY_GATE_OFF=1` in env to disable for benchmarking.

Numbers (3-repeat median):
- v3 (256 samples): recall 82.1% → **89.3%** (+7.2pp), precision 64.1% → **73.4%** (+9.3pp), SeqAcc 46.1% → **58.2%** (+12.1pp). Per-run correct [213, 204, 204] → [223, 225, 224]. **Stable-pass 186 → 216 (+30), stable-fail 30 → 25 (−5), flaky 40 → 15 (−25)** — the gate doesn't just lift the median, it makes the pipeline noticeably more deterministic.
- v2 blind check: recall 85.6% → **87.9%** (+2.3pp), precision 66.6% → **68.9%** (+2.3pp), SeqAcc 53.5% → **55.8%** (+2.3pp). Per-run correct [37, 36, 36] → [37, 37, 37]. Smaller gain than v3, consistent with v2 being mostly clean professional recitations where short-prefix decodes are less ambiguous to begin with — the bigger v3 win comes from the 80 noisier TLOG-clean samples where decode stability matters more.

The improvement is roughly an order of magnitude bigger than the prior streaming experiments because it targets a different failure class: not "score threshold tuning" (which the matcher/tracker attempts on 2026-04-21 exhausted) but "the upstream signal that scores are computed from is unreliable until enough context arrives." Three matcher tweaks moved nothing measurable on v1; this one moved v3 SeqAcc 12pp.

Targets specifically: long single-verse and multi-verse samples where an early streaming chunk happened to score well against a wrong verse and got committed before the correct verse's evidence accumulated. On v3 baseline-vs-gated diffs, samples like `tlog_m020_010_105` (got `[20:34]` baseline, suppressed and recovered with gate), `ea_alafasy_034005` (`[22:51, 22:52, 22:53]` → `[22:51]`), `multi_055_001_004` (`[20:5, 55:2, 55:3, 55:4]` → correct on gated runs in v1) flip from stable-fail to stable-pass.

The diagnostic that motivated this: `npx tsx test/diagnose-context-sweep.ts` — for each test sample, runs inference on prefixes [1, 2, 3, 5, 10]s of audio and reports phoneme WER vs the expected reference plus prefix-vs-full-decode stability. Reproduces the ~50% instability finding in ~2 min on a Mac.

Measurement commands:
```
DECODE_STABILITY_GATE_OFF=1 npx tsx test/stability-report.ts --repeats=3 --corpus=test_corpus_v3 --json=test/stab-gate-baseline-v3.json
                            npx tsx test/stability-report.ts --repeats=3 --corpus=test_corpus_v3 --json=test/stab-gate-on-v3.json
DECODE_STABILITY_GATE_OFF=1 npx tsx test/stability-report.ts --repeats=3 --corpus=test_corpus_v2 --json=test/stab-gate-baseline-v2.json
                            npx tsx test/stability-report.ts --repeats=3 --corpus=test_corpus_v2 --json=test/stab-gate-on-v2.json
```
Raw JSON at `web/frontend/test/stab-gate-{baseline,on}-{v2,v3}.json`. 38 vitest cases pass (no new cases — existing coverage exercises the unchanged commit paths; the gated path is exercised by `stability-report` on real audio because mocked tests run a single cycle and never hit the multi-cycle stability comparison).

**2026-04-22 — silence-flush pending emission on final flush** (commit `508844b`)
When the utterance ends and the tracker has auto-advanced to a pending next-verse emission that never got fresh-audio confirmation, emit the pending message instead of rolling it back — but only when the advance had strong acoustic evidence at the time. Specifically, capture `prefixScore - suffixScore` as `pendingEmissionMargin` at advance time (from the existing `ADVANCE_RELATIVE_MARGIN < 3.0` gate). On `finalFlush`, emit the pending message only when `pendingEmissionMargin < ADVANCE_FLUSH_STRICT_MARGIN` (0.5, much tighter than the normal advance gate). The tighter threshold prevents one-verse overshoot when the reciter actually stopped at the penultimate verse.

Numbers (3-repeat median):
- v3 (256 samples): recall 83.4% → 83.7% (+0.3pp), precision 63.5% → 64.4% (+0.9pp), SeqAcc 44.1% → **46.1%** (+2.0pp). Per-run correct [204, 207, 207] → [209, 212, 208]. **Stable-fail 34 → 28 (−6)** — the six samples gained are the structural win, not variance.
- v2 blind check: recall 82.7% → **85.6%** (+2.9pp), precision 63.7% → 68.1% (+4.4pp), SeqAcc 46.5% → 48.8% (+2.3pp). Same-direction movement on v2 confirms it's not an overfit to v3.

Targets specifically: `multi_114_001_006` (Al-Nas 1-6 dropping verse 6 on silence), `user_ikhlas_2_3` (Al-Ikhlas verses 2-3 dropping verse 3), and similar last-verse-of-span cases where utterance ends before the pending emission could be confirmed by fresh audio. SeqAcc gains more than recall because this fix specifically repairs the last component of ordered sequences, which is exactly what exact-match SeqAcc weighs.

Measurement commands:
```
npx tsx test/stability-report.ts --repeats=3 --corpus=test_corpus_v3 --json=test/silence-flush-v3-stability.json
npx tsx test/stability-report.ts --repeats=3 --corpus=test_corpus_v2 --json=test/silence-flush-v2-stability.json
```
Raw JSON at `web/frontend/test/silence-flush-v3-stability.json` and `…-v2-stability.json`. 38 vitest cases pass (including 2 new coverage cases for strict-margin-emits and loose-margin-suppresses).

**2026-04-22 — v3 benchmark corpus (256 samples)** (scripts: `benchmark/build_v3_corpus.py`, `benchmark/augment_v3_corpus.py`, `benchmark/tlog_filter_v3.py`)
After four consecutive falsified streaming experiments (three matcher/tracker + v7 streaming-aug training) all landing inside or just outside the ±3–6-sample v1 variance envelope, the bottleneck became measurement fidelity rather than idea generation. Rebuilt the corpus at ~5× the size.

Sources and composition:
- **EveryAyah singles (140)**: 80 short + 60 medium + 20 long, drawn by reciter-alternating across `Alafasy_128kbps` and `Husary_128kbps`, picked from shuffled Quran pools that don't overlap (surah, ayah) with v1/v2.
- **EveryAyah multi-ayah (20)**: 29 hand-picked 3–6-ayah sequences concatenated via ffmpeg pipe→f32le with 0.5s silence gaps and written at 16 kHz mono. 10 Shatri sequences 404'd (reciter dir not on everyayah.com); Alafasy + Husary sequences all succeeded. `Shatri_128kbps` does not exist on the CDN; six stale 404-HTML files landed on disk as `.mp3` and were removed during pruning, cutting 6 samples.
- **TLOG-clean 80**: replaces RetaSy entirely. TLOG's `clean` split is still noisy at the transcription level, so each candidate was streamed with `Audio(decode=False)`, ffmpeg-decoded to 16 kHz mono, greedy-CTC transcribed through the shipped FastConformer phoneme ONNX, and phoneme-compared against the canonical phoneme string for the filename-referenced (surah, ayah) in `quran_phonemes.json`. Only samples with Levenshtein ratio ≥ 0.75 pass; target 60 medium + 20 long. Hit the target in 12,180 scans (10 s median duration, median ratio 0.95, 2,129 rejected for ratio below threshold + 29 for decode failure).
- **User recordings (2)**: imran_23 + ikhlas_2_3 copied from v1.

Baseline at the time (3-repeat streaming, then-shipped v4-tlog phoneme ONNX):
- Per-run correct [204, 207, 207] / 256
- **Median recall 83.4%**, **precision 63.5%**, **SeqAcc 44.1%**
- Stable-pass 187, flaky 35, stable-fail 34

Compared to v1's 80.9% median recall, v3 shows the shipped pipeline at ~same headline recall but with **ten times the statistical power** per metric (σ of "correct" across the 3 runs is 1.7 samples on v3 vs 0.7 samples on v1, but in relative terms that's ±0.7pp vs ±1.3pp). This means a 3pp streaming improvement is now cleanly visible above noise, where on v1 it was indistinguishable from per-run jitter. Future attempts that were rejected as noise on v1 can be re-measured on v3; narrow tracker experiments (silence-flush final emission, late-verse stitching) now have a realistic path to acceptance.

Tracker raw results JSON: `web/frontend/test/v3-baseline-stability.json`. Both `stability-report.ts` and `benchmark/runner.py` already accept `--corpus=test_corpus_v3` without code changes.

**2026-04-22 — v7 streaming-aug training, falsified** (scaffold kept at `497fc91`, checkpoint discarded)
Curriculum-style fine-tune: start from v4-tlog weights, reuse v5-robust-u6 data (v4-tlog audio was not on the volume anymore), apply streaming-like augmentation (silence prob 0.2→0.6 + range 0.4s→1.5s, shift ±200→±400ms, white_noise prob 0.3→0.5, gain range widened). 3000 steps at LR 2e-5, best `val_loss=14.01` at step 3000 (still decreasing, but training completed as configured).

Ran 3-repeat v1 stability report against the shipped pipeline with the v7 ONNX swapped in. Per-run correct [35, 37, 35] vs v4 baseline [40, 39]; median **recall 71.7% (−9.2pp)**, **precision 60.6% (−6.2pp)**, **SeqAcc 43.4% (−3.8pp)**. Stable-pass 27 (−8), flaky samples 17 (+8). Outside ONNX variance — real regression.

Hypothesized cause: the expanded silence / shift windows shifted the model's output distribution such that in-distribution samples (v5-robust-u6 training data) got noisier CTC decodes at 300ms browser-streaming chunk sizes, not cleaner. The training signal optimizes full-utterance val_loss on full-audio inputs; streaming chunks see more of the augmentation than the full 10–30s training clip does (relative to its content). Put differently: the augmentor perturbs *seconds* of silence on a clip whose content is also seconds long, but at 300ms streaming that same perturbation is a qualitatively different signal.

Shipped ONNX restored to v4-tlog; v7 checkpoint discarded (stays on `fastconformer-phoneme-training` volume for possible revisit). Scaffold code (streaming-aug flag, init-from-checkpoint, manifest-reuse) kept in `scripts/train_fastconformer_phoneme_modal.py` for future training experiments. Raw stability JSON at `web/frontend/test/streaming-attempts-2026-04-21/v7-stream-aug-v1.json`.

**Takeaway:** data-augmentation matching the inference-time distribution is not obviously CTC-safe, even when the transcript is unchanged. A streaming-aware loss (e.g. compute CTC on random sub-windows of the clip) or direct streaming inference during training would be a more faithful approach.

**2026-04-21 — three matcher/tracker attempts, all falsified** (no commit — worktrees discarded)
Three narrow attempts to close the streaming-vs-batch gap. All landed **inside** the ±3–6 sample ONNX variance envelope on 2-run v1; none shipped. Baseline: 35 stable-pass / 9 stable-fail / 9 flaky, medianRecall 80.9%, medianSeqAcc 47.2%, per-run [40, 39] correct.

1. **Rare-phoneme n-gram surah expansion (always-on)** — ported the w2v-phonemes 5-gram rarity vote into `QuranDB.retrieveCandidates` to broaden the Pass 2 surah set when Levenshtein alone put the right surah outside the top-N. Result: seqAcc +0.9pp, **recall −2.2pp, precision −1.6pp**. The extra surah candidates surfaced spans whose coincidental ratio() beat the correct verse. `retasy_024` recovered but other samples regressed in compensation.
2. **Rare-phoneme n-gram, gated to low-text-confidence paths only** — same mechanism, activated only when the primary match is weak. Ship-blocking variance: per-run correct swung **[44, 34]**. One run at 44/53 was the best observed sample count across all attempts, but seqAcc dropped −5.7pp in the median.
3. **Short-text first-match gate** — raise `FIRST_MATCH_THRESHOLD` to 0.82/0.9 when decoded phoneme text is < 15/10 chars, to suppress the "short ambiguous chunk latches onto a distant verse" failure class (retasy_024/025: 1:7 → 82:11; multi_055: 55:1 → 20:5). Result: recall −1.7pp, seqAcc −1.0pp. Target failures still failed identically — the wrong verse wins at a cycle when text is already long enough to clear the gate.

Raw per-sample JSON lives in `web/frontend/test/streaming-attempts-2026-04-21/{baseline-main,ngram-always-on,ngram-gated,short-text-gate}-v1.json`. The working hypothesis is that these nine stable-fail samples are at the **ASR quality floor** (CTC decoded phonemes that genuinely look more like the wrong verse than the right one) and cannot be fixed by matcher/tracker tuning alone. The productive next lever is training-side (v7 streaming-aug fine-tune).

**2026-04-11 — deferred emission** (commit `63774dc`)  
Auto-advanced `verse_match` messages are now held as *pending* until fresh audio produces primary word alignment on the next verse; if tracking stales, the pending emission is silently dropped with full state rollback. This prevents cascades where verse N completing triggers emission of N+1, N+2, … without audio evidence.

v1: precision **53.8% → 66.8%** (+13.0pp), SeqAcc **26.4% → 47.2%** (+20.8pp), recall **78.9% → 78.6%** (−0.3pp). Same pattern on v2 blind check. 0 stable-pass → stable-fail regressions across 5 runs.

Measurement tool: `npx tsx web/frontend/test/stability-report.ts --repeats=5 [--corpus=test_corpus_v2]` produces per-sample stability classification + JSON.

**2026-04-03 — Phase A fixes**  
Short-utterance CTC rescue, span-aware commit, acoustic-dominant override. Also widened our understanding of variance: ONNX is ±3–6 samples/run on v1 (not ±2–3 as previously assumed). Earlier one-shot 45/53 and 50/53 figures sat at the high end of that distribution; the realistic pre-deferred-emission streaming baseline was 40–44/53.

## All experiments — streaming (Python, 3s chunks)

`StreamingPipeline` feeds 3s audio segments to each model, accumulates text into `VerseTracker` for progressive matching. Mirrors the browser pattern but with larger chunks.

| Experiment | Base model | FT | Type | Size | v1 Rec | v1 Prec | v1 Seq | v1 Lat | v2 Rec | v2 Prec | v2 Seq | v2 Lat |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **tadabur-whisper-small** | FaisaI/tadabur-Whisper-Small | ✓ | arabic | 461 MB | **87%** | 58% | 42% | 3.3s | **84%** | 58% | 47% | 3.8s |
| **fastconformer-lm-fusion** | nvidia FastConformer | — | arabic | 115 MB | 82% | **66%** | **55%** | **0.8s** | 74% | **59%** | **53%** | **1.0s** |
| fastconformer-ctc-rescore | nvidia FastConformer | ✓ | arabic | 260 MB | 81% | 64% | 53% | 1.0s | 77% | 61% | 53% | 1.2s |
| fastconformer-phoneme | nvidia FastConformer | ✓ | phoneme | 436 MB | 81% | 64% | 53% | 1.0s | 77% | 61% | 53% | 1.2s |
| nvidia-fastconformer | nvidia FastConformer | — | arabic | 115 MB | 81% | 64% | 53% | 1.0s | 77% | 61% | 53% | 1.2s |
| fastconformer-nbest-bruteforce | nvidia FastConformer | — | arabic | 550 MB | 80% | 61% | 49% | 0.8s | 77% | 60% | 51% | 1.0s |
| rabah-pruned-ctc/8L-ft-fn | rabah wav2vec2-xlsr-quran | ✓ | arabic | 145 MB | 71% | 55% | 42% | 2.7s | 65% | 49% | 40% | 3.4s |
| whisper-lora | whisper-small + LoRA | ✓ | arabic | 485 MB | 64% | 40% | 19% | 5.6s | 72% | 49% | 37% | 6.3s |
| whisper-small | whisper-small | — | arabic | 461 MB | 63% | 42% | 26% | 3.8s | 53% | 33% | 21% | 6.0s |
| rabah-pruned-ctc/12L-ft-es | rabah wav2vec2-xlsr-quran | ✓ | arabic | 193 MB | 61% | 41% | 25% | 3.4s | 56% | 40% | 33% | 4.4s |
| two-stage | moonshine-tiny + wav2vec2 | ✓ | arabic | 463 MB | 47% | 23% | 13% | 3.7s | 38% | 24% | 19% | 5.8s |
| distilled-ctc | wav2vec2-base (distilled) | ✓ | arabic | 360 MB | 7% | 7% | 6% | 0.5s | 5% | 3% | 2% | 0.5s |

`tadabur-whisper-small` has the highest raw streaming recall but at 3–5× FastConformer latency. FastConformer variants dominate the speed/accuracy/size frontier. `w2v-phonemes` cannot stream — no chunked `transcribe()` path.

## All experiments — batch (Python, full-file)

Full-file transcription then single `matchVerse()` call.

| Experiment | Base model | FT | Type | Size | v1 Rec | v1 Prec | v1 Seq | v1 Lat | v2 Rec | v2 Prec | v2 Seq | v2 Lat |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **c2c-direct-mixed-tta** (Cyberistic winning entry) | nvidia FastConformer | — | arabic | **88 MB** | **100%** | **100%** | **100%** | **0.84s** | — | — | — | — |
| **c2c-direct-mixed** | nvidia FastConformer | — | arabic | **88 MB** | 98% | 98% | 98% | **0.72s** | — | — | — | — |
| **zipformer-ctc** (v3.1 base weights, our tracker, Node) | streaming Zipformer2-CTC (k2) | — | tajweed-phoneme | 73 MB | **100%** | **100%** | **100%** | 0.99s | **98%** | **98%** | **98%** | 1.04s |
| **w2v-phonemes/large** | hetchyy/r7 | — | phoneme | 970 MB | **100%** | **100%** | **100%** | 15.2s | **95%** | **95%** | **95%** | 30.4s |
| **w2v-phonemes/base** | hetchyy/r15_95m | — | phoneme | 388 MB | — | — | — | — | — | — | — | — |
| **w2v-phonemes/base-local-int8** | hetchyy/r15_95m | — | phoneme | 118 MB | — | — | — | — | — | — | — | — |
| **fastconformer-lm-fusion** | nvidia FastConformer | — | arabic | 115 MB | 95% | 96% | **94%** | 7.2s | **95%** | **95%** | **95%** | 6.6s |
| **nvidia-fastconformer** | nvidia FastConformer | — | arabic | 115 MB | 95% | 95% | 92% | **0.7s** | 93% | 90% | 86% | **0.9s** |
| fastconformer-phoneme | nvidia FastConformer | ✓ | phoneme | 436 MB | 95% | 95% | 92% | 7.9s | 93% | 90% | 86% | 7.1s |
| fastconformer-ctc-rescore | nvidia FastConformer | ✓ | arabic | 260 MB | 95% | 95% | 92% | 7.3s | 93% | 90% | 86% | 6.7s |
| fastconformer-nbest-bruteforce | nvidia FastConformer | — | arabic | 550 MB | 95% | 95% | 92% | 0.6s | 93% | 90% | 86% | 0.9s |
| tadabur-whisper-small | FaisaI/tadabur-Whisper-Small | ✓ | arabic | 461 MB | 86% | 88% | 79% | 1.3s | 87% | 87% | 81% | 1.4s |
| whisper-lora | whisper-small + LoRA | ✓ | arabic | 485 MB | 82% | 86% | 77% | 2.3s | 81% | 84% | 79% | 2.1s |
| rabah-pruned-ctc/8L-ft-fn | rabah wav2vec2-xlsr-quran | ✓ | arabic | 145 MB | 75% | 75% | 74% | 3.7s | 77% | 77% | 77% | 3.9s |
| whisper-small | whisper-small | — | arabic | 461 MB | 73% | 76% | 68% | 1.0s | 50% | 50% | 47% | 1.1s |
| two-stage | moonshine-tiny + wav2vec2 | ✓ | arabic | 463 MB | 69% | 69% | 66% | 2.3s | 56% | 56% | 51% | 2.2s |
| rabah-pruned-ctc/12L-ft-es | rabah wav2vec2-xlsr-quran | ✓ | arabic | 193 MB | 63% | 63% | 60% | 5.3s | 67% | 67% | 67% | 5.2s |
| rabah-pruned-ctc/8L-ft-es | rabah wav2vec2-xlsr-quran | ✓ | arabic | 145 MB | 55% | 55% | 55% | 4.0s | 47% | 47% | 47% | 4.0s |
| rabah-pruned-ctc/6L-ft-es | rabah wav2vec2-xlsr-quran | ✓ | arabic | 121 MB | 54% | 54% | 51% | 3.3s | 56% | 56% | 56% | 3.1s |
| distilled-ctc | wav2vec2-base (distilled) | ✓ | arabic | 360 MB | 30% | 29% | 26% | 0.6s | 26% | 26% | 26% | 6.2s |

### Phoneme matcher: strategy comparison

Historical ONNX phoneme model via Python `predict()`, swapping out the matching strategy:

| Matching strategy | v1 Recall | v1 SeqAcc | v2 Recall | v2 SeqAcc |
|---|---|---|---|---|
| Simple `ratio()` | 79% | 75% | 87% | 86% |
| **Multi-pass (fragment + span)** | **90%** | **87%** | **87%** | **84%** |

The multi-pass matcher (ported from the browser's `quran-db.ts` — fragment scoring, short-query boost, bismillah stripping, multi-verse spans) adds +11pp v1 recall at zero decode cost. Matching quality was the bottleneck, not decoding.

### 0% recall — broken or inapplicable

| Experiment | Base model | Type | Size | Reason |
|---|---|---|---|---|
| contrastive | HuBERT + AraBERT | embedding | 900 MB | English encoder → useless Arabic features |
| contrastive-v2 | HuBERT + AraBERT | embedding | 367 MB | Same fundamental issue as v1 |
| embedding-search | HuBERT + FAISS | embedding | 397 MB | HuBERT encodes speaker identity, not content |
| ctc-alignment | wav2vec2-xlsr-53-arabic | arabic | 1.2 GB | `transcribe()` path broken; runner uses it |
| tarteel-whisper-base | tarteel-ai/whisper-base-ar-quran | arabic | 290 MB | Model loading errors on all samples |
| streaming-asr | mlx-whisper base | arabic | 145 MB | Needs mlx-whisper (not installed) |
| two-stage-faster-whisper-pruned | faster-whisper + pruned CTC | arabic | — | Needs faster-whisper (not installed) |

## Deep dive: Rabah pruned CTC variants

Layer pruning + optional fine-tuning applied to `rabah2026/wav2vec2-large-xlsr-53-arabic-quran-v_final`.

| Variant | Layers | Pruning | FT | v1 Rec | v1 Seq | v2 Rec | v2 Seq | Lat | Size |
|---|---|---|---|---|---|---|---|---|---|
| 8L-ft-fn-int8 | 8 | first_n | ✓ | **75%** | **74%** | **77%** | **77%** | 3.7s | 145 MB |
| 12L-ft-es-int8 | 12 | evenly_spaced | ✓ | 63% | 60% | 67% | 67% | 5.3s | 193 MB |
| 12L-int8 | 12 | evenly_spaced | — | 62% | 62% | 51% | 51% | 5.5s | 193 MB |
| 8L-ft-es-int8 | 8 | evenly_spaced | ✓ | 55% | 55% | 47% | 47% | 4.0s | 145 MB |
| 6L-ft-es-int8 | 6 | evenly_spaced | ✓ | 54% | 51% | 56% | 56% | 3.3s | 121 MB |
| 8L-int8 | 8 | evenly_spaced | — | 2% | 2% | 0% | 0% | 4.0s | 145 MB |
| 6L-int8 | 6 | evenly_spaced | — | 0% | 0% | 0% | 0% | 3.2s | 121 MB |

`first_n` pruning (keep layers 0–7) beats `evenly_spaced` by ~20pp at the same layer count. Fine-tuning the CTC head is non-optional — unfinetuned pruned models score near 0%.

## Deep dive: TLOG data-mix fine-tunes

Fine-tuning the phoneme CTC head with varying amounts of TLOG (phone-recorded recitation).

| Model | TLOG | Filter | Streaming v1 | Streaming v2 | Notes |
|---|---|---|---|---|---|
| **v4-tlog** (shipped) | ~18K (5/verse) | 0.3 | **45/53 (85%)** † | **32/43 (74%)** † | best checkpoint |
| v5-robust-u6 | 0 (no TLOG) | — | 43/53 (81%) | 33/43 (77%) | removing TLOG also hurts |
| v4-tlog-heavy | ~53K (15/verse) | 0.3 | 36–38/53 (70%) | 25/43 (58%) | regression |
| v4-tlog-hq | ~74K (30/verse) | 0.5 | 29–31/53 (56%) | 23–24/43 (54%) | bigger regression |
| v6-augmented | ~29K (5/verse) | none | 26/53 (49%) NS | — | +MUSAN +teacher relabel, worst |

† v4-tlog figures are single-run; the post-Phase-A median was 40–44/53 v1.

**Takeaways:** ~18K TLOG at filter=0.3 is a genuine sweet spot. Scaling up volume regresses; removing TLOG also regresses; combining multiple data-side changes (v6) makes attribution impossible. **Rule: one data change per training run.**

**v6-augmented failure detail:** unfiltered TLOG (29K) + teacher pseudo-labels on 75% of samples + MUSAN noise aug, all together. Training metrics looked healthy (val_loss=58.39 at step 6500) but downstream accuracy collapsed. Unfiltered TLOG alone contains ~38% bad samples per the quality filter; the teacher relabeler added an unknown additional error rate on the rest. Streaming export also crashed with an ONNX mutex error (NeMo <2.7 compat).

## Zipformer2-CTC (streaming phoneme model + fine-tunes)

Base weights are `Quran-Lab/zipformer_p-arabic-v3` v3.1 (HF; NPL-1.2 — share-alike, non-commercial; see [NOTICE.md](../NOTICE.md)), int8 via `quantize_dynamic(MatMul QInt8)`, streaming contract T=61/hop=48/left 256, trained by the authors with chunk mix 8/16/24 frames (which is why we fine-tune at `--chunk-size 8,16,24`). Our tracker/harness and all fine-tunes are ours. Checkpoints live at volume path `/vol/reference/`; local copy `data/zipformer/reference/`.

### Tracker eval (our harness; median; scores identical across repeats)

| Model | Size | v1 (53) | v2 (43) | v3 (256) Rec/Prec/Seq | qlab (583) | qlab EA / nufais / tlog | lat v3 / qlab |
|---|---|---|---|---|---|---|---|
| **v3.1 fp32** | 251 MB | **53/53** | 42/43 | 96.9 / 96.7 / 96.5 **(247/256)** | **571/583** (97.9%) | 184/184, 193/200, 194/199 | 0.93 s / 0.58 s |
| v3.1 int8 (vendored) | 69 MB | **53/53** | 42/43 | same 247/256 | same 571/583 | same split | 0.75 s / 0.48 s |
| v3 fp32 | 251 MB | 52/53 | **43/43** | same 247/256 | same 571/583 | same split | 0.88 s / 0.58 s |
| v3 int8 | 69 MB | 52/53 | **43/43** | same 247/256 | same 571/583 | same split | 0.75 s / 0.48 s |
| ft-v31 fp32 | 248 MB | 45/53 | 39/43 | 92.3 / 94.5 / 90.2 **(231/256)** | 566/583 (97.1%) | 184/184, 194/200, 188/199 | 0.85 s / 0.56 s |
| ft-v31 int8 | 66 MB | 45/53 | 39/43 | same 231/256 | same 566/583 | same split | 0.68 s / 0.44 s |
| ft-v31 fp32 + `ALLOW_GAPS=1` | 248 MB | 45/53 | 39/43 | 92.3 / 94.5 / 90.2 **(231/256)** | 566/583 (97.1%) | 184/184, 194/200, 188/199 | 1.48 s / — |
| ft-gentle fp32 (1 ep, lr 0.001) | 248 MB | 44/53 | — | 90.5 / 94.5 / 87.9 **(225/256)** | 571/583 (97.9%) | 184/184, 194/200, 193/199 | 0.87 s / 0.55 s |
| interp-ftv31-a0.5 fp32 | 248 MB | 52/53 | — | 96.9 / 96.9 / 96.9 **(248/256)** | **572/583** (98.1%) | 184/184, 194/200, 194/199 | 1.60 s / 1.13 s |
| interp-ftv31-a0.25 fp32 | 248 MB | **53/53** | — | 96.8 / 96.7 / 96.1 **(246/256)** | 571/583 (97.9%) | 184/184, 193/200, 194/199 | 1.81 s / 0.97 s |
| **interp-gentle-a0.5 fp32** | 248 MB | **53/53** | — | 96.9 / 96.9 / 96.9 **(248/256)** | **572/583** (98.1%) | 184/184, 194/200, 194/199 | 0.87 s / 0.91 s |
| **interp-gentle-a0.5 int8** | 66 MB | **53/53** | **43/43** | same 248/256 | same 572/583 | same split | 0.67 s / 0.44 s |
| ft-multi ep1 fp32 | 248 MB | 46/53 | 35/43 | 87.8 / 88.7 / 87.1 **(223/256)** | 558/583 (95.7%) | 184/184, 194/200, 180/199 | 0.84 s / 0.54 s |
| ft-multi ep2 fp32 | 248 MB | 42/53 | — | 93.2 / 93.8 / 93.0 **(238/256)** | 551/583 (94.5%) | 183/184, 194/200, 174/199 | 0.84 s / 0.55 s |
| interp-multi-a0.5 fp32 | 248 MB | **53/53** | 42/43 | 96.9 / 96.9 / 96.9 **(248/256)** | **572/583** (98.1%) | 184/184, 194/200, 194/199 | 0.83 s / 0.54 s |
| ft-multi-full ep1 fp32 | 248 MB | 45/53 | 39/43 | 90.6 / 93.0 / 89.1 **(228/256)** | 563/583 (96.6%) | 184/184, 194/200, 185/199 | 0.86 s / 0.55 s |
| ft-multi-full ep2 fp32 | 248 MB | 47/53 | 41/43 | 90.0 / 92.2 / 87.9 **(225/256)** | 551/583 (94.5%) | 184/184, 194/200, 173/199 | 0.83 s / 0.55 s |
| interp-mf1-a0.25 fp32 | 248 MB | **53/53** | **43/43** | 96.9 / 96.7 / 96.5 **(247/256)** | 571/583 (97.9%) | 184/184, 193/200, 194/199 | 0.82 s / 0.53 s |
| interp-mf1-a0.5 fp32 | 248 MB | **53/53** | **43/43** | 96.9 / 96.9 / 96.9 **(248/256)** | **572/583** (98.1%) | 184/184, 194/200, 194/199 | 0.82 s / 0.53 s |
| interp-mf1-a0.75 fp32 | 248 MB | 52/53 | **43/43** | 96.5 / 96.5 / 96.5 **(247/256)** | **572/583** (98.1%) | 184/184, 194/200, 194/199 | 0.82 s / 0.53 s |
| interp-mf2-a0.5 fp32 | 248 MB | **53/53** | **43/43** | 96.9 / 96.9 / 96.9 **(248/256)** | **572/583** (98.1%) | 184/184, 194/200, 194/199 | 0.82 s / 0.53 s |

v3 vs v3.1 swap one crowd clip: v3 misses `retasy_012` (114:2→114:3); v3.1 misses `retasy_v2_012` (1:3→55:1). v3-corpus and qlab miss *sets* are identical across all four ONNX files. Repeats never differed in correct-count (latency only). Grid: v3.1 fp32+int8 all corpora ×3; v3 fp32 on v3/qlab ×3; v3 fp32 v1/v2 and v3 int8 all ×1. ft-v31 fp32+int8 all corpora ×3 (scores identical across repeats).

**v3 multi vs single** (21 clips with `expected_verses` length > 1; 235 single):

| model | multi (21) | single (235) |
|---|---|---|
| v3.1 fp32 | **21/21** | 226/235 |
| ft-v31 ep5 | 9/21 | 222/235 |
| ft-gentle ep1 | 3/21 | 222/235 |
| ft-multi ep1 | 17/21 | 206/235 |
| ft-multi ep2 | 19/21 | 219/235 |
| interp-multi-a0.5 | **21/21** | **227/235** |
| **interp-gentle-a0.5** | **21/21** | **227/235** |
| ft-multi-full ep1 | 11/21 | 217/235 |
| ft-multi-full ep2 | 10/21 | 215/235 |
| interp-mf1-a0.25 | **21/21** | 226/235 |
| interp-mf1-a0.5 | **21/21** | **227/235** |
| interp-mf1-a0.75 | **21/21** | 226/235 |
| interp-mf2-a0.5 | **21/21** | **227/235** |

**PER (ONNX streaming greedy, v3.1 fp32, qlab):** overall **5.56%** (exact 49.1%); everyayah_heldout 2.39%, qul_alnufais 7.78%, tlog_holdout 7.06%. Not comparable 1:1 to the authors' published PER (different decode, gold is `ordered_quran_phonemes.json` by surah:ayah, torchaudio kaldi fbank). Wrapper: `experiments/zipformer-ctc/reference_tools/per_onnx_wrapper.py`.

**PER (ft-v31 fp32, same wrapper):** overall **4.37%** (exact 61.6%); EA 1.69%, nufais 4.59%, tlog 8.18%. int8 4.35%. Acoustic PER improved vs v3.1; tracker SeqAcc did not.

**PER (ft-gentle fp32):** overall **4.32%** (exact 66.6%); EA 1.47%, nufais 4.55%, tlog 8.39%.

**PER (interp-gentle-a0.5 fp32):** overall **4.09%** (exact 68.4%); EA 1.48%, nufais 6.57%, tlog 4.35%. Best PER of the three and the first to clear the tracker bar.

**PER (interp-mf1-a0.5 fp32):** overall **4.27%** (exact 66.2%); EA 1.59%, nufais 6.86%, tlog 4.45%. Same tracker counts as interp-gentle-a0.5; acoustics slightly worse.

Reproduction:

```bash
cd .worktrees/sota-tilawa
ZIPFORMER_DATA_DIR=/Users/rock/ai/projects/offline-tarteel/data/zipformer \
ZIPFORMER_MODEL=/Users/rock/ai/projects/offline-tarteel/data/zipformer/reference/zipformer_p_arabic_v3.1.onnx \
ZIPFORMER_CORPUS=/Users/rock/ai/projects/offline-tarteel/data/zipformer/quran.json \
ZIPFORMER_ORT_DIR=/Users/rock/ai/projects/offline-tarteel/web/frontend/node_modules \
/Users/rock/ai/projects/offline-tarteel/.venv/bin/python -m benchmark.runner --experiment zipformer-ctc --corpus test_corpus_v3
# full grid: .venv/bin/python experiments/zipformer-ctc/eval_reference_grid.py
# fetch: modal run scripts/fetch_reference_zipformer_modal.py
```

Raw JSON: `benchmark/results/2026-09-14_16*.json` / `_17*.json` / `_18*.json`; ledger `benchmark/results/qlab_v3_eval_ledger.json`; PER `benchmark/results/v31_fp32_qlab_per.json`.

### Fine-tune data mix (staged 2026-09-15)

Volume `zipformer-ctc-training` `/manifests/`. All five `*_cuts_fbank.jsonl.gz` exist. OOV 0. Speed perturb ×3 at train time.

| source | clips | hours | notes |
|---|---|---|---|
| everyayah | 128,204 | 358.0 | train+validation, 1 skip_short |
| qua | 239,975 | 747.2 | fbank 11/12 shards ≈ 685 h perturbed; shard 11 hung twice, skipped |
| iqra | 16,070 | 21.5 | 55,321 low_match dropped (MSA+Quran mix) |
| retasy | 357 | 0.36 | `correct` only; icefall drops ~3 s clips labelled 2:255 |
| tlog | 41,083 | 100.0 | qlab tlog_holdout ids excluded |
| **total (raw)** | **425,689** | **1,227** | ~2,700 h with ×3 perturb |

### ft-v31 (NOT PROMOTED)

Fine-tune from Quran-Lab v3.1 `.pt` (`--init-from`, inverse-permute CTC blank 250→0, `--pos-dim 192`). Config: 5 epochs, `--max-duration 1200`, `--avg 3` (epochs 2–5), `--base-lr 0.005`, `--warmup-batches 500`, chunks `8,16,24`, left `128,256`. GPU `H100:4` DDP after `201f8d1` (worker `world_size=torch.cuda.device_count()`). Train app `ap-6In0UqeDBVXxeLVFXw1aUe` (cancelled first client `ap-a4vuTLIWmrU9rOEoVMrX3g` had `world_size=1` because `ZIPFORMER_GPU` was unset on the worker). Volume `/vol/exp/ft-v31`, export `/vol/exports/ft-v31` (`model.onnx` 259.6 MB, `model.int8.onnx` 69.2 MB, T=61 hop=48, `io_diff=[]`). Trained on the 11/12 QUA fbank merge (shard 11 later finished; this run did not see it).

CTC first-batch 0.6411 (pretrained, <1.0). `metrics.jsonl`: ep1 train 0.1066 / valid 0.0884; ep5 train 0.0796 / valid 0.0324. Wall 9536 s (~2.65 h). Max GPU mem ~18 GB / 80 GB. Epoch-3 valid 0.0342 is worse than ep4/5 — no extra `--epoch 3 --avg 1` export.

Promotion bar (strict): qlab ≥ 572 AND v1 = 53 AND v3 ≥ 247 vs v3.1 baseline 571 / 53 / 247. **NOT PROMOTED** — qlab 566/583, v1 45/53, v3 231/256. tlog_holdout 188/199 vs 194; nufais 194/200 vs 193 (the only slice that improved). int8 matches fp32 scores, ~20% faster.

v1/v3 regressions are mostly multi-ayah truncation (first 1–2 ayahs only). qlab: −6 tlog_holdout, +1 nufais (`qul_alnufais__37_43`). ONNX PER **4.37%** (EA 1.69 / nufais 4.59 / tlog 8.18; exact 61.6%) vs v3.1 **5.56%** — acoustics improved, tracker SeqAcc did not.

Epoch sweep (fp32 `--avg 1` except ep5 avg 3; one deterministic harness run):

| ckpt | v1 | v3 |
|---|---|---|
| v3.1 init | **53/53** | **247/256** |
| ep1 avg1 | 46/53 | 225/256 |
| ep2 avg1 | 44/53 | 214/256 |
| ep5 avg3 | 45/53 | 231/256 |

Most of the damage is epoch 1; epoch 2 is the trough; epoch 5 recovers some v3 but not v1. Not a clean “more FT → worse” slope.

**Failure mode (ep5, 5 v3 multi clips):** mixed acoustic + matcher, acoustic first. CTC transcript **drops short connecting ayahs** (Fatiha 1:3/1:4/1:6 absent; Fil 105:2/105:4 absent; 25:66 head absent). Later ayahs that *are* in the transcript often still get tracker tallies (`ok` full), but (1) `MIN_WORD_FRACTION=0.5` rejects partials (109:4 ok+unsure=2/5 words; 25:66 unsure=1/4) and (2) `predict()` `_contiguous_head` stops at the first gap so SeqAcc looks like prefix truncation (1:1–7→1:1–2 even though 1:5 and 1:7 were emitted). Needs B1 multi-ayah windows and/or tracker re-tune (`okDistance`/`unsureDistance`/word-fraction + don't truncate at holes).

Raw JSON: `benchmark/results/2026-09-15_18*.json` / `_19*.json`; ledger `benchmark/results/ft_v31_eval_ledger.json`; PER `benchmark/results/ft_v31_fp32_qlab_per.json`. Epoch-1/2: `2026-09-15_194659.json` (v1), `_195041.json` (v3), `_195131.json` (v1), `_195511.json` (v3). Export apps `ap-6aM9VJL2SXHIT54dfZMBMr` (ep1), `ap-MAxnoFmA6Ahh017w6RDpeu` (ep2).

### E2 gentle fine-tune + interpolation (PROMOTED: interp-gentle-a0.5)

Hypothesis: lr 0.005 over-fit in epoch 1; a 1-ep lr 0.001 FT, or a WiSE-FT blend with the v3.1 init, keeps PER gain without dropping multi-ayah. `--export-interp INIT_PT:FT_PT:ALPHA` inverse-permutes the reference CTC head to icefall blank=0 *before* `alpha*ft+(1-alpha)*init` (`998f34c`). ft-v31 blends used **epoch-5.pt** (not avg-3).

`ft-gentle` (`ap-1tiu0kP4AiIsn2sxQzLooc`, H100:4, 1 ep, lr 0.001, warmup 1000, avg 1): train 0.1108 / valid 0.0832 / 1662 s — **not** gentler on the tracker (v1 44/53, v3 225/256, same prefix-truncation as ft-v31 ep1). **interp-gentle-a0.5** (init ⊕ ft-gentle ep1, `ap-qj8yZzLGTgg9CZnexqPIka`) is **53 / 248 / 572** (EA 184, nufais 194, tlog 194): v3 gained `tlog_m000_100_001`, qlab gained `qul_alnufais__37_43`. interp-ftv31-a0.5 hits 572/248 but v1 52 (`multi_036_001_005` 36:1–5→36:2–5); a0.25 keeps v1 53 but v3 246 (`ea_alafasy_multi_044_001_005`). **PROMOTED** vs bar qlab≥572 AND v1=53 AND v3≥247. Shipping artefact is dynamic-int8 MatMul QInt8: **259,593,848 B fp32 / 69,245,985 B int8**; int8 matches fp32 on v1/v3/qlab and is **43/43** on v2 (recovers `retasy_v2_012`). Raw: `2026-09-15_210522.json`–`_220939.json`; int8 `_224627` (v1) `_224703` (v2) `_225000` (v3) `_225419` (qlab); PER `interp_gentle_a0.5_qlab_per.json`, `ft_gentle_qlab_per.json`.

### E1 multi-ayah windows (NOT PROMOTED; interp-gentle-a0.5 stays)

Hypothesis: isolated-ayah FT forgot short connector ayahs; synthetic 2–4 ayah windows (`everyayah_multi`) recover multi-verse SeqAcc. Built 40,000 windows (179.4 h raw, 0 OOV, n_ayahs 2/3/4 = 22198/11903/5899, mean 7.02 phonemes/s) from EveryAyah via lhotse `append` + 0–800 ms silence (noise mix skipped — inode cap). MixedCut left window text on ayah-1 (`f9070dc` put it on the first non-padding track; `888f31a` flattened fbank to MonoCut so icefall collates whole-cut text against `cut.num_frames` instead of the first-ayah interval). Mix `everyayah,everyayah_multi,tlog,iqra` (no qua), lr 0.002, 2 ep, H100:4. Loss: ep1 train 0.160 / valid 0.153 (986 s); ep2 0.134 / 0.073 (898 s).

Multi windows **do** repair raw FT multi-ayah (ft-v31 9/21 → ft-multi ep2 **19/21**) but the studio-heavy reduced mix wrecks the phone slice (tlog 194 → 174; qlab 551). ep1 is milder on tlog (180) and v1 (46) but weaker on v3 (223, multi 17/21). **interp-multi-a0.5** (0.5·v3.1 ⊕ 0.5·ft-multi ep1, sha `0cf247b3`) is 53/42/248/572 (multi 21/21, single 227/235, qlab 184/194/194) — ties interp-gentle-a0.5 on v1/v3/qlab, loses v2 42 vs 43. **interp-gentle-a0.5 remains the promoted candidate.** Next: multi windows + full mix + gentle LR, then blend. Raw: `2026-09-16_064849.json` (ep2 v1, `29644ea2`), `_065228` (v3), `_065748` (qlab); `_071046` (ep1 v1, `c8d7cd51`), `_071131` (v2), `_071510` (v3), `_072026` (qlab); interp `_070025` (v1), `_070110` (v2), `_070447` (v3), `_071003` (qlab). Report `.superpowers/sdd/plan-sota-tilawa/exp-E1-report.md`.

### E4 multi windows + full mix + gentle FT (NOT PROMOTED; interp-gentle-a0.5 stays)

Hypothesis: E1's tlog collapse was the missing qua/full mix, not the windows; combine windows (mux weight 2.5 → ~27% of 5,061 perturbed hours) with E2's gentle schedule, then blend. `ft-multi-full` (`ap-ZRkTcdmpX5RdZOtW92dA9N`, H100:4, 2 ep, lr 0.001, warmup 1000, `--source-weights everyayah_multi=2.5`): ep1 train 0.120 / valid 0.081 (1896 s), ep2 0.109 / 0.039 (1777 s). Raw FT still dies — ep1 45/39/228(multi **11/21**)/563(tlog 185); ep2 47/41/225(10/21)/551(tlog 173). Windows only repaired multi when they dominated a *small* mix (E1, no qua); diluted by qua they do not. **interp-mf1-a0.5** and **interp-mf2-a0.5** (sha `7bfc65f0` / `fdee4541`) are 53/43/248/572 with the **same miss set** as interp-gentle-a0.5; PER 4.27% vs 4.09%. a0.25 drops the two E2-gain clips (`tlog_m000_100_001`, `qul_alnufais__37_43`); a0.75 drops `retasy_016` on v1. **NOT PROMOTED** (qlab tie, not strictly greater). Raw: `2026-09-16_090027.json`–`_100159.json`; PER `interp_mf1_a0.5_qlab_per.json`. Report `.superpowers/sdd/plan-sota-tilawa/exp-E4-report.md`.

### E3 tracker re-tune (diagnostic, NOT PROMOTED)

Matcher-only probe on ft-v31 ep5 avg-3 (`sha256` `7c7f0f4d…`). v2 grid of 12 configs: only `ZIPFORMER_ALLOW_GAPS=1` appeared to move v2 (40/43), via `_contiguous_head` inventing a corpus-short hole. Fix round 1 requires both neighbours already emitted and `ok+unsure≥1`, and Python skips a hole only if that ayah is already in `verses`. After that, ft-v31+gaps = default: **45/53, 39/43, 231/256, 566/583**. The three “recovered” clips (`multi_036_001_005`, `ea_multi_056_001_004`, `ea_alafasy_multi_095_001_005`) revert. Reference + gaps stays **53/247/571**. Not a promotion candidate — the loss is acoustic, not a prefix-fill matcher bug. Raw: grid `2026-09-15_205959.json`–`_211248.json`; honest re-verify `_220245` (v1), `_222224` (v2), `_220911` (v3), ref `_221000`/`_221313`/`_222037`.

### Miss adjudication with Gemini 3.1 Pro

Blind + A/B informed listen of the 22 v3.1/interp-gentle misses (21 v3+qlab + `retasy_v2_012`). `gemini-3.1-pro-preview` / `gemini-pro-latest` return free-tier limit 0 on the AI Studio key; ran `generateContent` on Flash (`gemini-3.5-flash`, `gemini-3.6-flash`, `gemini-3.1-flash-lite`), temperature 0, JSON schema. Control: 3/3 unique v3 shorts (`ea_alafasy_056058` 56:58, `ea_husary_081008` 81:8, `ea_husary_106003` 106:3) identified blindly. Script `benchmark/adjudicate_gemini.py`; raw `benchmark/results/gemini_adjudication_2026-09-16.json`.

| id | corpus | expected | predicted | text-similarity | Gemini blind surah:ayah | Gemini informed verdict | class | notes |
|---|---|---|---|---|---|---|---|---|
| qul_alnufais__21_38 | qlab | 21:38 | 10:48 | 1.000 | 67:25 | both-identical | IDENTICAL_TEXT | same wording as 10:48 / 21:38 / 67:25 |
| qul_alnufais__37_43 | qlab | 37:43 | 52:17 | 0.632 | 56:12 | A | LABEL_OK_MODEL_WRONG | audio is 37:43/56:12 «في جنات النعيم»; 52:17 has extra words. v3.1 only; gentle recovered |
| qul_alnufais__55_30 | qlab | 55:30 | 55:13 | 1.000 | 55:13 | both-identical | IDENTICAL_TEXT | Ar-Rahman refrain |
| qul_alnufais__55_40 | qlab | 55:40 | 55:13 | 1.000 | 55:13 | both-identical | IDENTICAL_TEXT | Ar-Rahman refrain |
| qul_alnufais__56_12 | qlab | 56:12 | 37:43 | 1.000 | 56:12 | both-identical | IDENTICAL_TEXT | «في جنات النعيم» |
| qul_alnufais__83_13 | qlab | 83:13 | 68:15 | 1.000 | 68:15 | both-identical | IDENTICAL_TEXT | |
| qul_alnufais__8_51 | qlab | 8:51 | 3:182 | 1.000 | 3:182 | both-identical | IDENTICAL_TEXT | |
| tlog_holdout__37_176_undefined_Bc1Te4g | qlab | 37:176 | 26:204 | 1.000 | 37:176 | both-identical | IDENTICAL_TEXT | |
| tlog_holdout__38_73_1028803212 | qlab | 38:73 | 15:30 | 1.000 | 15:30 | both-identical | IDENTICAL_TEXT | |
| tlog_holdout__38_79_1059280208 | qlab | 38:79 | 15:36 | 1.000 | 15:36 | both-identical | IDENTICAL_TEXT | |
| tlog_holdout__70_29_3740714225 | qlab | 70:29 | 23:5 | 1.000 | 23:5 | both-identical | IDENTICAL_TEXT | |
| tlog_holdout__77_45_6585124791 | qlab | 77:45 | 77:15 | 1.000 | 77:15 | both-identical | IDENTICAL_TEXT | |
| ea_alafasy_030001 | v3 | 30:1 | 2:1 | 1.000 | 2:1 | both-identical | IDENTICAL_TEXT | muqattaʿat الم |
| ea_alafasy_055053 | v3 | 55:53 | 55:13 | 1.000 | 55:13 | both-identical | IDENTICAL_TEXT | Ar-Rahman refrain |
| ea_alafasy_081019 | v3 | 81:19 | 69:40 | 1.000 | 81:19 | both-identical | IDENTICAL_TEXT | |
| ea_husary_026122 | v3 | 26:122 | 26:9 | 1.000 | 26:9 | both-identical | IDENTICAL_TEXT | |
| ea_husary_037082 | v3 | 37:82 | 26:66 | 1.000 | 37:82 | both-identical | IDENTICAL_TEXT | |
| tlog_m000_100_001 | v3 | 100:1 | 100:1–2 | 0.835 | 100:1–2 | B | LABEL_WRONG | clip continues into 100:2; gold is truncated. v3.1 only (gentle matches truncated gold) |
| tlog_m008_107_001 | v3 | 107:1 | 106:4 | 0.304 | 106:4 | B | LABEL_WRONG | audio is 106:4, not 107:1 |
| tlog_m043_010_043 | v3 | 10:43 | 10:42 | 0.804 | 10:42 | B | LABEL_WRONG | audio is 10:42 |
| tlog_m044_010_043 | v3 | 10:43 | 10:42 | 0.804 | 10:42 | B | LABEL_WRONG | audio is 10:42 (same mislabel as m043) |
| retasy_v2_012 | v2 | 1:3 | 55:1 | 0.622 | 1:3 | A | LABEL_OK_MODEL_WRONG | audio is 1:3 «الرحمن الرحيم»; 55:1 adds basmala. v3.1 only; gentle recovered |

**16 IDENTICAL_TEXT / 4 LABEL_WRONG / 2 LABEL_OK_MODEL_WRONG / 0 BAD_CLIP / 0 UNCLEAR.** Duplicate-ayah collisions are the bulk (qlab 11/12, v3 5/9): ASR cannot pick among textually identical copies, and Gemini's own blind ID hops between those copies too. The four LABEL_WRONG are all v3 tlog gold errors (100:1 missing 100:2; 107:1 is 106:4; two clips labelled 10:43 are 10:42). The only genuine v3.1 errors are `qul_alnufais__37_43` (37:43 → 52:17) and `retasy_v2_012` (1:3 → 55:1); interp-gentle-a0.5 already recovers both. **Ceiling on current labels:** v3.1 is one qlab + one v2 miss behind the unsolvable-duplicate floor; interp-gentle-a0.5 is **at** that floor (**248/256**, **572/583**). Further SeqAcc requires a duplicate-ayah tie-break and/or relabeling those four tlog clips — not more fine-tuning.

### Oracle check of the remaining misses (Gemini 3.1 Pro, 2026-09-16)

The 21 clips still missed by the v3.1 base and/or interp-gentle-a0.5 were independently transcribed with `gemini-3.1-pro-preview` (raw transcripts scored against `quran.json`; see `artifacts/gemini_oracle/`). 14 of 16 confusable pairs are verbatim-identical text — 55:53/55:30/55:40↔55:13, 81:19↔69:40, 37:82↔26:66, 26:122↔26:9, 37:43↔56:12, 21:38↔10:48, 83:13↔68:15, 37:176↔26:204, 70:29↔23:5, 77:45↔77:15, 38:73↔15:30, 38:79↔15:36 — so the ID is undecidable from audio alone. Two `test_corpus_v3` labels are wrong: `tlog_m043_010_043` and `tlog_m044_010_043` are 10:42 (the model's prediction); `tlog_m008_107_001` is 106:4 followed by 107:1 (label incomplete); `qul_alnufais__8_51` uses بظلام (3:182 wording) — a probable label/recitation variant. Zero genuine model errors remain on distinguishable unique text, so the effective ceiling is **251/256** on v3 (not 249) and **≈573–574/583** on q-lab without context priors. Remaining SeqAcc has to come from previous-ayah / surah continuity, which the tracker's `hint` mechanism already supports. Manifests are unchanged; gold issues are in `benchmark/test_corpus_v3/KNOWN_LABEL_ISSUES.md`.

### Phase 0 (Beyond-QLab v3): leak audit, TLOG filter, v3 control (2026-09-30)

**Held-out reciters were in the FT mix.** `greentechapps/everyayah_curated_1s_20s` was re-split after q-lab was cut (q-lab wavs are `test-000NN-of-00013`; the repo now has 31 test shards whose reciters are Husary_128kbps / Saood_ash-Shuraym / Ahmed_Neana / Mohammad_al_Tablaway). The three `everyayah_heldout` reciters (Sahl_Yassin, Akram_AlAlaqimy, Muhsin_Al_Qasim — 12/12 sampled clips matched by duration on everyayah.com) now sit in train/validation, which staging ingested. `everyayah_cuts_fbank`: 44,376 cuts / 120 h (×3 perturb) from them; `everyayah_multi`: 66.7 h more; 11/12 sampled benchmark clips are the same recording in training (Δdur ≤ 0.05 s). **Every FT/interp EA-slice number above (ft-v31, ft-gentle, interp-*, E1, E4) is leaked.** Also: 101/127 `test_corpus_v3` Alafasy ayahs are in the FT mix, so v3-corpus EA/multi tracker counts are on seen audio for FT runs. Fix: `shared/leak_guard.py`, `*_rx_cuts_fbank` manifests (re-verified at 0 flags), staging drops the reciters at ingest, `train()` refuses leaky manifests. Audit: `scripts/phase0_modal.py --action leak-audit` (apps `ap-O2XdOsOiehSl4JOODDYYxp`, `ap-eGr0HK4nfcPaOHG43dG88N`).

**q-lab PER, ONNX streaming greedy fp32, two golds** (`scripts/qlab_per_eval.py` + `qlab_per_report.py`). `ordered` = `ordered_quran_phonemes[surah:ayah]` (all prior numbers); `text` = Quran-Lab gold `quran_text2phoneme[norm(reference_text)]` with v1.1 references (misses 82 tlog / 11 nufais / 4 EA clips — their `quran_per_eval.py` drops misses). The golds disagree at ayah-final madd (4 vs 2 harakat) on 259/486 shared clips; `madd-free` collapses madd runs. The 11 v1.1 nufais clips recite the ayah twice, so `ordered` scores correct transcription as insertions (v3 nufais I 7.16 on 200 vs 3.31 on 189).

| model | ordered ALL / EA / nufais / tlog (583) | madd-free ALL / tlog | text ALL / tlog (486 / 117) |
|---|---|---|---|
| **v3 (control)** | 6.27 / 4.24 / 7.91 / 6.90 | 5.26 / 5.42 | 4.15 / 5.89 |
| v3 int8 | 6.27 / 4.23 / 7.97 / 6.83 | — | 4.12 / 5.75 |
| v3.1 | 5.56 / 2.39 / 7.78 / 7.06 | 4.96 / 5.93 | 4.19 / 7.24 |
| ft-gentle † | 4.32 / 1.47 / 4.55 / 8.39 (D 6.27) | 4.17 / 8.17 | 4.00 / 8.62 |
| interp α=0.1 † | 5.14 / 1.96 / 7.55 / 6.36 | 4.73 / 5.54 | 4.16 / 7.24 |
| interp α=0.3 † | 4.49 / 1.53 / 7.22 / 4.89 | 4.33 / 4.66 | 4.15 / 7.24 |
| interp-gentle α=0.5 † | 4.09 / 1.48 / 6.57 / 4.35 | 4.00 / 4.22 | 4.06 / 6.85 |
| interp α=0.7 † | 3.73 / 1.44 / 5.70 / 4.25 | 3.63 / 4.11 | 3.87 / 6.35 |
| interp α=0.9 † | 3.67 / 1.35 / 4.82 / 5.50 (D 3.21) | 3.55 / 5.32 | 3.77 / 6.85 |

† EA leaked (above). Interp = v3.1 ⊕ ft-gentle epoch-1 (α on FT); exports `/vol/exports/interp-gentle-a{0.1,0.3,0.7,0.9}`.

**Interp artifact check.** The tlog gain is a smooth interior minimum of the α curve (madd-free 5.93 → 5.54 → 4.66 → 4.22 → 4.11 → 5.32 → 8.17), not an artifact: FT deletions only take off at α ≥ 0.9. It is gold-dependent — under the 2-harakat text gold every blend is worse than v3 on tlog (best 6.35 vs 5.89) because FT labels put 4-harakat madd at waqf. **FT learns to not hear deviations:** tlog insertions fall monotonically with α (2.24 → 0.57); on the 46 v3-non-clean holdout clips (repeats / restarts) insertions go v3 10.4 → interp 5.5 → ft-gentle 2.7; on the 11 doubled-recitation nufais clips hyp/ref (single-ayah gold) goes v3 2.00 (10/11 transcribe both) → ft-gentle 1.37 (3/11). Part of every FT "gain" on tlog/nufais is suppressing real speech the labels omit.

**Held-out multi-ayah windows** (`benchmark/build_heldout_multi.py`, 58 windows of 2–4 ayahs from the three q-lab reciters, 0.5 s gaps, 30 with a ≤4-word bridge ayah; audio + manifest off-repo). v3 PER 2.33, 0/157 ayahs dropped (≥50% deleted); v3.1 1.60, 0; interp α=0.3 / 0.5 / 0.7 † 0.70 / 0.63 / 0.70, 0; α=0.9 † 12.48, 17 dropped; **ft-gentle † 27.16 (D 26.90), 48/157 ayahs dropped (26/72 short), 36/58 windows** — on audio it trained on.

**TLOG filter** (`phase0_modal.py --action tlog-filter --model v3`, app `ap-V5UTfjexs1ZWUT89ZCy68M`, 64 × 2-CPU shards, 0 errors; per-clip scores at `/vol/phase0/tlog_filter/v3/`). All 41,083 staged TLOG clips vs their training label: **clean (PER ≤ 0.10) 14,521 / 31.6 h; suspect 5,882 / 13.0 h; > 0.35 20,680 / 55.4 h**; median PER 0.36. Forced-score medians per token: −0.06 / −0.90 / −10.8. 3,732 non-clean clips (10.3 h) decode within PER 0.10 of a *different* ayah — 2,568 of them the previous ayah (label = audio + 1); across all clips with ayah ≥ 2, 6,904 match the previous ayah better than the label (not a basmala-numbering shift: 0 ayah-1 clips are the basmala). The FT mix trained on all 100 h. `tlog_clean_v3_cuts_fbank` = 14,354 cuts (clean ∩ `tlog_rx`). q-lab `tlog_holdout` under the same filter: 154 clean / 39 suspect / 7 > 0.35, and 0/46 non-clean decode closer to another ayah (only identical-text pairs) — the holdout was curated, its residual error is reciter deviation, not wrong labels. tlog PER on the v3-clean 154 (selection favours v3): v3 3.11, v3.1 3.36, ft-gentle 5.86, interp-gentle-a0.5 1.67.

### Phase 0 completion + A0 (2026-09-30, afternoon)

**Dev split (reciter-disjoint).** Six everyayah.com Hafs reciters that sit outside the curated 35-reciter set *and* outside every training manifest (EA 28 dirs + 34 QUA mushaf names): Ghamadi_40kbps, Ali_Jaber_64kbps, Fares_Abbad_64kbps, khalefa_al_tunaiji_64kbps, Karim_Mansoori_40kbps, Parhizgar_48kbps. 569 single ayahs + 48 2–4-ayah windows, 1.91 h (`benchmark/build_heldout_multi.py --reciters dev`, off-repo; on the volume at `/vol/phase0/corpora/dev_everyayah`). Quran-Lab says v3 saw "36 EveryAyah reciters"; the curated set is 35, so these six are *probably* not in v3's EA tier, but could be in its 1,161-reciter archive tier (unknowable). Phone-domain dev: 267 clean TLOG clips hash-held out of `tlog_clean_v3` (clip-disjoint only) + 300 non-clean TLOG clips (never trained on) for the insertion gate. All three q-lab reciters stay test-only. Excluded as dev: Ibrahim Akhdar, Mahmoud Ali Al-Banna, Mustafa Ismail, Saud Al-Shuraim (QUA mushafs), Ahmed_Neana / Mohammad_al_Tablaway (likely in v3's EA tier).

**QUA fingerprint.** 399 QUA clips are (surah, ayah, ±60 ms) twins of held-out EA / nufais clips. Best-lag fbank cosine: twins max 0.377, same-ayah other-recording controls max 0.371, same-recording positive control (q-lab wav vs everyayah.com mp3) 0.80–0.87 → 0 copies of held-out recordings in QUA.

**Perturbed-copy bug.** fbank manifests hold `_sp0.9` / `_sp1.1` copies under their own ids; the first `tlog_clean_v3` kept only the 1.0× cut (14,090 of 42,270). Fixed (flags follow the base id) before A0; `tlog_clean_v3` = 42,270 cuts.

**Headline PER** (promotion metric) = Quran-Lab v1.1 references over all 600 clips: their `quran_text2phoneme` gold where the table has the reference (503), else ordered phonemes × repeats with waqf madd at 2 harakat (97). Madd-free = secondary. **v3 control: headline 4.45 (EA 3.10 / nufais 4.64 / tlog 6.35), madd-free 3.76.**

**Correction eval (synthetic) — withdrawn.** An earlier version of this entry measured the correction engine on mistakes spliced into clean recitation. Deliberately creating incorrect Quran recitation, including by editing audio, is not permissible; the builder, the audio and the numbers were removed (2026-10-01). Correction is now evaluated only on genuine slips (see the correction entries below).

**A0** (`ap-yqax5YcK7boRkMdpshQumQ`, H100:4, init v3 `.pt`, `everyayah_rx,qua_rx,iqra_rx,retasy_rx,tlog_clean_v3`, 2 ep, lr 0.001, warmup 1000; leak-check 0 flags on all five sources; valid 0.0292 ep1 / 0.0110 mid-ep2; `/vol/exp/a0-v3-clean`). Exports `/vol/exports/a0-ep{1,2}[-a0.5|-a0.7]` (α on FT, init v3). Eval: `scripts/eval_modal.py` + `promotion_gates.py`.

| metric | v3 | ep1 | ep1 α0.5 | ep1 α0.7 | ep2 | ep2 α0.5 | ep2 α0.7 |
|---|---|---|---|---|---|---|---|
| headline | 4.45 | 4.42 | 4.37 | 4.26 | 4.46 | **4.17** | 4.26 |
| madd-free headline | 3.76 | 3.11 | 3.15 | 2.99 | 3.23 | **2.96** | 3.03 |
| madd-free EA / nufais / tlog | 2.89 / 3.62 / 5.41 | 1.54 / 4.22 / 3.91 | 2.06 / 3.45 / 4.42 | 1.76 / 3.52 / 4.13 | 1.47 / 4.66 / 3.83 | 1.80 / 3.42 / 4.09 | 1.63 / 4.02 / 3.73 |
| held-out multi: ayahs dropped (PER) | 0 (2.34) | 11 (7.37) | 0 (1.14) | 0 (1.36) | 40 (22.25) | 0 (1.02) | 0 (1.19) |
| non-clean ins: holdout / tlog dev (floor 0.85×v3) | 10.38 / 22.01 | 5.62 / 16.56 | 9.29 / 22.12 | 7.57 / 20.61 | 5.78 / 15.29 | 8.67 / 21.51 | 7.03 / 19.48 |
| dev single ordered / madd-free | 2.89 / 1.00 | 0.79 / 0.73 | 1.66 / 1.53 | 1.28 / 1.20 | 0.63 / 0.60 | 1.14 / 1.03 | 0.83 / 0.78 |
| dev multi drops | 0 | 12 | 0 | 0 | 28 | 0 | 0 |
| tlog dev | 3.76 | 1.94 | 1.97 | 1.76 | 1.92 | 1.74 | 1.73 |
| tracker held-out multi / v1 | 56/58, 52/53 | | 56/58, 52/53 | | | 56/58, 53/53 | |
| **gates** | | 3 fail | **all pass** | ins | all fail | ins (8.67 < 8.83) | ins |

Paired bootstrap (2,000 resamples, clip level) of the headline delta vs v3: ep1-α0.5 [−0.36, +0.17] (not significant), ep1-α0.7 [−0.54, +0.17], ep2-α0.5 [−0.56, −0.01]. Raw FT again deletes whole ayahs in multi-ayah audio (ep2: 40/157 held-out, 28 dev) even trained leak-free from v3, and suppresses insertions on deviant speech; the blends repair both. Headline − madd-free is 0.69 for v3 and ~1.2 for FT/blends: our training labels put waqf madd at 4 harakat while the v1.1 gold uses 2, costing FT ≈ 0.5 pp of headline. **Verdict: ep1-α0.5 passes every gate but its headline gain is inside noise; ep2-α0.5 has the only significant gain and misses the insertion floor by 0.16 pp. Not promoted.** Next single-change arms: waqf-madd label convention (2 harakat at ayah end), multi-ayah windows (A0w).

### Arms after A0 (2026-09-30, evening): A0e, A0w, A0t, A1

One change per arm, same recipe as A0 (v3 init, lr 0.001, warmup 1000, H100:4), same eval + gates; CIs are paired clip-level bootstraps of the headline delta (2,000 resamples, 95%). Best control logged: **a0-ep1-a0.5** (4.37, passes all gates, CI vs v3 [−0.35, +0.18]).

- **A0e — waqf-2 labels** (`ap-WQyAQPHJXwB01o9ZoMLrRB`, sources `*_w2`). Ayah-final 4-beat madd (optionally before ≤2 closing chars) → 2 beats; matches the v1.1 gold on 406/487 shared clips (was 228); 51–82% of cuts per source change, rebuilt labels = originals on every cut. Headline drops ~0.6 pp at every variant (CI vs A0 all negative, e.g. ep1-α0.5 −0.60 [−0.76, −0.43]); madd-free vs A0 ≈ 0 for blends (it is a label-convention fix, not acoustics). **a0e-ep1-a0.5: 3.77, all gates pass, CI vs v3 [−0.96, −0.43].**
- **A0w — multi-ayah windows** on A0e (`ap-86NAkX4OJwTIkzMaBNcE7j`, + `everyayah_multi_rx_w2x3` = the leak-free window set repeated ×3 → 30% of epoch hours; lhotse `CutSet.mux` does not stop early, so mux weights alone never changed exposure — E4's "2.5× weight" was a no-op on totals). Raw FT ayah drops 11/25 → 2/1 (ep1/ep2). **a0w-ep1-a0.5: 3.60, all gates, CI vs A0e counterpart −0.17 [−0.27, −0.07]** — new best. a0w-ep2-a0.5 also passes (3.60, best insertion 9.60) but vs A0e it is +0.05 [−0.04, +0.16].
- **A0t — relabelled TLOG** on A0w (+ `tlog_relab_v3_w2_rx`: 3,690 non-clean clips whose decode is within PER 0.10 of a different ayah, relabelled; 1 clip dropped as a holdout twin after relabel — the train-time leak check refused the first launch). No gain vs A0w: ep1-α0.5 +0.01 [−0.07, +0.10], ep2-α0.5 +0.11 [+0.02, +0.20]. **Killed** (plan rule: no gain); A0w stays the base.

| variant | headline | madd-free | drops | ins holdout / tlog-dev | tracker held-out, v1 | gates failed | CI vs v3 | CI vs previous arm |
|---|---|---|---|---|---|---|---|---|
| v3 | 4.45 | 3.76 | 0 | 10.38 / 22.01 (floors 8.83 / 18.71) | 56/58, 52/53 | — | — | — |
| a0-ep1-a0.5 | 4.37 | 3.15 | 0 | 9.29 / 22.12 | 56, 52 | — | −0.09 [−0.35, 0.18] | — |
| a0e-ep1-a0.5 | 3.77 | 3.09 | 0 | 8.98 / 21.90 | 56, 53 | — | −0.69 [−0.96, −0.43] | vs A0 −0.60 [−0.76, −0.43] |
| a0e-ep2-a0.7 | 3.46 | 2.77 | 0 | 6.48 / 19.37 | 56, 53 | ins holdout | −1.00 [−1.38, −0.64] | vs A0 −0.81 [−1.12, −0.55] |
| **a0w-ep1-a0.5** | **3.60** | 2.93 | 0 | 9.29 / 22.16 | 56, 53 | — | −0.86 [−1.12, −0.60] | vs A0e −0.17 [−0.27, −0.07] |
| a0w-ep2-a0.5 | 3.60 | 2.93 | 0 | 9.60 / 22.16 | — | — | −0.86 [−1.10, −0.62] | vs A0e +0.05 [−0.04, 0.16] |
| a0w-ep2-a0.7 | 3.41 | 2.74 | 0 | 7.81 / 20.75 | 56, 53 | ins holdout | −1.04 [−1.37, −0.72] | vs A0e −0.04 [−0.19, 0.11] |
| a0t-ep1-a0.5 | 3.61 | 2.95 | 0 | 9.45 / 22.07 | 56, — | — | −0.84 [−1.10, −0.60] | vs A0w +0.01 [−0.07, 0.10] |
| a0t-ep2-a0.7 | 3.37 | 2.68 | 0 | 7.03 / 20.42 | 56, — | ins holdout | −1.09 [−1.45, −0.74] | vs A0w −0.05 [−0.18, 0.08] |
| a1-ep1-a0.5 | 3.95 | 3.25 | 0 | 10.23 / 22.75 | 56, 53 | — | −0.51 [−0.77, −0.25] | vs A0w +0.35 [+0.23, +0.48] |
| a1-ep1-a0.7 | 3.49 | 2.81 | 0 | 7.57 / 20.78 | 56, 53 | ins holdout | −0.97 [−1.28, −0.67] | vs A0w −0.05 [−0.24, 0.12] |

- **A1 — CR-CTC** on A0w (`--use-cr-ctc 1 --enable-spec-aug 0 --cr-loss-scale 0.2 --time-mask-ratio 2.5`, `--max-duration 600` per the recipe, 1 epoch — every best variant so far is epoch 1 and Eden's schedule does not depend on total epochs). Worse at α 0.5 (+0.35 [+0.23, +0.48] vs a0w-ep1-a0.5; raw ep1 +0.21 [0.00, +0.43]); α 0.7 ties (−0.05, CI crosses 0) and fails the insertion floor. It does keep more insertions (10.23 at α 0.5, above v3's floor comfortably). **Killed** (no PER gain vs A0w). Confound: halved max-duration at the same LR means 2× optimizer steps of half-size batches.

**Current best: a0w-ep1-a0.5** (waqf-2 labels + ×3 multi-ayah windows, epoch 1, 50/50 with v3; `/vol/exports/a0w-ep1-a0.5`): headline 3.60 vs v3 4.45 (−0.86 [−1.12, −0.60]), madd-free 2.93 vs 3.76, 0 dropped ayahs, insertions 9.29 / 22.16, tracker 56/58 + v1 53/53. Not promoted to the shipped model (no browser/int8/latency row yet).

- **A0w50 — windows at ~50% of epoch hours** (`ap-H1WZ2wrjkdk0WDzm2Fe6xC`, `everyayah_multi_rx_w2x7`, 735,693 cuts; otherwise identical to A0w; leak-check 0 flags; valid 0.0116 / 0.0048). No gain: best gate-passing variant ep1-α0.5 = 3.66 vs a0w-ep1-a0.5 3.60, CI +0.06 [−0.02, +0.15]; ep2-α0.5 3.78, +0.18 [+0.08, +0.28] (worse). α 0.7 variants score 3.40–3.43 but fail the insertion floor, as in every arm. ep1-α0.5: tracker 56/58. Raw drops 2 / 1, same as ×3. **Killed** — 30% windows stay; 70% not run. Spend ~$31.8.

**int8 + browser (2026-10-01).** Dynamic-int8 exports, same eval:

| model | headline | madd-free | drops | ins holdout / tlog-dev | gates vs v3 | tracker held-out, v1 | browser `test:browser` |
|---|---|---|---|---|---|---|---|
| a0w-ep1-a0.5 int8 | 3.59 | 2.92 | 0 | 9.37 / 22.10 | pass | 56/58, 53/53 | 6/6, RTF 0.100, ready 1112 ms |
| v3 int8 | 4.47 | 3.79 | 0 | 10.46 / 22.05 | headline (4.468 vs 4.452) | 56/58, 52/53 | — |
| shipped interp-gentle-a0.5 int8 | 4.16 | 2.95 | 0 | 5.70 / 15.91 | **both insertion gates** | — | 6/6, RTF 0.100, ready 1437 ms |

a0w int8 − fp32: headline −0.01, madd-free −0.01, insertions +0.08 / −0.05, tracker unchanged. a0w int8 also passes vs v3 int8. The currently shipped model fails the insertion floor (it suppresses deviant speech). Browser latency is end-to-end over the 6 default clips (134 s audio, ~13.5 s wall); identical for both models.

Full 6-variant rows per arm (raw ep1/ep2 + α 0.5/0.7) are in the per-arm reports. Pattern across every arm: α 0.7 and raw epochs score lower headline but fail the insertion floor (they learn to not transcribe deviations); α 0.5 at epoch 1 is the only setting that passes everything each time.

## Correction eval on real recordings (baseline)

2026-10-01. Correction mode of the shipped recitation engine (`recognize(..., mode="correction")`, issues dismissed on sight) over real audio. One CPU pass per model (RTF 0.052 shipped int8, 0.068 v3 fp32, 0.062 a0w fp32). Engine rules were not changed. Scorer: `lab/scripts/correction_eval.py`.

Real slips only. No audio was spliced, deleted, duplicated, swapped, or resynthesized, and no acted-mistake set was used. Help `use=slip` is a reciter's self-reported accidental slip at an unknown spot. TLOG rows are an unverified recall set from `locate_slips.py` (no one has listened; substitutions in particular may be a shared model mis-hear). Per-clip rows stay in `/tmp/correction_eval/`.

A slip counts as caught when any issue is on the same surah:ayah and `word` is within ±1 of the located word. Kind-correct: omitted → `possible_omission`, substituted → `possible_substitution`, repeated/restarted → any word kind or `unclear_ayah`. Exact-word is distance 0. Latency is the earliest catching issue's `atSeconds` minus the slip span start (median over catches that have a span).

### Data

| set | clips | minutes | what it is |
|---|---:|---:|---|
| Help clean (dev+test) | 266 | 48.79 | `use=clean`. Dev 105 / 19.82 min, test 161 / 28.98 min |
| Help slip | 10 | 3.38 | Self-reported slips. 8 single-ayah, 2 multi-ayah |
| Help slips located | 5 clips, 10 slips | — | v3 and a0w agree on a review-grade word. 9 substituted, 1 repeated. Both multi-ayah clips stayed unlocated |
| TLOG clean dev | 267 | 33.68 | `phase0/tlog_filter/v3/dev_ids.txt` (PER ≤ 0.10) |
| v1 | 53 | 16.95 | `test_corpus` manifest |
| TLOG candidates | 1,119 clips / 1,519 slips | 202.72 | Every omitted (1,027), repeated (74), restarted (18), plus 400/2,795 substituted rows (seed 0). Unverified |

Help locations use the same review bar as the TLOG file (`is_review_slip`: whole-word edit, repeat, restart, or a partial edit of ≥2 tokens covering at least half the word) and keep a word only when v3 and a0w both mark it. Multi-ayah takes are aligned as one passage (`ayah_local_slip`) and the word index is mapped back into its ayah. Both 2-ayah clips produced a review slip on at least one model; the models did not share a word, so those slips are unlocated and are not in the recall denominator. Three single-ayah clips produced no review slip on either model.

### False flags per clean minute

Primary guardrail. Parentheses are issue count and clips with any flag.

| model | help clean | TLOG clean dev | v1 |
|---|---|---|---|
| shipped interp-gentle-a0.5 int8 | 0.020 (1, 1/266) | 0.119 (4, 4/267) | 0 (0/53) |
| v3 fp32 | 0.041 (2, 2/266) | 0.119 (4, 4/267) | 0.059 (1, 1/53) |
| a0w-ep1-a0.5 fp32 | 0.061 (3, 3/266) | 0.089 (3, 3/267) | 0 (0/53) |

Shipped help-clean's only issue is `unclear_ayah`. Shipped v1 is clean. TLOG-dev flags are mostly `possible_vowel`.

### Recall

Help, on the 10 located slips (unverified word spot; the clip itself is a real self-report):

| model | recall | kind-correct | exact word | median latency |
|---|---|---|---|---|
| shipped | 1/10 | 0/10 | 1/10 | 5.12 s |
| v3 | 2/10 | 1/10 | 1/10 | 4.56 s |
| a0w | 1/10 | 0/10 | 1/10 | 5.12 s |

The shipped and a0w catch is a `possible_vowel` on a partial substitution (exact word, wrong kind). v3's kind-correct hit is `possible_substitution` on one of those nine. The one repeated slip was missed by all three. No omitted or restarted slip was located in this set.

TLOG candidates, unverified, by kind. Cells are caught/n.

| model | omitted (1027) | repeated (74) | restarted (18) | substituted sample (400) | all (1519) |
|---|---|---|---|---|---|
| shipped recall | 2 | 1 | 0 | 4 | 7 |
| shipped kind-correct | 2 | 1 | 0 | 0 | 3 |
| shipped exact | 2 | 0 | 0 | 2 | 4 |
| shipped median latency | 3.48 s | 2.20 s | — | 5.09 s | 4.48 s |
| v3 recall | 2 | 1 | 0 | 7 | 10 |
| v3 kind-correct | 2 | 1 | 0 | 3 | 6 |
| v3 exact | 2 | 0 | 0 | 3 | 5 |
| v3 median latency | 3.48 s | 2.20 s | — | 2.88 s | 2.78 s |
| a0w recall | 3 | 2 | 1 | 6 | 12 |
| a0w kind-correct | 2 | 2 | 1 | 3 | 8 |
| a0w exact | 3 | 1 | 0 | 2 | 6 |
| a0w median latency | 2.88 s | 3.00 s | 3.92 s | 2.62 s | 2.76 s |

811/1,519 TLOG slips are the first or last word of the ayah. Every model caught 0 of those. All catches are among the 708 middle words (shipped 7/708, v3 10/708, a0w 12/708).

### Help clean, false flags per minute by slice

Same minutes for every model. Cell is flags/min (issues).

| slice | value | clips | min | shipped | v3 | a0w |
|---|---|---:|---:|---:|---:|---:|
| device | phone | 196 | 36.48 | 0.027 (1) | 0.055 (2) | 0.055 (2) |
| device | laptop | 55 | 8.93 | 0 | 0 | 0 |
| device | desktop | 6 | 1.66 | 0 | 0 | 0 |
| device | tablet | 6 | 1.29 | 0 | 0 | 0.777 (1) |
| device | headset | 3 | 0.43 | 0 | 0 | 0 |
| gender | unknown | 134 | 25.44 | 0.039 (1) | 0.039 (1) | 0.118 (3) |
| gender | male | 128 | 22.45 | 0 | 0.045 (1) | 0 |
| gender | female | 4 | 0.91 | 0 | 0 | 0 |
| level | unknown | 134 | 25.44 | 0.039 (1) | 0.039 (1) | 0.118 (3) |
| level | intermediate | 80 | 15.07 | 0 | 0 | 0 |
| level | hafiz | 25 | 4.31 | 0 | 0 | 0 |
| level | beginner | 27 | 3.98 | 0 | 0.251 (1) | 0 |
| ayah | single | 223 | 36.15 | 0 | 0.028 (1) | 0.055 (2) |
| ayah | multi | 43 | 12.65 | 0.079 (1) | 0.079 (1) | 0.079 (1) |
| split | dev | 105 | 19.82 | 0 | 0.050 (1) | 0 |
| split | test | 161 | 28.98 | 0.035 (1) | 0.035 (1) | 0.104 (3) |

Gender and level are missing on 134 help-clean clips (recorded as unknown). Shipped's single help-clean flag is on a multi-ayah test clip from a phone.

### Why recall stays low

The word rules in `packages/core/src/recitation/correction.ts` are written for a gross mismatch between two clearly heard words. That is what the numbers show.

1. **Both neighbours must already be clear, so edges never flag.** `possibleWordIssues` returns nothing unless the previous and next word are the same ayah and `clearWord`: `ok`, distance ≤ 0.15, margin ≥ 0.55, heardRatio in [0.75, 1.3]. The comment on that gate is explicit: do not infer leading or trailing omissions. 811/1,519 TLOG slips sit on word 0 or the last word, and all three models caught none of them.
2. **Omission and substitution require a gross hole, not a partial edit.** Omission fires only for `state === 'skipped'` and `heardRatio === 0`. Substitution fires only for `wrong` with distance ≥ 0.6, margin ≥ 0.65, and heardRatio in [0.5, 1.5]. On the shipped model, partial omissions were 0/259 and whole-word omissions 2/768; kind-correct substitutions were 0/400. Of the 49 issues on the shipped candidate run, 41 are `possible_vowel`, which needs the consonant skeleton to already match (distance ≤ 0.15) plus a sure vowel error. A partial substitution the aligner marks is often inside that vowel gate and outside the substitution gate, so it can count as a catch and still miss kind-correct.
3. **Nothing in `possibleWordIssues` represents a repeat or a restart, and a candidate must hold for 12 frames.** Repeated and restarted slips are kind-correct only if some other word kind or `unclear_ayah` happens to land on them. `unclear_ayah` is an ayah-level kind: the session raises it when ayah N+2 is matched immediately after N, which a restart inside one ayah does not do. Shipped recall is 1/74 repeated and 0/18 restarted. `CorrectionController.observe` also waits until the same kind has persisted for 12 frames before it raises, then a dismiss suppresses that `wordIndex` for the rest of the clip. Median latency on the catches that exist is 2–5 s after the slip span start.

## Correction rules v2 (real recordings)

2026-10-01. Four rule changes in `packages/core/src/recitation/correction.ts`, made one at a time, each ablated, then frozen and scored once on held-out data. Same data, scorer and no-synthetic-mistakes rule as the baseline above. Aggregates only; per-clip rows stay in `/tmp/correction_eval/`.

**Split.** Tuning: help clean `split=dev` (105 clips, 19.82 min), TLOG clean dev (267, 33.68 min), and the TLOG candidate half with an even sha1 of the clip id (699 slips: 476 omitted, 31 repeated, 7 restarted, 185 substituted). Held-out: help clean `split=test` (161, 28.98 min), help slips (all 10 located slips are in test, so tuning had no help recall data), the odd TLOG half (820 slips: 551 / 43 / 11 / 215), and v1 (53, 16.95 min). TLOG clean dev has no split and is reported in both.

**Two measurements.**
- *Live*: `correction_eval.py run`, the same harness as the baseline (issues dismissed on sight). This is the headline.
- *Paired replay*: `run --trace` records every controller input (verdicts, cursor, frame, settle snapshots) with a controller that never flags. Then `replay_correction.ts` re-runs any rule set offline over the same trace, so two rule sets are compared on identical engine output. All ablations use it.

Two things make the measurements differ.
- *Inference noise.* onnxruntime CPU inference is not bit-identical between runs with 4 intra-op threads; it is with 1. Margins move in the third decimal, and that can flip a word state. Any single live run, the baseline included, carries per-clip noise of a catch or two. Paired replay removes it.
- *Dismiss on sight.* After a dismiss the session resets the streaming encoder and re-tracks from the resume cursor, behind the reciter. The next 1–2 s decode poorly (words come back unsure or wrong, and the tracker briefly loses lock), so a second slip in the same clip is rarely flagged live. In the app, audio is dropped while the dialog is open, so this harness artifact affects baseline and v2 alike. Replay does not dismiss, so it reads higher (shipped tuning: 13 live-equivalent catches in replay versus 9 live).

Cells below are caught / kind-correct / exact-word / median latency. FF is false flags per clean minute (issues).

### Changes

1. **Partial omission** (`omissionMaxHeard`, default 1). A `skipped` word (heard ratio < 0.34) with clear neighbours is an omission even when part of it was heard. Before, only heard ratio 0 counted.
2. **GOP rule** (`gopFlag` −3, `gopAnchor` −2). Correction mode now attaches CTC posteriors, and interior verdicts carry goodness-of-pronunciation scores: `gop`, the per-token log ratio of forcing the expected word against the free best path over its window, plus `gopNone`, `gopTwice`, `repGain` and `pairGop`. A word the aligner already marks `wrong` or `skipped` is flagged when its GOP is ≤ −3, it is the local minimum, and both same-ayah neighbours fit (clear, or GOP ≥ −2 and not skipped). The neighbours do not have to be `clear`. Kind: omission if the word is skipped or silence fits the window clearly better (`gopNone − gop ≥ 2` and `gopNone ≥ −3`), else substitution. `ok`, `unsure` and `pending` words are never flagged by GOP.
3. **Settle** (`settle`). When the tracker is dropped (stop, surah completed, silent idle), the session runs `CorrectionController.settle()` once on settled verdicts, covering only the words `observe()` never judged with both neighbours and the word after settled. No persistence is needed, because settled verdicts do not change. Vowel flags stay observe-only.
4. **`possible_repetition`** (`repetitionGain` 5). A word that fits once (GOP ≥ −2) but gains ≥ 5 nats per token from forcing a second copy, with a clear left neighbour. A neighbour that also repeats vetoes it, since a phrase restart (waqf then ibtida') is accepted practice. A retry that repeats the word again is not accepted.

New kind and thresholds are optional fields with defaults, so older thresholds objects and switch statements keep working. The web demo has a label for the new kind.

### Ablations (tuning part, paired replay)

Each row replays the final code with later rules switched off (cumulative) or one rule switched off (leave-one-out). TLOG columns are the 699 tuning-half slips (omitted 476, repeated 31, restarted 7, substituted 185).

| rules | model | FF help clean dev | FF TLOG clean dev | omitted | repeated | restarted | substituted | all |
|---|---|---|---|---|---|---|---|---|
| baseline | shipped | 0.000 (0) | 0.119 (4) | 1/1/1/5.16s | 1/1/0/2.20s | 0/0/0/— | 1/0/0/4.48s | 3/2/1/4.48s |
| baseline | a0w | 0.000 (0) | 0.089 (3) | 2/1/2/4.02s | 2/2/1/3.00s | 0/0/0/— | 4/2/1/2.62s | 8/5/4/2.76s |
| +partial omission | shipped | 0.000 (0) | 0.119 (4) | 2/2/2/3.94s | 1/1/0/2.20s | 0/0/0/— | 2/0/1/3.16s | 5/3/3/2.72s |
| +partial omission | a0w | 0.000 (0) | 0.089 (3) | 4/3/3/2.80s | 2/2/1/3.00s | 0/0/0/— | 6/2/2/2.62s | 12/7/6/2.68s |
| +GOP | shipped | 0.000 (0) | 0.119 (4) | 3/3/2/2.72s | 2/2/0/7.34s | 0/0/0/— | 7/4/3/4.48s | 12/9/5/3.60s |
| +GOP | a0w | 0.000 (0) | 0.089 (3) | 4/3/3/2.80s | 2/2/1/3.00s | 0/0/0/— | 13/9/5/3.44s | 19/14/9/2.88s |
| +settle | shipped | 0.000 (0) | 0.119 (4) | 3/3/2/2.72s | 2/2/0/7.34s | 0/0/0/— | 7/4/3/4.48s | 12/9/5/3.60s |
| +settle | a0w | 0.000 (0) | 0.089 (3) | 5/4/4/2.88s | 2/2/1/3.00s | 0/0/0/— | 13/9/5/3.44s | 20/15/10/3.16s |
| +repetition (final) | shipped | 0.000 (0) | 0.119 (4) | 3/3/2/2.72s | 2/2/0/7.34s | 1/1/1/3.16s | 7/4/3/4.48s | 13/10/6/3.16s |
| +repetition (final) | a0w | 0.000 (0) | 0.089 (3) | 5/4/4/2.88s | 2/2/1/3.00s | 1/1/1/3.16s | 13/9/5/3.44s | 21/16/11/3.16s |
| final − partial omission | shipped | 0.000 (0) | 0.119 (4) | 3/3/2/2.72s | 2/2/0/7.34s | 1/1/1/3.16s | 7/4/3/4.48s | 13/10/6/3.16s |
| final − partial omission | a0w | 0.000 (0) | 0.089 (3) | 5/4/4/2.88s | 2/2/1/3.00s | 1/1/1/3.16s | 13/9/5/3.44s | 21/16/11/3.16s |
| final − GOP | shipped | 0.000 (0) | 0.119 (4) | 2/2/2/3.94s | 1/1/0/2.20s | 1/1/1/3.16s | 2/0/1/3.16s | 6/4/4/2.94s |
| final − GOP | a0w | 0.000 (0) | 0.089 (3) | 4/3/3/2.80s | 2/2/1/3.00s | 1/1/1/3.16s | 6/2/2/2.62s | 13/8/7/2.72s |
| final − settle | shipped | 0.000 (0) | 0.119 (4) | 3/3/2/2.72s | 2/2/0/7.34s | 1/1/1/3.16s | 7/4/3/4.48s | 13/10/6/3.16s |
| final − settle | a0w | 0.000 (0) | 0.089 (3) | 4/3/3/2.80s | 2/2/1/3.00s | 1/1/1/3.16s | 13/9/5/3.44s | 20/15/10/3.02s |
| final − repetition | shipped | 0.000 (0) | 0.119 (4) | 3/3/2/2.72s | 2/2/0/7.34s | 0/0/0/— | 7/4/3/4.48s | 12/9/5/3.60s |
| final − repetition | a0w | 0.000 (0) | 0.089 (3) | 5/4/4/2.88s | 2/2/1/3.00s | 0/0/0/— | 13/9/5/3.44s | 20/15/10/3.16s |

- GOP carries most of the gain: shipped 5 → 12, a0w 12 → 19, mostly substitutions, which gain kind-correct catches (shipped 0 → 4).
- Partial omission is subsumed once GOP is on (removing it from the final rules changes nothing). It is kept because it holds without GOP scores (tracking-mode verdicts, custom hosts).
- Settle and repetition add one catch each, at no FF cost.
- FF did not move in any row. Every catch is an interior slip (span more than 0.3 s from either clip edge).

Live on the tuning part, final rules: shipped 9/699 (kind-correct 8, exact 3, 4.48 s) versus 3/699 at baseline. a0w 18/699 (12, 8, 3.02 s) versus 8/699. FF: help clean dev 0 / 0, TLOG clean dev 4 / 3, identical to baseline.

### Rejected (raised FF, or recall that was an artifact)

| change | why rejected |
|---|---|
| GOP alone (any word state) at −3 / −5 / −8 | Shipped TLOG clean dev 4 → 5 at every threshold |
| GOP on `unsure` words too | Shipped TLOG clean dev 4 → 5; a0w help clean dev 0 → 4, TLOG clean dev 3 → 6 |
| `gopFlag` −2 / −2.5 | Help clean dev 0 → 1–2 |
| Stricter anchors or heard-ratio gates on the GOP rule | Raised FF on one of the clean sets, no recall gain |
| Neighbours from the adjacent ayah | +1 help clean FF on both models, no recall gain |
| First/last word with only the inner neighbour | +4 / +6 catches, all at clip boundaries (segmentation cuts), 0 interior |
| `repetitionGain` 2 / 3 | +1 help clean FF on shipped. 4 was clean but sits next to that cliff, so 5 was kept |
| Repetition from backward tracker jumps | 8 same-ayah backward jumps per clean set on clean takes |
| `persistFrames` 0 (raise on first sight) | Recall unchanged on shipped, latency 3.6 → 3.1 s. Removes the guard against revised hypotheses |
| `persistFrames` 18 / 24 | Latency only. 6 and 8 behave like 12 at the 480 ms chunk size |

A GOP veto for the three shipped `possible_vowel` flags on TLOG clean dev was not attempted: those words have GOP near 0, and so do real single-vowel errors.

### Held-out (scored once, rules frozen)

Live, same harness as the baseline:

| set | shipped baseline | shipped v2 | a0w baseline | a0w v2 |
|---|---|---|---|---|
| FF help clean test | 0.035 (1) | 0.035 (1) | 0.104 (3) | 0.104 (3) |
| FF TLOG clean dev | 0.119 (4) | 0.119 (4) | 0.089 (3) | 0.089 (3) |
| FF v1 | 0 | 0 | 0 | 0 |
| Help slips (10) | 1/0/1/5.12s | 1/1/1/4.16s | 1/0/1/5.12s | 1/0/1/5.12s |
| TLOG omitted (551) | 1/1/1/1.80s | 7/6/7/2.76s | 1/1/1/1.80s | 7/6/7/2.72s |
| TLOG repeated (43) | 0/0/0/— | 1/1/1/2.10s | 0/0/0/— | 1/1/1/8.36s |
| TLOG restarted (11) | 0/0/0/— | 0/0/0/— | 1/1/0/3.92s | 2/2/0/3.84s |
| TLOG substituted (215) | 3/0/2/5.71s | 9/2/6/2.88s | 2/1/1/2.72s | 11/9/6/2.56s |
| TLOG all (820) | 4/1/3/4.13s | 17/9/14/2.76s | 4/3/2/2.72s | 21/18/14/2.76s |
| TLOG interior (323) | 4/1/3 | 16/8/13 | 4/3/2 | 21/18/14 |

Paired replay on held-out traces (baseline rules versus final rules on identical engine output; TLOG clean dev reuses its tuning trace):

| set | shipped baseline | shipped v2 | a0w baseline | a0w v2 |
|---|---|---|---|---|
| FF help clean test | 0.035 (1) | 0.035 (1) | 0.104 (3) | 0.104 (3) |
| FF TLOG clean dev | 0.119 (4) | 0.119 (4) | 0.089 (3) | 0.089 (3) |
| FF v1 | 0 | 0 | 0 | 0 |
| Help slips (10) | 1/0/1/5.12s | 1/0/1/5.12s | 1/0/1/5.12s | 1/0/1/5.12s |
| TLOG omitted (551) | 1/1/1/1.80s | 7/6/7/2.76s | 1/1/1/1.80s | 7/6/7/2.72s |
| TLOG repeated (43) | 0/0/0/— | 2/2/2/5.23s | 0/0/0/— | 1/1/1/8.36s |
| TLOG restarted (11) | 0/0/0/— | 0/0/0/— | 1/1/0/3.92s | 2/2/0/3.84s |
| TLOG substituted (215) | 3/0/2/5.71s | 10/3/7/2.92s | 2/1/1/2.72s | 12/10/7/2.70s |
| TLOG all (820) | 4/1/3/4.13s | 19/11/16/2.88s | 4/3/2/2.72s | 22/19/15/2.80s |
| TLOG interior (323) | 4/1/3 | 18/10/15 | 4/3/2 | 22/19/15 |

**FF did not rise.** On every clean set and both models, v2 raises exactly the same issues on the same clips as the baseline.
- Help clean, dev+test: shipped 0.020 (1 issue in 48.79 min, the `unclear_ayah`), a0w 0.061 (3).
- TLOG clean dev: shipped 0.119 (4), a0w 0.089 (3).
- v1: 0 on both.

Help slips did not move: 1/10 on both models, the same word as the baseline. It is `possible_vowel` in the paired replay; the live v2 run labels it `possible_substitution`, which is inference noise, not a rule effect. Nine of the ten located help slips are substitutions.

**What is still missed.** 438 of the 820 held-out slips sit within 0.3 s of a clip edge. Most of those are probably segmentation cuts rather than slips, and leading or trailing words have no live rule because a reciter may start or stop anywhere. On the 323 interior slips, recall is 5% (shipped) and 7% (a0w). On the tuning half, 78 of the 103 interior substitutions shipped still misses end as `ok` (44) or `unsure` (34) in the aligner (wrong 8, skipped 2, not aligned 15). Letting GOP flag those states raised FF in every variant tried.

## Correction rules on acted help mistakes (re-tune)

2026-10-02. Acted takes from tilawa.dev/help (`kind != "none"`) used privately as a labelled correction set. Never published: audio, ids, labels and per-clip outputs stay on the Modal volume (`/help/`, manifest `use="acted"`) and in `/tmp`. Only these aggregates are committed.

**Export (743 rows).** `kind`: none 305 (12 with `extra_mistake`), skip_word 95, substitution 85, vowel 76, skip_ayah 70, repeat 63, tajweed 49. Every acted row has exactly one `mistakes` entry with `ayah` and a 1-based `word` (none for skip_ayah) plus `expected`. Substitutions also carry `said` / `said_from`, vowels `letter` / `from` / `to`, tajweed `rule`. `mechanism` (14 values) and `scenario` (33) are set on 339 rows.

**Split.** Speaker-disjoint. The 74 speakers in the earlier clean-only manifest keep their split (0 moved). The 46 new speakers are split by the same hash order. Dev: 56 speakers; clean 110 (20.7 min), acted 204 (29.9 min). Test: 64 speakers; clean 183 (32.6 min), acted 234 (34.2 min). Acted per kind, dev/test: skip_word 39/56, substitution 39/46, vowel 37/39, skip_ayah 34/36, repeat 33/30, tajweed 22/27. TLOG clean dev (267, 33.7 min) and v1 (53, 17.0 min) have no split and guard both halves.

**Localisation** (`correction_eval.py locate-acted`). Label to phoneme-corpus word: 437/438 (0.998 [0.987, 1.000]). Forced-alignment span: 435. An edit at or next to that word in the free decode of either model: 348/438 (0.795 [0.754, 0.830]); both models 320; exact word 321. Confirmed by kind: skip_word 95/95, repeat 62/63, skip_ayah 67/70, substitution 73/85, vowel 38/76, tajweed 13/49.

**Scoring** (`acted_eval.py`). A flag hits a label on the same ayah within ±1 word (skip_ayah: anywhere in the ayah). Recall covers the five in-scope kinds. Tajweed is reported but left out, because the engine does not grade tajweed. Precision counts every error flag on the split's acted and clean help takes, and a flag that hits no label is a false alarm. Repetition notes are counted separately. CIs: Wilson for P and R, speaker-cluster bootstrap for F1 and paired deltas.

**Tuning (dev only, paired trace replay, 1 thread).** Hard constraint: clean false flags on help clean dev, TLOG clean dev and v1 no higher than the shipped engine (shipped model, old rules). The GOP grid had 2017 rule sets, crossing flag, anchor, states, kinds, persistence, settle and local-min. The other-rules grid had 864, crossing partial omission, settle, repetition mode and gain, and the vowel margins. Findings:
- GOP adds at most +1 dev catch on shipped and +3 on a0w over GOP off. FF is unchanged at `gopFlag` −5 when GOP is gated to wrong words and substitutions only, and kept out of settle. v2's −3 adds a help clean dev flag on a0w.
- Settle adds most of v2's gain (shipped dev 20 → 27 catches). Its 4 dev false alarms are all on a short ayah outside the take, from a lock onto a similar ayah. They look the same as its hits in every verdict field.
- Partial omission doubles skip_word catches (7 → 14 on dev).
- `vowelWordMargin` 0.5 → 0.8 halves TLOG clean dev FF (shipped 4 → 2, a0w 3 → 2) and costs one shipped dev vowel catch.
- Repetition as a note: at gain 5, 5 of 6 dev notes hit a label (shipped) and 6 of 8 (a0w). There are 0 notes on clean sets.
- Structural misses that no threshold reaches. On dev, 26 of 39 substitution and 15 of 39 skip_word labelled words never appear in any verdict snapshot: the tracker follows the reciter into the similar passage or loses lock. Every skip_ayah take skips the middle of 3 ayahs, and the tracker then aligns ayah N+2's audio onto N+1, so N+2 is never matched and the ayah-gap rule fires on 2 of 34.

Frozen re-tuned set: `gopFlag −5, gopAnchor −1, gopOnSkipped false, gopOmission false, settleGop false, vowelWordMargin 0.8, repetitionMode note, repetitionGain 5`. Everything else is as in v2.

**Test (scored once).** Replay on identical traces. Cells are P / R / F1 over 207 in-scope labels, then flags, then clean FF per minute (issues).

| model · rules | P | R | F1 | flags | help clean | TLOG clean dev | v1 |
|---|---|---|---|---|---|---|---|
| shipped · old | 0.895 [0.69, 0.97] | 0.082 | 0.150 [0.08, 0.22] | 19 | 0.031 (1) | 0.119 (4) | 0 |
| shipped · v2 | 0.884 | 0.184 | 0.304 | 43 | 0.092 (3) | 0.119 (4) | 0 |
| shipped · v2 GOP off | 0.944 | 0.164 | 0.280 | 36 | 0.031 (1) | 0.119 (4) | 0 |
| shipped · re-tuned | 0.944 | 0.164 | 0.280 | 36 | 0.061 (2) | 0.059 (2) | 0 |
| shipped · re-tuned GOP off | 0.970 | 0.155 | 0.267 | 33 | 0.031 (1) | 0.059 (2) | 0 |
| a0w · old | 0.885 | 0.111 | 0.197 | 26 | 0.092 (3) | 0.089 (3) | 0 |
| a0w · v2 | 0.920 | 0.217 | 0.352 | 50 | 0.123 (4) | 0.089 (3) | 0 |
| a0w · v2 GOP off | 0.930 | 0.188 | 0.313 | 43 | 0.092 (3) | 0.089 (3) | 0 |
| a0w · re-tuned | 0.953 | 0.198 | 0.328 | 43 | 0.061 (2) | 0.059 (2) | 0 |
| a0w · re-tuned GOP off | 0.950 | 0.184 | 0.308 | 40 | 0.061 (2) | 0.059 (2) | 0 |

Paired F1 vs shipped · old: v2 +0.154 [+0.098, +0.201]; v2 GOP off +0.129 [+0.071, +0.174]; re-tuned +0.129 [+0.063, +0.182]; re-tuned GOP off +0.116 [+0.049, +0.165]. GOP's own contribution: re-tuned vs v2 GOP off on shipped is +0.000 [−0.030, +0.033], and a0w re-tuned vs a0w v2 GOP off is +0.015 [−0.012, +0.046]. The extra shipped help clean flag in re-tuned is a GOP `possible_substitution`, the same flag v2 raises.

Live, same harness as the baseline (1 thread, issues dismissed on sight):

| model · rules | P | R | F1 | exact | median latency | help clean | TLOG clean dev | v1 | RTF |
|---|---|---|---|---|---|---|---|---|---|
| shipped · old | 0.895 [0.69, 0.97] | 0.082 [0.05, 0.13] | 0.150 [0.08, 0.22] | 17 | 3.48 s | 0.031 (1) | 0.089 (3) | 0.059 (1) | 0.060 |
| shipped · new defaults | 0.969 [0.84, 0.99] | 0.150 [0.11, 0.20] | 0.259 [0.16, 0.35] | 30 | 2.96 s | 0.031 (1) | 0.059 (2) | 0 | 0.060 |
| a0w · new defaults | 0.949 [0.83, 0.99] | 0.179 [0.13, 0.24] | 0.301 [0.21, 0.39] | 35 | 3.28 s | 0.061 (2) | 0.059 (2) | 0 | 0.093 |

Paired vs shipped · old (live): shipped new defaults have ΔF1 +0.109 [+0.040, +0.160], ΔR +0.068 [+0.023, +0.104], ΔP +0.074 [0.000, +0.233]. a0w new defaults have ΔF1 +0.150 [+0.088, +0.203].

Per kind on test, shipped new defaults (live), as kind-correct R / F1, then caught any kind / n: skip_word 0.38 / 0.55, 21/56; substitution 0.11 / 0.20, 5/46; vowel 0.03 / 0.05, 1/39; skip_ayah 0.08 / 0.15, 3/36; repeat 0 (1/30 caught by an error flag; the live harness does not record notes, and the replay has 0/30 notes); tajweed 0/27. The baseline was skip_word 9/56, substitution 3/46, vowel 1/39, skip_ayah 3/36, repeat 1/30. Every flag that hits is on the exact word except one repeat. Precision per kind is 1.00 except skip_ayah at 0.75.

Replay versus live: the trace rounds margins to 3 decimals, so a borderline vowel or substitution flag can flip (shipped old TLOG clean dev 4 replay vs 3 live, v1 0 vs 1).

**Listening-check queue A (by-ear tags, re-traced with 1 thread, paired replay).** Precision of the flags on the 29 tagged slips, lenient / strict:

| model · rules | flagged | lenient | strict |
|---|---|---|---|
| shipped · old | 4 | 3/4 | 2/4 |
| shipped · v2 | 18 | 11/17 | 7/17 |
| shipped · v2 GOP off | 6 | 5/6 | 3/6 |
| shipped · re-tuned | 4 | 4/4 [0.51, 1.0] | 3/4 |
| shipped · re-tuned GOP off | 3 | 3/3 [0.44, 1.0] | 3/3 |
| a0w · re-tuned | 11 | 8/10 | 4/10 |
| a0w · re-tuned GOP off | 8 | 6/7 | 4/7 |

Queue B (100): no rule set flags a real slip (0/40). v2 flags 3 not_slip, and the new defaults flag 0 on shipped.

**Call.** Ship v2 with GOP off, `vowelWordMargin` 0.8 and repetition as a soft note. These are the new SDK defaults, and they pass every gate on the shipped model. GOP stays opt-in with the re-tuned gating as its default shape: it now passes the by-ear check, but it adds no test F1 on shipped and adds one help clean false flag. The GOP-off variant of the re-tuned set was named after the test run. GOP off was the pre-declared fallback from the listening check, and every other field was fixed on dev.

Not done: a synthetic set. The dev acted half already has 181 in-scope labels, and the limits above are structural (tracker lock), which splicing cannot probe.

## Tracker in correction mode (acted mistakes)

2026-10-02. Same private acted set, split, labels and scorer as the section above. Aggregates only. Shipped model, correction rules at the current defaults throughout; only the tracker changes. Tracking mode is untouched: every new knob acts only in correction mode, and with all knobs at 0 and no passage the engine reproduces the previous issues and verses on all 634 dev clips.

**Method.** `harness.ts` can record each clip's model log_probs (`ZIPFORMER_LP_CACHE`, `ZIPFORMER_LP_MODE=record`) and replay them with no ONNX. That lets engine variants run on identical acoustic output: a paired replay. Both modes keep the encoder running across idle / dismiss resets, which the live session restarts. `scripts/tracker_correction_eval.py` runs the sets in 4 parallel harnesses (1 thread each). `scripts/tracker_diag.py` classifies each missed label from the per-word verdict states and the tracker timeline. Live runs are not reproducible run to run: two identical full live passes differ on 3 of 183 help clean clips and 1 acted clip, verses included, even at 1 intra-op thread. Paired replay is the headline here, and live is reported for reference.

**Dev diagnosis (replay, current defaults).** "Untracked" means the labelled word never reached a controller snapshot.
- substitution (39): 26 untracked, of which 16 locked only onto another ayah and 10 never locked (fallback only). Of the 11 tracked, 9 read `ok`, 1 `unsure`, 1 `wrong`. 2 caught.
- skip_word (39): 15 untracked (8 no lock, 6 other ayah, 1 late lock), 9 tracked but not flagged, 15 caught.
- The other ayah is usually a near-identical copy (refrains 55:13…, 37:80 / 37:131, 7:111 / 26:36, 79:39 / 79:41). On a 4–7 s single-ayah take nothing in the audio separates them, so this is a missing-information problem, not a threshold one.
- skip_ayah (34): N matched, then N+1 unmatched and N+2 unmatched in 15, N+1 matched (N+2's audio bent onto it) in 10, caught 2. N+2 typically completes only while `stop()` flushes the tail, where the ayah-gap rule did not run.
- The tracked skip_word misses include 6 on an ayah's second-to-last word with no verdict on the last word. The take ended, the open segment stopped at the cursor inside word n−2, and the last word's audio was aligned onto n−2.

**Changes** (correction mode only):
1. `ZipformerSession.setExpected(passage)`. Lock at the passage start after any isti'adha / basmala, search only inside it, no relocation, optional `outsideJumpCost`. The harness passes the manifest passage per clip (`--expected`); TLOG clean dev has none.
2. Skipped-ayah check. Heard chars on interior ayah A that fit A+1 (semi-global distance ≤ 0.35, ≥ 0.25 better than A, ≥ 10 chars, A+1 reading starts within the first 10% of them) raise `possible_skipped_ayah`. With a passage, the ayah-gap rule also runs at stop.
3. `anchorAyahEnd` 2. Settled verdicts realign through the ayah end when the cursor stopped within 2 words of it.
4. No passage: stop-time alignment of a take that never locked (`stopAlignDistance` 0.35, a detached tracer that never feeds tallies), with a mid-ayah lock started at the ayah's first word (`backfillRatio` 1.5).

**Dev tuning (replay; FF = help clean dev / TLOG clean dev / v1 issues).**

| step | P | R | F1 | FF |
|---|---|---|---|---|
| current defaults | 0.844 | 0.149 | 0.254 | 0 / 2 / 0 |
| + passage lock | 1.000 | 0.215 | 0.355 | 0 / 2 / 0 |
| + gap rule at stop | 1.000 | 0.221 | 0.362 | 0 / 2 / 0 |
| + skipped-ayah check | 1.000 | 0.249 | 0.398 | 0 / 2 / 0 |
| + anchorAyahEnd 2 (final, passage) | 1.000 | 0.276 | 0.433 | 0 / 2 / 0 |
| final, no passage | 0.829 | 0.188 | 0.306 | 0 / 2 / 0 |

Rejected or neutral on dev:
- `skipMaxHead` 0.25: +1 help clean FF. A free-ended A+1 match hides inside "badly decoded A, then A+1".
- Prefix-anchored skip fit: no gain at FF-safe thresholds.
- `substitutionDistance` 0.5: +4 catches, +1 help clean FF. 0.55 gave +1 catch next to that cliff, so 0.6 stays.
- `outsideJumpCost` 12: no change; the takes are too short for in-surah jumps.
- `anchorAyahEnd` 3: same as 2.
- `backfillRatio` alone: no change.
- The skip thresholds sit on a plateau: R 0.260–0.282 over maxDistance 0.3–0.45, margin 0.15–0.3, head 0.05–0.15, all at 0 FF.

**Dev after (final, passage).** substitution untracked 26 → 5. The remaining 24 tracked misses read `ok` 9, `unsure` 8, `wrong` 7; 5 of the `wrong` ones have clear neighbours and distance 0.54–0.59, and the substitution rule needs 0.6. skip_word untracked 15 → 7. skip_ayah caught 2 → 8.

**Test (scored once; paired replay).** CIs: Wilson for P and R, speaker bootstrap for F1.

| config | P | R | F1 | exact | median latency | help clean FF | TLOG clean dev FF | v1 FF |
|---|---|---|---|---|---|---|---|---|
| shipped engine (old rules) | 0.895 [0.69, 0.97] | 0.082 [0.05, 0.13] | 0.150 [0.08, 0.22] | 17 | 3.52 s | 0.031 (1) | 0.119 (4) | 0 |
| current defaults | 0.970 [0.85, 0.99] | 0.155 [0.11, 0.21] | 0.267 [0.18, 0.35] | 30 | 3.01 s | 0.031 (1) | 0.059 (2) | 0 |
| new tracker, no passage | 0.955 [0.85, 0.99] | 0.203 [0.15, 0.26] | 0.335 [0.25, 0.42] | 40 | 2.88 s | 0 | 0.059 (2) | 0 |
| **new tracker + passage** | **0.983 [0.91, 1.00]** | **0.280 [0.22, 0.34]** | **0.436 [0.33, 0.52]** | 56 | 2.95 s | 0.031 (1) | 0.059 (2) | 0 |

- Paired ΔF1 against the current defaults: +0.169 [+0.116, +0.227] with a passage, +0.068 [+0.035, +0.107] without. ΔR: +0.126 [+0.084, +0.174] and +0.048 [+0.025, +0.075].
- Against the shipped engine: +0.286 [+0.195, +0.370] and +0.184 [+0.113, +0.243].
- Every hit is on the exact word.

Per kind, test, new tracker + passage. Kind-correct R / F1, then caught / n (current defaults in brackets):
- skip_word: 0.61 / 0.76, 35/56 (21)
- substitution: 0.20 / 0.33, 9/46 (5)
- skip_ayah: 0.31 / 0.46, 11/36 (3), P 0.92
- vowel: 1/39 (1)
- repeat: 0 kind-correct, 2/30 caught by an error flag (2)
- tajweed: 0/27

Live, one pass each (1 thread, issues dismissed on sight), as P / R / F1, then help / TLOG clean dev / v1 issues:
- shipped engine: 0.895 / 0.082 / 0.150; 1 / 4 / 2
- current defaults (two passes): 0.912 / 0.150 / 0.257 and 0.941 / 0.155 / 0.266; 3 / 2 / 0 and 2 / 2 / 0
- new, no passage: 0.955 / 0.203 / 0.335; 0 / 2 / 0
- new + passage: 0.967 / 0.285 / 0.440; 2 / 2 / 0
- The two live help clean flags with a passage are both `possible_skipped_ayah`. In one (83:35), the transcript runs from the end of 83:34 straight into 83:36 with nothing of 83:35, which looks like a real skip in a take labelled clean (not verified by ear). The other (23:40) is false: 40 was partly heard, and the check fired after a dismiss reset. Replay, which has no reset, raises neither.
- RTF 0.059–0.068 for every config.

**Gates.** Clean FF is no worse than the shipped engine on any set in replay. In live, help clean is 2 against the shipped engine's 1, with one of the 2 a likely real skip; TLOG clean dev and v1 are better. Tracking mode: v1 SeqAcc 53/53 and held-out multi 56/58, unchanged. vitest: 125 passed, plus 1 failure that is pre-existing on this machine (the int8 `runner.node` margin is 2.5e-4 off the dump, tolerance 1e-4, and fails the same with the change stashed). `test:browser` 6/6, `test:correction` 0 flags on 8 runs plus silence, demo build ok.

**Headline.** The main app does not know the passage in correction mode, so its path is "new tracker, no passage": test F1 goes from 0.267 to 0.335 and recall from 0.155 to 0.203, with clean false flags no higher than the shipped engine. `setExpected` is opt-in for hosts that know the passage (0.436).

**Next ideas.** Auto-infer the expected passage from the first confident lock (the session calls `setExpected` on itself after N seconds), to recover most of the passage gain for free recitation.

Bug found while checking order independence on test: the session kept its skipped-ayah candidate across `reset()`. Fixed (a reset bug, not a re-tune), and all test numbers above are after the fix.

## Similar-verse check (acted mistakes; KILLED on the unseen-word gate)

2026-10-02. Same private acted set, split, labels and scorer as above; aggregates only. Main `04337ab` engine at defaults (slip head off = rules only), paired replay on recorded log_probs for a0w-ep1-a0.5 int8 and the shipped int8. Lab-only prototype (`scripts/similar_verse.py`, `scripts/similar_verse_eval.py`); no SDK change.

**Rule.** Nothing is fit on labels. Index: for every ayah, up to 8 other ayahs sharing ≥3 words and ≥50% of its words in a word-level alignment (diacritic-free keys); 1,803 ayahs have look-alikes (5,067 pairs). Per take, the free-decode phonemes are aligned (reference-global, query ends skippable at 0.5/char, SDK phoneme cost table) to the passage, and to the passage with one region of one ayah swapped for a look-alike's wording ("lookalike"), with 1–2 interior words dropped ("drop"), or with an eye-skip between a repeated phrase ("eyeskip"). A variant that beats the canonical text by `margin` cost units, whose region plus one word each side aligns at ≤ `loc` cost per char and whose ayah aligns at ≤ 0.5, flags the region's first word (`possible_substitution` / `possible_omission`), at most one per ayah, and only when no engine issue sits within ±1 word. Without a passage the slot's alternatives are the tracker's verse and its look-alikes, and the variant must also beat every pure reading of that slot. The flag time is when the region plus one following word has been decoded.

**Dev (tuning only).** Two guards came from dev clean false flags: takes whose audio is not the expected ayah (garbage alignment favours shorter variants) and skipped ayahs; the local-fit and ayah-fit checks remove both. Eyeskip never added a catch. a0w + passage, look-alike margin 3–4: substitution 11 → 20/39 with 9/9 new flags correct; drop margin 8–10 at loc ≤ 0.25: skip_word 19 → 25–26/39; FF unchanged (1 / 2 / 0). Without a passage, look-alike adds 0 (a swapped-in look-alike word on a single-ayah take reads as the other ayah) and drop adds skip_words only. Drop margin 8 cost shipped + passage one help clean dev flag, so 10 was fixed. Frozen before test: lookalike + drop, margin 3.5 / drop 10, one-word swaps, loc 0.25, ayah fit 0.5.

**Test (scored once).** P / R / F1 over 207 in-scope labels; FF is clean issues on help clean test / TLOG clean dev / v1.

| config | P | R | F1 | ΔF1 [95%] | substitution | skip_word | new flags correct | FF base → new |
|---|---|---|---|---|---|---|---|---|
| a0w + passage | 0.980 → 0.988 | 0.232 → 0.382 | 0.375 → 0.551 | +0.176 [+0.114, +0.231] | 9 → 28/46 | 23 → 35/56 | 31/31 | 1/2/0 → 1/2/0 |
| a0w, no passage | 0.950 → 0.942 | 0.184 → 0.237 | 0.308 → 0.378 | +0.071 [+0.036, +0.108] | 5 → 5 | 22 → 33 | 11/12 | 1/2/0 → 1/2/0 |
| shipped + passage | 0.967 → 0.975 | 0.285 → 0.382 | 0.440 → 0.549 | +0.108 [+0.067, +0.150] | 8 → 27 | 35 → 36 | 20/20 | 2/2/0 → 2/2/0 |
| shipped, no passage | 0.955 → 0.957 | 0.203 → 0.217 | 0.335 → 0.354 | +0.020 [0.000, +0.040] | 4 → 4 | 31 → 34 | 3/3 | 0/2/0 → 0/2/0 |

Seen / unseen (a test label is unseen when no acted-dev label of its kind names the same word). a0w + passage: substitution seen 4 → 23/34 (+19), unseen 5 → 5/12 (+0); skip_word seen 20 → 31/50 (+11), unseen 3 → 4/6 (+1). **Unseen extra = 1 of 18 < 3: kill.** shipped + passage: unseen +1 (skip_word). Median latency of the new hits 2.2 s (a0w) / 2.7 s (shipped) after the label's start, against 3.3 s / 2.9 s for engine hits.

**Why the gate cannot move.** All 12 unseen test substitutions are ordinary word swaps on ayahs that have no look-alike anywhere in the Quran (index coverage 0/12). All 30 seen `similar_passage` substitutions borrow their word from an indexed look-alike. The help prompts that elicit look-alike slips repeat across speakers, so every such word is a dev label too. The rule has no learned parameters beyond five dev thresholds, so the seen gain is not lexical memorisation in the slip-head sense, but the thresholds were chosen on dev takes of the same verses, and this split cannot show transfer to new verse pairs. The unseen skip_word drops sit at margins 5–14; the frozen 10 takes two (one already an engine catch).

## Ayah-order check (acted skip_ayah; KEEP on a0w, KILL on shipped emissions)

2026-10-02. Same private acted set, split, labels and scorer; aggregates only. Main `04337ab` engine at defaults (rules only), live 1-thread runs recording the free-decode tokens. Lab-only (`scripts/ayah_order.py`, `scripts/ayah_order_eval.py`); no SDK or tracker change.

**Rule.** Nothing is fit on labels. Cut the free-decode token stream at pauses (≥ 12 blank frames = 0.48 s). Align each segment to every contiguous word span of a window of ayahs (tracker's first verse −1 .. +4, or the expected passage), or call it isti'adha/basmala, or garbage at 0.5/char. Chain the segments twice: in order (restarts cost 2, forward gaps cost 0.5/char but may not contain a whole ayah) and free (whole-ayah jumps cost 0.5). Flag `possible_skipped_ayah` on the jumped ayah when the free chain wins by ≥ 0.04 per char of the post-jump segment, both flanking segments fit ≤ 0.3/char, ≤ 8 unexplained chars sit between them, and the ayah is not read later.

**Pre-declared.** No-passage skip_ayah 8/36 → ≥ 16/36. Kill if < +3 or FF rises on help clean test or v1.

**Test (scored once).** FF = clean issues help clean test / v1 / TLOG clean dev (TLOG provisional).

| config | skip_ayah | all 207 | U_test 62 | new flags correct | FF base → new |
|---|---|---|---|---|---|
| a0w, no passage | 8 → 21/36 | 37 → 50 | 17 → 30 | 13/13 | 1/0/2 → 1/0/2 |
| a0w + passage | 14 → 26/36 | 48 → 60 | 23 → 35 | 12/12 | 2/0/2 → 2/0/2 |
| shipped, no passage | 3 → 9/36 | 42 → 48 | 12 → 18 | 6/6 | 0/0/2 → **2**/0/2 |
| shipped + passage | 12 → 17/36 | 59 → 64 | 21 → 26 | 5/5 | 2/0/3 → **4**/0/3 |

Other kinds are unchanged. New-hit latency is 3.3 s on a0w and 4.0 s on shipped after the label start. With the similar-verse check on as well, a0w + passage goes 48 → 91/207 and a0w without a passage 37 → 61/207, with help clean / v1 FF equal to rules-only. **Caveat:** 31/36 test skip_ayah labels sit on dev-labelled prompts, and every new catch is on those; the 5 prompt-disjoint labels gain 0 (4/5 already caught). The two shipped clean flags are decode-driven: a partial ayah read as a restart, and an ayah decoded as garbage before a cheap jump. a0w raises neither. Two textually identical neighbour pairs (94:5/6, 109:3/5) make 8/34 dev labels uncatchable by text.

## Per-experiment notes

**c2c-direct-mixed-tta** — Cyberistic's winning entry and current champion. It runs the mixed int4+int8 FastConformer ONNX once at 1.0x speed, skips augmentation for confident predictions, and only runs 0.9x/1.1x speed-perturbed passes on low-confidence samples. Reproduced locally over 3 runs at 100% recall, 100% precision, and 100% sequence accuracy on v1 (53 samples), with 0.84s average latency.

**c2c-direct-mixed** — Same CTC re-rank algorithm without TTA, using `web/frontend/public/fastconformer_full_mixed.onnx` (88 MB). This is the model now loaded by the browser worker. Reproduced at 98% recall / 98% precision / 98% sequence accuracy on v1 at 0.72s average latency; TTA recovers the remaining miss.

**zipformer-ctc** — Streaming Zipformer2-CTC phoneme model (v3.1 base weights, NPL-1.2) run through our own recitation engine (`packages/core/src/recitation/`); benchmark wrapper in `experiments/zipformer-ctc/`; runner alias `prompter-zipformer` keeps historical result JSON names. Key finding: Streaming Zipformer2-CTC over a 251-token tajweed-phoneme vocab (letters+harakat, madd length as repetition), greedy CTC, then a whole-Quran 5-gram phoneme index + graded-cost semi-global alignment to locate, and a per-surah online DP tracker with per-word verdicts. The harness runs a recognize-mode host loop (search → track → verdicts, re-search after silence) and emits ayahs with ≥50% ok/unsure words; because the live engine refuses to lock on short clips, a whole-ayah nearest-match fallback over the same phoneme distance handles clips where no lock happened (pure engine: 74% v1; with fallback: 100%). Deterministic across runs (no ±3–6 sample jitter). **v1 53/53, v2 42/43, v3 247/256 (96.9% / 96.7% / 96.5%), qlab 571/583 (97.9%)** on the v3.1 base. Native MIT harness + interp-gentle-a0.5 (2026-09-17): **v1 53/53, v2 43/43** (`benchmark/results/2026-09-17_065319.json`, `2026-09-17_065532.json`). vs champion `c2c-direct-mixed-tta` at 241/256 (94.8% / 94.9% / 94.1%) on the same v3 run — it takes all 8 Husary multi-verse samples the champion loses. Remaining v3 misses are textually identical/near-identical ayahs (`55:53→55:13`, `81:19→69:40`, `37:82→26:66`, `30:1→2:1`, `26:122→26:9`, `10:43→10:42`), plus one short crowd clip (`107:1→106:4`) and one over-run (`100:1` → `100:1,100:2`). Takeaways for our stack: (1) a phoneme alphabet that encodes harakat + madd length gives the matcher far more discriminative chars per second than BPE text; (2) locate-then-track with graded substitution costs beats per-chunk `matchVerse()` on multi-verse; (3) the shipped 69 MB int8 is already the v3.1 base checkpoint — fine-tune that, don't train from scratch. Runs at ~5% RTF single-threaded CPU.

**ctc-alignment** — CTC forced alignment with `jonatasgrosman/wav2vec2-large-xlsr-53-arabic` (1.2 GB). Scores verses directly against frame-level logits via the CTC forward algorithm, skipping greedy-decode information loss. Too large (6×) and too slow (5×) for on-device.

**nvidia-fastconformer** — `nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0`. Best speed/accuracy/size balance for streaming. A fine-tune sweep (v1, v2a, v2b, v3c) failed to beat the zero-shot baseline.

**fastconformer-ctc-rescore** — Two-stage: FastConformer ASR + CTC re-score top-50 candidates with the fine-tuned 8L Rabah head. Re-scoring doesn't recover failures — both models miss the same hard cases (short isolated letters, multi-verse).

**fastconformer-nbest-bruteforce** — N-best beam search + CTC brute-force. Regressed vs baseline: beam candidates without an LM are near-identical. A Quran-specific LM or constrained decode would be needed.

**fastconformer-lm-fusion** — FastConformer + pyctcdecode Quran LM. Best batch SeqAcc (94% v1, 95% v2) but too much added latency for streaming and awkward in-browser.

**fastconformer-phoneme** — Fine-tuned FastConformer CTC head on a 69-phoneme Buckwalter vocab. Former shipped ONNX model (`fastconformer_phoneme_q8.onnx`, 131 MB), now kept for historical experiments and streaming-regression comparisons. Trained on 71K Iqra + 55K TTS + 1.8K RetaSy + ~18K filtered TLOG.

**w2v-phonemes** — Phoneme CTC + Levenshtein matching. `large-int8` (r7, 970 MB INT8 ONNX) hits **100% batch on v1 and 96.1% / 96.1% / 96.1% (recall/precision/SeqAcc) on v3** — the strongest batch oracle we have, but 1 GB is too large to ship to browser. `base` (r15_95m, 388 MB fp32) is now accessible with Ahmed's read token and hit **97.1% / 97.1% / 97.1%** on the downloadable EveryAyah slice of v3 (174 samples, avg 0.90s CPU, result `benchmark/results/2026-04-29_091708.json`). A Modal-exported local dynamic-int8 ONNX (`base-local-int8`, 118 MB, artifacts in `data/r15-onnx/` or Modal volume `w2v-phonemes-r15`) preserved the same **97.1% / 97.1% / 97.1%** on that slice, with avg CPU latency 1.11s (result `benchmark/results/2026-04-29_100633.json`), and scored **96.0% / 96.1% / 95.7%** on full v3 (256 samples, avg 0.89s CPU, result `benchmark/results/2026-04-29_103225.json`). Both fp32 and int8 fail the same five EveryAyah short/repeated-phrase collisions (`55:53→55:13`, `81:19→69:40`, `37:82→26:66`, `30:1→2:1`, `26:122→26:9`), so the remaining batch error is mostly context/ambiguity rather than acoustic quality. A phoneme-aware naive chunked baseline (`predict_streaming`, 3s independent chunks) scored only **20.6% / 12.2% / 3.9%** on full v3 (result `benchmark/results/2026-04-29_103627.json`), confirming r15 is a batch/verifier model, not a true streaming model. O(T²) wav2vec2 attention and independent chunk CTC collapse are the blockers; use r15/r7 as verifier/teacher while true streaming should be cache-aware FastConformer RNNT/CTC. As of 2026-04-22 `_decode_phonemes` chunks audio >25s into 25s windows with 1s overlap, each independently CTC-collapsed then concatenated — without chunking, a single 200s sample bloats memory to 22 GB and effectively hangs on Apple Silicon's ArmKleidiAI MatMul path. Upstream `base-int8` (`hetchyy/r15_95m_onnx_int8`) still returns 404 on HF; our local `base-local-int8` entry is shown only when `data/r15-onnx/model_int8.onnx` or `R15_ONNX_DIR/model_int8.onnx` exists. HF token required for fp32.

Use case: r7 remains the highest-accuracy distillation teacher; r15 is now a plausible server-side/batch verifier and quantization candidate if a real int8 export can be produced.

**tadabur-whisper-small** — Best Whisper fine-tune we tested. Highest streaming recall (87% v1) at 3× FastConformer latency.

**rabah-pruned-ctc** — Layer-pruned Rabah CTC; see deep-dive above.

**two-stage** — Moonshine Tiny Arabic (103 MB) for fast ASR + CTC re-score on top 50 candidates, falling back to a large CTC. Blocked on the small CTC model.

**whisper-lora / whisper-small** — Whisper-small base + optional LoRA. LoRA helps vs base; both trail FastConformer, especially streaming.

**distilled-ctc (failed)** — wav2vec2-base knowledge-distilled from a large CTC teacher. English-only pretraining means no usable Arabic speech features.

**contrastive / contrastive-v2 / embedding-search (failed)** — All three failed for the same reason: English-pretrained audio encoders (HuBERT, wav2vec2-base) don't produce useful features for Arabic.

## Key findings

1. **FastConformer dominates for streaming.** Best speed/accuracy/size tradeoff across every viable experiment.
2. **CTC forced alignment is the most accurate batch approach**, but too large (1.2 GB) for on-device.
3. **ASR quality is the bottleneck.** All ASR-based approaches fail on the same samples.
4. **English-pretrained audio encoders fail on Arabic.** wav2vec2-base, HuBERT, Moonshine can't produce useful features.
5. **Pruning + fine-tuning works.** 24→8 layers with `first_n` pruning + CTC fine-tuning recovers most accuracy (75% at 145 MB).
6. **Short verses are hard across all approaches** — under 3–4 words doesn't give enough signal.
7. **Matching quality matters more than decode strategy.** Multi-pass phoneme matching takes Python batch from 79%→90% v1. pyctcdecode beam is worse than greedy for this model.
8. **Beam-candidate injection into the tracker regressed.** The verse/span trie (1.7M nodes, 2.2ms decode) works correctly, but beam-matched verses override correct greedy results. Surah-level expansion is the safer next step.
9. **TLOG: one quality-filtered bucket wins.** ~18K filtered at 0.3 is the sweet spot; more volume, lower filter, no TLOG, or combined data changes all regress.
10. **Streaming precision had a cascade bug.** Auto-advanced `verse_match` messages emitted without audio evidence. Deferred emission (2026-04-11) fixes it: +13pp precision, +20.8pp SeqAcc on v1.
11. **Cyberistic's text CTC rerank moved the batch ceiling.** `c2c-direct-mixed-tta` is now the v1 champion at 100% / 100% / 100% with an 88 MB ONNX. The previous v4-tlog phoneme model remains useful as a historical streaming baseline, but new shipped runtime work should start from `fastconformer_full_mixed.onnx`, `vocab.json`, and `quran_ctc_tokens.json`.
12. **r7 (Ahmed's 1B wav2vec2 phoneme CTC) is still a strong v3 batch oracle.** 96.1% / 96.1% / 96.1% on v3 (256 samples, full-file batch), but 1 GB is too large to ship and wav2vec2 attention is not streaming-friendly. Use r7/r15 as teacher/verifier candidates, not the browser runtime.
13. **v3 SeqAcc is mostly a tracker state problem, not a recognizability problem.** Exact-match diagnostics (`web/frontend/test/analyze-v3-stability.ts`) show the v3 gap is dominated by extra emissions: cached streaming exact-fail runs include 124 `extra_after_expected` and 29 `wrong_surah_jump` cases across 768 runs. Comparing those cached streaming outputs against the r7 batch oracle (`web/frontend/test/compare-streaming-oracle.ts --stability-json=... --oracle-results=benchmark/results/r7-v3-batch.json`) shows the first long/medium exact-fail samples are `streaming_tracker_loss`: r7 predicts the exact expected verse while streaming emits expected+extras. The old phoneme ONNX full-file path was too weak to serve as this oracle; it often missed the expected verse on those same long clips. Two tempting runtime invariants were falsified and reverted: consuming the buffer after evidence-backed stale exits, and blocking selected candidates dominated by the current fusion leader. The next tracker attempt needs explicit segment ownership / active-hypothesis comparison, not score-threshold or rank gates.
14. **The v3.1 base model ships as onnxruntime dynamic-int8; int8 and fp32 score identically through our tracker, so int8 (66 MB) is the shipping artefact.** Tracker scores are deterministic; v3.1 vs v3 only swap one v1/v2 crowd clip. Fine-tune v3.1 rather than training Zipformer CTC from scratch.
15. **A 5-epoch full-mix fine-tune of v3.1 at lr 0.005 dropped tracker SeqAcc even as CTC valid loss fell 0.088→0.032 and ONNX PER 5.56%→4.37%.** ft-v31: v1 53→45, v3 247→231, qlab 571→566 (tlog −6, nufais +1). Failures are mostly multi-ayah truncations, not wrong-surah. Lower LR / fewer epochs / freeze encoder next — do not ship this checkpoint.
16. **Fine-tuning v3.1 on isolated-ayah data lowers PER but breaks multi-ayah tracking.** Epoch sweep (fp32): v1 53→46→44→45 and v3 247→225→214→231 at init/ep1/ep2/ep5. CTC skips short connecting ayahs (acoustic); `MIN_WORD_FRACTION=0.5` plus `_contiguous_head` then report only the prefix (matcher). **E1 B1 windows repair that failure** (ft-v31 9/21 multi → ft-multi ep2 19/21) but a reduced studio-heavy mix wrecks tlog (194→174). **E4** put the same windows in the full mix at mux ×2.5 (~27% of hours) + E2's gentle LR: raw multi stays broken (11/21) because qua still dominates; α=0.5 blends (interp-mf1-a0.5 / interp-mf2-a0.5) **tie** interp-gentle-a0.5 on 53/43/248/572 with an identical miss set and worse PER (4.27% vs 4.09%). α=0.5 of a mild FT is an attractor, not a knob that stacks data recipes. interp-gentle-a0.5 stays the promoted candidate. Matcher-only (E3) recovered 0 misses.
17. **Gemini 3.1 Pro oracle on remaining Zipformer misses: 0 genuine model errors.** Independent `gemini-3.1-pro-preview` transcripts of the 21 reference/interp-gentle misses (2026-09-16): 14/16 confusable pairs are verbatim-identical in `quran.json` and undecidable from audio. v3 gold has two confirmed mislabels (`tlog_m043_010_043`, `tlog_m044_010_043` are 10:42) plus an incomplete span (`tlog_m008_107_001` is 106:4 then 107:1); qlab `qul_alnufais__8_51` uses بظلام (3:182 wording). Effective ceiling is **251/256** v3 (not 249) and **≈573–574/583** q-lab without context priors — further gains need the tracker's `hint` (previous ayah / surah continuity), not more isolated-ayah FT. Manifests unchanged; see `artifacts/gemini_oracle/misses_oracle.json` and `benchmark/test_corpus_v3/KNOWN_LABEL_ISSUES.md`.

## Methodology

- **Batch:** experiment's `transcribe()` processes the full audio file. `StreamingPipeline` matches transcript against all 6,236 verses via Levenshtein. Per-sample R/P/SeqAcc, averaged.
- **Python streaming:** 3s chunks, independent transcription per chunk, accumulated text fed to `VerseTracker` for progressive matching.
- **Browser/RN streaming:** `RecitationTracker` feeds 300ms chunks through ONNX with a 4s silence tail to flush discovery. Current runtime uses Cyberistic's raw-audio `fastconformer_full_mixed.onnx`; older stability artifacts before the swap used `fastconformer_phoneme_q8.onnx`.
- **Latency:** wall-clock per sample, excluding first-sample warmup. Apple Silicon (CPU).
- **Variance:** ONNX inference is non-deterministic at ±3–6 samples/run on v1. Always report medians over 3 runs (max).

Raw JSON results live in `benchmark/results/`. Stability JSON from streaming runs lives in `web/frontend/test/*-stability.json`.

## Roadmap

Designs in `docs/plans/` for the work remaining between 78.6% streaming recall and the 95% target:

- **Curriculum / hard-example fine-tune (v7)** — start from v4-tlog, short low-LR second stage weighted by current failure buckets: short/noisy RetaSy, huruf-muqatta'at openers, clipped-start TLOG.
- **Streaming-like augmentation** — explicit start/end truncation, mild reverb, random short-window crops, adjacent-ayah concatenation. Current augmentor only has speed/gain/noise/shift/silence; the model never sees what streaming actually produces.
- **Phoneme n-gram anchoring in the browser matcher** — port rare-phoneme voting from `experiments/w2v-phonemes/` into `quran-db.ts` for surah-level expansion when `ratio()` is weak.
- **Teacher distillation (w2v-phonemes/large → FastConformer)** — use the 100%-batch teacher to generate soft labels. The earlier failed distillation used English wav2vec2-base as the student; that's what falsified, not the distillation idea.
- **Segment-aware tracker state** — replace implicit "current rolling buffer" ownership with explicit audio segments / active verse hypotheses. Diagnostics show stale exits after real word/acoustic progress can replay already-assigned audio through open discovery, causing expected+extra cascades. A safe fix should compare rediscovery candidates against the active verse/segment hypothesis before emitting, rather than relying on elapsed-time lockouts, rank gates, or buffer clearing.
- **Deferred A4 — gated trie beam candidate expansion** — expand the candidate surah set (don't inject direct candidates). Beam infrastructure already wired in `inference.ts`. A partial beam-derived surah-expansion probe was reverted because uncalibrated beam hints still pushed wrong-initial/wrong-surah paths; any future beam hint must first prove calibration against diagnostics.
