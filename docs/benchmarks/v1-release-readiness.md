# Tilawa V1 release readiness (lane G)

Date: 2026-11-11. Evaluation only: no listener, tracker or checker code, threshold or bar was changed, nothing was folded or discarded, and autolab was not touched.

## Candidate

| Item | Identity |
|---|---|
| Engine | `441dddc3` (research `main`, frozen in `.scratch/tilawa-v1-release/freeze.md`) |
| Listener, full | `spliced-clean@1-int8`, sha256 `714e98addee73b1b8781ad7d99b6d3c745acd9116638baf4e12891f53a15af1b` |
| Listener, lite | `spliced-clean-lite@1-int8`, sha256 `4974de78d3c35d2542e7a24d99b747e1160d6e31e366fb68492b74a789890f1b` |
| Reference | `ref-v2` |
| Scope | Tracking plus the 4 memorization classes (wrong, omitted and added words, skipped passages). The SDK reports the 11 pronunciation classes as `unsupported`. |

## Verdict

**Not ready for a stable V1.** 7 of the 13 scoreboard gates pass and 6 fail. The 5 `BLOCKED` requirements have no data yet. The candidate also has a **reproducible session crash** when a recitation reaches the end of An-Nas. With that crash fixed (a separate, recorded checker experiment, then a re-freeze), it can ship as a **labelled preview** of tracking plus memorization feedback, with the measured false-warning rates stated (see [Prerelease claims](#prerelease-claims)). As frozen, `441dddc3` should not be published even as a preview.

## Evidence and method

- **Decodes reused only after re-verification.** The candidate's streaming listener (`createListener`, onnxruntime-node 1.30.0, 20 ms pushes, posteriors on) re-decoded a sample of every evidence set used. Each matched the stored file byte for byte (phonemes and top-5 posteriors): help eval clips 10/10 (`.scratch/sdk-help/spliced-clean-int8-node-lw`), tlog lead-in and lead-out 10/10 (`.scratch/leadin/evidence/spliced-clean-int8-lw`), stitched-session tlog clips 12/12 (`.scratch/jumps/evidence-int8`), and QUD recordings 2/2 (`.scratch/session-bench/qud/evidence`). The listener, fbank and ORT lockfile have not changed since 2026-10-07. The lab's `spliced-clean-int8` evidence (decoded on Modal Linux, `decode-v1.json`: same model hash, `pnpm --filter @tilawa/sdk decode`, before the 2026-11-01 gap re-decode) differs from a Mac re-decode only at near-tie frames: 3 of about 1,840 phonemes differ in lab-v1.5-p05, each a one-frame token at confidence about 0.5. That is platform floating point, not a different decode, so no re-decode was needed.
- **Scoreboard row** `441dddc3 × spliced-clean-int8` (`scoreboard.json`/`.md`): the CPU replay of those decodes through the frozen tracker and checker. `stitched-warnings-v1` was re-run at `441dddc3` for both listeners. Latency is deliberately left off the row. The scoreboard would count a desktop latency report as data for the device-latency `BLOCKED` item, and only desktop numbers exist.
- **SDK parity.** `pnpm bench:help`, which runs the shipped session API end to end, reproduces the replay exactly on the 388 help clips: declared 56 TP / 11 FP / 8 FN, and free (no lead-in) 30 / 10 / 34 (`v1-release/sdk-help-spliced-clean-int8-441dddc3.json`).
- **Lab score of the shipped path** (Modal run `laneg-v1-int8-441dddc3-20261111T131243Z`, 200 CPU shards; summary in `v1-release/lab-v1.5u-spliced-clean-int8-441dddc3.json`). This is lab-v1.5u at `441dddc3`, scored on the **plain streaming int8 evidence**. The ledger's main line instead uses `exp-listener-teacher-insert-marginal-gate-ad6eb8c0` evidence, which includes the fp32 insertion teacher and the post-hoc gap re-decode. V1 does not ship the teacher. The SDK applies the gap re-decode only to `session.evidence()` after `end()`, never to the live checker. So the main-line numbers (practice 24, exam 109 wrong calls) are not what the V1 SDK emits. The V1 numbers are practice 35 and exam 108.
- **QUD** is a regression report only; see [QUD regression](#qud-regression).

## Gate table

Bars are the scoreboard's (`GATES` in `packages/core/src/bench/scoreboard.ts`), unchanged. Evidence paths are under `docs/benchmarks/` unless marked `.scratch` (main checkout, private).

| # | Gate | Value at 441dddc3 | Bar | Status | Evidence | Δ vs last measured int8 row (ed956b4) |
|---:|---|---:|---|---|---|---|
| 1 | False warnings per 5 min, lead-out recitation | 1.69 (18 in 53.2 min, 354 lead-outs) | ≤ 1 | **fail**, 69% short | scoreboard row; `.scratch/leadin/evidence/spliced-clean-int8-lw` | 2.73 → 1.69 |
| 2 | False warnings per 5 min, stitched sessions | 1.35 (200 in 743 min) | ≤ 1 | **fail**, 35% short | `stitched-warnings-v1.json` (re-run at 441dddc3) | 1.46 → 1.35 |
| 3 | Listener gold PER | 4.82% (233 clips) | ≤ 3.7% | **fail**, 32% short | `gold-per-v1-scores.json` (listener unchanged, decode re-verified) | = |
| 4 | Tracker: correct ayah within 5 s (help clean clips) | 84.0% (189/225) | ≥ 95% | **fail**, 12% short | scoreboard row | = |
| 5 | Free recitation with lead-in/out: warnings correct | 81.2% (56/69) | ≥ 90% | **fail**, 10% short | scoreboard row | 77.9% → 81.2% |
| 6 | Memorization session (declared passage): warnings correct | 83.6% (56/67) | ≥ 90% | **fail**, 7% short | scoreboard row; `bench:help` | 87.5% → 83.6% |
| 7 | Free recitation with lead-in/out: mistakes caught | 87.5% (56/64) | ≥ 80% | pass | scoreboard row | 82.8% → 87.5% |
| 8 | Memorization session: mistakes caught | 87.5% (56/64) | ≥ 80% | pass | scoreboard row | = |
| 9 | False warnings per 5 min, short clips | 0.76 (7 on 225 clean clips, about 46 min) | ≤ 1 | pass | scoreboard row | 0.32 → 0.76 |
| 10 | False warnings on repeats | 0 (23 repeat clips) | ≤ 0 | pass | scoreboard row | = |
| 11 | False warnings per 5 min, lead-in recitation | 0.54 (6 in 55.4 min) | ≤ 1 | pass | scoreboard row | = |
| 12 | Tracker: false jumps (help clean clips) | 0.0% (0/225) | ≤ 2% | pass | scoreboard row | = |
| 13 | Real-time factor (tracker + checker replay, this machine) | 0.0018 | ≤ 0.05 | pass | scoreboard row (`perf.ts`) | = |

| # | `BLOCKED` requirement | Status | What is needed to measure it |
|---:|---|---|---|
| B1 | Word highlight / warning latency on device | **unmeasured on device**. Desktop only (Apple M4, onnxruntime-node, 20 ms pushes, 388 help clips): highlight p50 332 / p95 468 ms (bar ≤ 600), suspected p50 291 / p95 1,079 ms (bar about 1 s). Lite: 462 / 1,130 ms p95. | Device-floor runs: 2020 mid-range Android (Snapdragon 7xx, 4 GB) with `lite` on WASM SIMD without WebGPU, iPhone XR/11, and the last two Chrome and Safari. Run `measureLatency` with `realtime=true` in the sample harness, plus battery/thermal over a 10-minute session. |
| B2 | Self-correction and long-session false warnings | **unmeasured.** Lab sessions are 3–8 min with 0 warnings on planted repeats/restarts. The professional QUD recordings (up to 58 min) give 8 warnings on their real repeats/restarts. No learner self-corrections or learner sessions longer than a few minutes exist. | Recordings of real self-corrections (`countSelfCorrections` both ways) and 20+ minute sessions from learners: scripted-error volunteers or reviewed `/help` takes. |
| B3 | Scripted-error recordings (tracker and memorization gate set in PLAN) | **unmeasured**: no recordings exist. | 5–15 volunteers reading generated scripts with planted errors, several phones, held out by speaker (PLAN, Ground truth). |
| B4 | Pronunciation: short vowels | **unmeasured. Out of V1 scope**: the SDK reports the class `unsupported` (30 positives in the help eval split; the checker has no pronunciation classes). | Its own track: a precision ≥ 80% check on scripted errors / qdat_bench. |
| B5 | Pronunciation: confusable pairs, shadda | **unmeasured. Out of V1 scope**, reported `unsupported`. | No recordings with positives (ticket 05, phase 2). |

PLAN treats an `unsupported` pronunciation class as a valid shipped state, but the scoreboard's `BLOCKED` list counts B4 and B5 as release blockers. Shipping a stable V1 with them unmeasured needs an explicit user decision. This report does not waive them.

## Measured behaviour (the numbers behind the claims)

### Missed mistakes and false warnings

| Test set | Audio | Missed | Caught | Warnings correct | False warnings | FW per 5 min |
|---|---|---:|---:|---:|---:|---:|
| Lab exam (held-out reciters), phone-degraded (headline) | 100 sessions, 427 min, professional reciters, planted mistakes | 47 / 532 | 91.2% (30 caught with another class) | 88.8% | 61 | 0.71 |
| Lab exam, studio | same | 20 / 532 | 96.2% | 97.9% | 11 | 0.13 |
| Lab practice, phone | 100 sessions, 429 min | 24 / 617 | 96.1% | 98.2% | 11 | 0.13 |
| Lab real-user slice (tlog, RetaSy, Sobolev), practice + exam | 580 short sessions, 125.8 recited min, near-clean | 0 / 2 | 2 / 2 | — | 9 | 0.36 |
| Help clips, free recitation with lead-in and lead-out (gate) | 388 learner clips, 29 held-out speakers | 8 / 64 | 87.5% | 81.2% | 13 inside clips + 6 in lead-ins + 18 in lead-outs | lead-in 0.54, lead-out 1.69 |
| Help clips, declared passage (gate) | same | 8 / 64 | 87.5% | 83.6% | 11 | 0.87 on clean clips |
| Stitched clean sessions (gate) | 129 sessions, 743 min | — | — | — | 200 | 1.35 overall: tlog 1.22, help speakers 3.48 (28 in 40 min) |

Lab exam wrong calls by class: wrong word 17 missed of 86, added word 20 of 101, omitted word 8 of 219, skipped passage 2 of 126; false warnings 36 wrong word, 14 omitted, 7 skipped, 4 added. Attribution: listener 99, tracker 4, checker 5 of 108. On the lab, the listener on phone audio is the loss. Teacher reference on the same exam: 124 wrong calls.

### Tracking acquisition and false jumps

| Test set | Locked within 5 s | Time on the right place | False jumps |
|---|---:|---:|---:|
| Lab exam, phone (from the disambiguation point) | 98.7% (222/225 entries; 3 took 5–20 s) | 99.2% | 38 in 427 min (4 before the disambiguation point) |
| Lab practice, phone | 99.5% (220/221) | 99.3% | 32 in 429 min |
| Lab real-user slice (short clips) | 92.2% (555/602 entries; 42 never locked) | 98.5% | 14 in 126 min |
| Help clean clips, from the first decoded phoneme (gate) | 84.0% (189/225), median 2.64 s | — | 0/225 clips |
| Help, all clips | 78.1% (303/388) | — | 0/388 clips; 21 clips lock outside the passage |

The tracker gate fails on 10–15 s learner clips that start cold. On minutes-long sessions it locks within 5 s about 99% of the time, but those are professional reciters.

### Supported-class coverage

| Class | SDK state | Help gate positives (recall) | Lab exam positives (recall, phone) |
|---|---|---:|---:|
| wrong_word | supported | 29 (declared 79.3%) | 86 (80.2%) |
| omitted_word | supported | 35 (declared 94.3%) | 219 (96.3%) |
| added_word | supported | **0** | 101 (80.2%) |
| skipped_passage | supported | **0** | 126 (98.4%) |
| mutashabihat (forms another ayah) | slice | 26 (free with lead-in 76.9%) | 94 (88.3%) |
| 11 pronunciation classes | `unsupported` | not scored | not scored |

Added words and skipped passages have no learner-audio positives at all. Their only evidence is the spliced lab set, where inserted words always sit at pauses (`lab_sessions.PAUSE_SIDE`), a known bias.

### Speed (desktop only)

| Measure | full | lite | Machine |
|---|---:|---:|---|
| Listener RTF, onnxruntime-node CPU, 1 thread, 20 ms pushes | 0.026 | 0.018 | Mac mini M4 |
| Same at 320 ms pushes | 0.026 | 0.018 | |
| End-to-end session RTF (listener + tracker + checker, `bench:help`) | 0.055 | 0.037 | |
| Word highlight latency p50 / p95 | 332 / 468 ms | 329 / 462 ms | bar ≤ 600 ms p95 on the device floor |
| Suspected latency p50 / p95 | 291 / 1,079 ms | 260 / 1,130 ms | bar about 1 s |

Source: `v1-release/rtf-desktop-441dddc3.json` and `v1-release/sdk-help-*-441dddc3.json`. Other CPU jobs were running alongside, single-threaded each. **Device-floor runs are still pending.** Nothing here shows that `lite` runs in real time on a 2020 mid-range Android with WASM SIMD.

## QUD regression

Reported only, not a gate (PLAN). This covers the 11 non-hadr QUD recordings with int8 decodes (`session-bench/qud/evidence`, re-verified), replayed at `441dddc3` with context `none` (`v1-release/session-qud-int8-441dddc3.json`). The 4 hadr recordings are marked unsupported, as before.

| | 441dddc3 | Oct 14 run (`session-qud-int8-expanded-v1.json`), same 10 recordings |
|---|---:|---:|
| False warnings per 5 clean min | 2.05 (81 in 197.5 min) | 1.79 on the common 10 (69 in 192.6 min) |
| Same, without the noisy Luhaidan, Fatir | 0.97 (34 in 174.6 min) | 1.43 (50 in 174.6 min) |
| Correctly located time | 96.4% | 93.4% |
| Wrong-position events | 59 | 79 |
| First correct lock within 5 s | 6 / 11 | 5 / 10 |
| Warnings on repeats/restarts | 8 | 7 |

**One recording crashes the engine.** In `witr-muawwidhat` (Fatiha, then Ikhlas, Falaq and Nas) the checker throws `TypeError: Cannot read properties of undefined (reading 'surah')` at `checker.ts:2299` (`words[wi]` undefined) at 358.5 s, while the tracker is locked on 114:5. In the shipped SDK session this kills the session. Every later `pushAudio` rejects (939 of them), no further events arrive, and `end()` throws `Posterior rows start at frame 17514, expected 8956`. The lite tier fails the same way at 346.1 s (`reading 'end'`). On 2026-10-14 the same recording scored without error, so this is a regression from a later fold. Repro: `v1-release/an-nas-crash-repro.mts`.

## Blockers, ranked by shortfall

The crash comes first because it stops any release. The rest are ranked by shortfall, with unmeasured items treated as unbounded.

1. **Engine crash near the end of the Quran (new, release-stopping).** On the one test recording that reaches the end of An-Nas (114:5, commonly recited in witr and other prayers), the free-recitation session throws inside the checker and stays dead for the rest of the take, on both tiers. How often other recordings hit it is unmeasured (see [QUD regression](#qud-regression)). Fixing it is a checker change, so it needs its own lab experiment; this lane only reports it. Until it is fixed, no build should go public, preview or not.
2. **Unmeasured `BLOCKED` items** (unbounded shortfall): device latency (B1), self-correction and long sessions (B2), and scripted-error recordings (B3). These need volunteers and devices, not Modal compute. B4 and B5 (pronunciation) are out of V1 scope and need a user decision (see above).
3. **Lead-out false warnings, 1.69 per 5 min (69% short).** This is the checker on help-speaker phone audio, as `stitched-warnings-v1` already concluded. Help-speaker stitched sessions run 3.48 per 5 min, against 1.22 on tlog.
4. **Stitched-session false warnings, 1.35 per 5 min (35% short).**
5. **Listener gold PER, 4.82% (32% short).** Phone and laptop clips sit at 10.1–10.7%. The listener causes 99 of 108 lab exam wrong calls.
6. **Tracker within 5 s on short help clips, 84.0% (12% short).**
7. **Free-recitation precision, 81.2% (10% short).** On help clips, wrong-word warnings are the weak class (declared wrong-word precision 74.2%).
8. **Memorization-session precision, 83.6% (7% short).** This fell from 87.5% at ed956b4: checker folds tuned on the lab since then added 3 false warnings on help clips (11 vs 8), and doubled them on clean clips (8 vs 4).

Two structural notes for the lead:
- The lab main line in `lab-ledger.json` is scored on teacher-assisted, post-hoc evidence that V1 does not run live. Quote the V1 numbers from this report, not the ledger.
- Practice is 35 wrong calls, exam 108, on similar minutes. Ledger folds are judged on practice only, so the exam/practice gap is consistent with tuning to practice. The exam is the honest number.

## Prerelease claims

Proposed for `1.0.0-rc` (npm `next`) and a website labelled **Preview**, once the An-Nas crash is fixed and the fix re-scored. Each claim is measured above; re-check the numbers against the re-frozen build.

- "Tilawa follows your recitation anywhere in the Quran, without you choosing the surah, and flags memorization mistakes: wrong words, missed words, added words and skipped passages."
- "Pronunciation and tajweed are not checked in this version."
- "Runs on your device. After a one-time model download (about 68 MB, or 25 MB for the lite model), recognition needs no server and nothing is uploaded."
- "Preview quality. In our tests on long recitations recorded through a simulated phone, it caught about 9 in 10 planted mistakes, with about 0.7 false warnings per 5 minutes. On recordings from real users, we measured between 0.4 and 3.5 false warnings per 5 minutes, depending on the recording set; learners' phone recordings were at the high end. Please report wrong feedback so we can fix it."
- For the README/model card: "Measured on desktop CPU (onnxruntime-node): the full model runs at about 1/38 of real time and the lite model at about 1/55. Phone and browser performance has not been measured yet."

Not to claim yet: any device latency or real-time guarantee on phones or the device floor; React Native support; accuracy on children or on fast (hadr) recitation (QUD hadr recordings are marked unsupported); any pronunciation feedback; "stable" or "1.0".

## Modal spend

One lab scoring run (200 CPU shards plus merge, app `ap-mgxzRnfiHXQL3ijkxJZb5l`): **$2.77** (`modal billing report`). Budget: $150. No GPU and no re-decode jobs were needed. The rest was CPU replay on the Mac.
