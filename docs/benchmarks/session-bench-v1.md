# Representative sessions: first benchmark increment

Session measures are reported only. Existing clip, listener, runtime and lead-in release checks remain unchanged. Recorded-session agreement, coverage minimums and two side-by-side scoreboard runs are required before promotion.

## Available data

The held-out help set has 388 usable clips from 29 speakers. Its 169 strict consecutive same-speaker clean runs contain at most two clips; the longest totals 28.84 seconds. No minute-long run exists, and no error clip has both adjacent clean ayahs from the same held-out speaker. Labels do not contain word intervals. This cannot currently supply the proposed minutes-long error sessions. Tlog clusters have not yet been independently audited for these sessions; prior mixed-speaker/evidence-concatenated sessions are not reused.

The stitcher accepts a held-out timed catalogue, requires reviewed planted errors and clean continuations, keeps every session to one speaker, and writes continuous 16 kHz mono audio with seeded 300–900 ms gaps. That audio needs a fresh listener decode. No decode from individual clips is reused. Repeated or backward visits require their own occurrence spans; the basic clip catalogue builder is for monotonic single-ayah source clips.

## Initial baseline

Three original continuous QUD recordings (449.18 seconds, two reciters) were acquired at pinned dataset revision `76c3d804d082c3cc189601faf13a41996e17b6dc`. Each WAV was decoded in a fresh streaming SDK session using `spliced-clean-int8`, with 20 ms audio pushes. No per-clip evidence was concatenated. Source, waveform, student evidence and model hashes are recorded in `session-provenance-v1.json`; audio/evidence stay outside git. The source card verifies Hafs and reviewed word truth; annotation CC BY 4.0 explicitly excludes audio rights. Training reciter/source overlap is unknown, so this remains diagnostic.

`session-clean-int8-v1.json` contains the fresh student baseline; `session-baseline-v1.json` contains the comparison using existing windowed teacher evidence. Both replay the tracker without posteriors and the checker with posteriors, on 40 ms evidence cells, with no starting-passage hint.

| Recording | Timed speech | Student false warnings | Student located time | Teacher false warnings | Teacher located time |
|---|---:|---:|---:|---:|---:|
| Ali Jaber, Inshiqaq | 111.70 s | 0 | 97.06% | 0 | 97.06% |
| Ali Jaber, Nisa excerpt | 57.68 s | 4 | 87.56% | 6 | 97.75% |
| Ahmed Diban, Saad | 260.51 s | 24 | 79.78% | 15 | 81.93% |

Student aggregate: 28 confirmed false warnings in 7.16 clean-speech minutes (19.54 per five minutes); 85.60% correctly located sampled time, 10 wrong-position events, and all three first correct locks within five seconds. Teacher: 21 false warnings over the same timed speech (14.65 per five minutes), three wrong-position events. Saad includes one labelled repeat; neither listener confirms a warning on its repeat interval.

The single slow recording concealed substantial failures on faster professional recitation. These small diagnostic results do not estimate learner performance or select release thresholds. No error positives exist, so recall and error feedback delay are unmeasured. Position is the held locked top candidate, with a one-ayah/two-second tolerance. This is evidence replay after continuous CPU decoding, not a device-latency measurement.

Two final words were initially rejected because rounded evidence-frame duration was shorter than original audio. Conversion now uses the source audio duration; truth is not stretched. A regression test pins this case. All three recordings score without exclusions. See QUD reconciliation below for the all-16 conversion and replay.

## QUD reconciliation

`qudSession` converts QUD truth with these deterministic rules; `qudAudit` validates the result and records each recording as converted (with span, repeat, restart, non-Quran and surah-jump counts) or excluded with the reason.

- Clock: truth ending up to 0.1 s past the source audio duration is annotation rounding and is clipped to the duration. A larger overrun is rejected.
- Occurrences: one span per ayah occurrence, in recorded order. Forward words in the same ayah extend the span when the gap is at most 2 s and no non-Quran interval lies in the gap.
- Repeats and restarts: equal or backward movement within an ayah starts a `repeat` span. A return to an earlier ayah (compared in Quran order against the last recited word, across any special words) starts a `restart` span when it lands on word 1, otherwise a `repeat`. Every word of a repeat/restart span gets a matching `nonErrors` entry. Prayer Fatiha recitations after another surah therefore become restarts.
- Special words (`Basmala:n`, `Isti'adha:n`) become labelled non-Quran spans.
- Non-Quran intervals are clipped to the gap next to any word they partially overlap, never stretched over word truth, and they split the occurrence. A non-Quran interval that covers a whole word or lies inside one is contradictory and the recording is excluded.

All 16 recordings convert (`session-qud-audit-v1.json`). Repeat plus restart counts equal each recording's QUD `repeat_events`, except the prayer recordings, where the extra restarts are Fatiha recitations: prayer-baqarah-yasin 1, taraweeh-maryam 6, witr-muawwidhat 2. Minshawi, Luqman was previously rejected for a 36 ms final-word overrun; prayer-baqarah-yasin for a final non-Quran interval 92 ms past the audio end.

The all-16 teacher replay previously did not finish: the checker re-runs a windowed alignment on every 40 ms push, and per-cell cost lookups made long, dense recordings take over an hour. Cost tables are now computed once per window and phoneme. The output is identical (Nisa report rows byte-equal) and Nisa replays in 6 s instead of 58 s. `session-qud-teacher-v1.json` holds the all-16 teacher replay (15 minutes on one CPU core; Afasy, Baqara alone takes 6 minutes).

All-16 teacher result, reported only: 208 confirmed false warnings, 3.03 clean-speech false warnings per five minutes over 308.7 clean minutes, 97.37% correctly located sampled time, 67 wrong-position events, 17 warnings on repeat/restart intervals, and 8 of 16 first correct locks within five seconds. Luhaidan, Fatir (noisy, 27 repeats) dominates: 66 false warnings, 33 wrong-position events and 13 non-error warnings. The Diban and Jaber rows match the three-recording teacher baseline exactly. These are professional reciters, and the teacher evidence may overlap training, so this does not estimate learner performance.

## Wider baseline by context

Every QUD recording is scored at four contexts. They are built from the recorded words only, and none of them gives a true-start hint. In weakest-last order:

- assignment: the declared checker passage from the lowest to the highest recited word in Quran order;
- page: the Madani page of the first recited word;
- surah: the surah of the first recited word;
- none.

`sessions-cli --qud` builds these with `qudContexts` (`--qud-contexts own` keeps the session's own chain). Each context runs one replay, and the review dump reuses its events.

Teacher, all 16 recordings, 308.7 clean minutes (`session-qud-teacher-v1.json`):

| Context | False warnings | Per 5 clean min | Located time | Wrong position | Non-error warnings | First lock ≤ 5 s |
|---|---:|---:|---:|---:|---:|---:|
| assignment | 1621 | 26.25 | 97.37% | 67 | 33 | 8/16 |
| page | 188 | 3.04 | 97.45% | 65 | 18 | 9/16 |
| surah | 187 | 3.03 | 97.46% | 66 | 18 | 9/16 |
| none | 187 | 3.03 | 97.37% | 67 | 17 | 8/16 |

Fresh `spliced-clean-int8` student, 7.2 clean minutes (`session-qud-int8-v1.json`). It is paired with the current `qudSession` truth through `--student-evidence .scratch/session-bench/qud/evidence`:

| Context | False warnings | Per 5 clean min | Located time | Wrong position | Non-error warnings | First lock ≤ 5 s |
|---|---:|---:|---:|---:|---:|---:|
| assignment | 82 | 57.22 | 85.60% | 10 | 2 | 3/3 |
| page | 28 | 19.54 | 85.60% | 10 | 0 | 3/3 |
| surah | 29 | 20.24 | 85.82% | 10 | 0 | 3/3 |
| none | 28 | 19.54 | 85.60% | 10 | 0 | 3/3 |

Page and surah barely move anything. The range prior only nudges the tracker, and professional recitation acquires without it. Assignment is the outlier because it switches the checker to declared-passage mode. On fluent recitation that multiplies false warnings: taraweeh-maryam goes from 2 to 965, prayer-baqarah-yasin from 0 to 177, luhaidan-qiyamah from 13 to 137, and diban-sad from 15 to 63. For the multi-surah prayer recordings, the min..max assignment spans Fatiha through the last surah. That passage is wider than any real assignment, so those two rows overstate the effect. Even so, single-surah recordings rise as well. The session numbers move with context the way the spec intended, but in the opposite direction from what was expected for assignment. No error positives exist, so recall and delay remain unmeasured.

Student decodes are missing for 13 recordings: abdulbasit-raad, afasy-baqarah, ayyub-juz-amma, badri-juz-tabarak, husary-fath, luhaidan-fatir, luhaidan-haqqah, luhaidan-qiyamah, mandour-taha, minshawi-luqman, prayer-baqarah-yasin, taraweeh-maryam, witr-muawwidhat. Their WAVs have not been acquired. No decode was fabricated.

## Commands

From the repository root:

```sh
python3 tools/session_build.py audit --out .scratch/session-bench/coverage.json
python3 tools/session_build.py build --catalog timed-clips.json --out .scratch/session-bench/build --seed 12
pnpm typecheck
node packages/core/dist/bench/sessions-cli.js --qud .scratch/bench/qud --only jaber-inshiqaq --out docs/benchmarks/session-baseline-v1.json
node packages/core/dist/bench/sessions-cli.js --qud .scratch/bench/qud --audit --out docs/benchmarks/session-qud-audit-v1.json
node packages/core/dist/bench/sessions-cli.js --qud .scratch/bench/qud --out docs/benchmarks/session-qud-teacher-v1.json
node packages/core/dist/bench/sessions-cli.js --qud .scratch/bench/qud --student-evidence .scratch/session-bench/qud/evidence --listener spliced-clean-int8 --out docs/benchmarks/session-qud-int8-v1.json
# Acquired original QUD WAVs: fresh student decode, then score the saved timed manifest.
uv run --python cpython-3.12-macos-aarch64-none --with huggingface-hub --with pyarrow --with 'numpy>=2' tools/session_qud_audio.py --ids jaber-inshiqaq,jaber-nisa,diban-sad --revision 76c3d804d082c3cc189601faf13a41996e17b6dc --out .scratch/session-bench/qud
node packages/sdk/dist/bench/decode-cli.js --model .scratch/models/spliced-clean-int8 --wavs .scratch/session-bench/qud --out .scratch/session-bench/qud/evidence
node packages/core/dist/bench/sessions-cli.js --manifest .scratch/session-bench/qud/manifest.json --listener spliced-clean-int8 --out docs/benchmarks/session-clean-int8-v1.json
node packages/core/dist/bench/sessions-cli.js --manifest .scratch/session-bench/qud/teacher-manifest.json --out docs/benchmarks/session-baseline-v1.json
# For built sessions, first decode the continuous WAVs via the SDK into the manifest's evidence paths.
node packages/core/dist/bench/sessions-cli.js --manifest .scratch/session-bench/build/manifest.json --listener teacher --out docs/benchmarks/session-baseline-v1.json
```

From `packages/core`, add the reported measures to the scoreboard:

```sh
pnpm scoreboard --listener spliced-clean-int8 --evidence ../../.scratch/sdk-help/spliced-clean-int8-node-lw --leadin-evidence ../../.scratch/leadin/evidence/spliced-clean-int8-lw --sessions ../../docs/benchmarks/session-clean-int8-v1.json --note 'Add occurrence-aware session scoring and fresh continuous student baseline; session measures reported only, release gates unchanged'
```

Source, use track and context are kept separate. Assignment uses the declared checker passage; resume uses the tracker prior. Surah and page use the tracker range prior (`priorRange`). The replay supplies no true-start hint.

## Remaining work

- Audit same-speaker held-out tlog availability and obtain word-timed clean/error continuations; do not substitute voices.
- Acquire and decode the 13 remaining QUD WAVs for the student. Work out why declared-passage mode warns so often on fluent recitation (see Wider baseline by context).
- Error sessions are still needed. Aggregate delay distributions, per-class and mutashabihat coverage, and distinct twin ambiguity (initial scorer reports general tracker ambiguity).
- Add event-centred session audio review to the dashboard. Current integration adds reported scoreboard metrics, not session review items.
- Recorded sessions and independent review remain deferred. No collection or publication is initiated by this increment.

Validation: 10 targeted lifecycle/occurrence/position/context/audio-clock tests and deterministic WAV/same-speaker builder tests pass. The broad core suite's assertions passed, but its runner reported an existing worker update timeout; this is not represented as a clean full-suite pass.
