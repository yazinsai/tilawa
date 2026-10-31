# Real-user clean set v1 (`realuser-v1`)

Ordinary people reciting normally on phones, with no planted mistakes, scored for false warnings per recited minute (`.scratch/lab/issues/19-real-user-clean-set.md`). The test set's planted sessions are professional reciters, so they barely measure the false warnings users feel.

- Code: `tools/lab_realuser_core.py` (pure), `modal/lab_realuser.py` (Modal).
- Tests: `tools/test_lab_realuser.py`, `packages/core/test/lab.test.ts`, `tools/test_lab_ledger.py`.
- On the `tilawa-lab` volume:
  - clips: `/lab/data/realuser/`;
  - test set: `/lab/data/realuser-v1/`;
  - flags: `realuser-v1/flags-v1.json`;
  - queues: `realuser-v1/review/false-warnings.json`, `realuser-v1/review/flagged.json`.
- Triage record: `docs/benchmarks/lab-realuser-v1-review.json`.

## Build

```sh
modal run modal/lab_realuser.py --action fetch
modal run modal/lab_decode.py --listener teacher --test-set realuser/clips --variants phone
modal run modal/lab_realuser.py --action build
modal run modal/lab_decode.py --listener teacher --test-set realuser-v1 --variants phone
modal run modal/lab_realuser.py --action label
modal run modal/lab_decode.py --listener spliced-clean-int8 --model models/spliced-clean-int8 --test-set realuser-v1 --variants phone
modal run modal/lab_realuser.py --action join --base <current test set> --name <new name>
```

1. **Clips.**
   - sobolev: `sobolev210/quran-recitation-errors`, Hafs rows only. The `-test` repo's `metadata.jsonl` is byte-identical.
   - RetaSy: `/data/public/retasy/` (ticket 14). Only Quranic `correct` rows with qiraah `hafs` or none given, and one copy of identical audio. The Aya text is mapped to its words by letters with the diacritics dropped. Ayat al-Kursi clips are fragments of 2:255, so they map to a run of words.
   - tlog: tlog-v2 clean-match test split, held out by speaker cluster. It is sampled as contiguous chains of one cluster's clips, round-robin over clusters, to 90 min.
2. **Each clip is checked against its label by the teacher** (`verify_clip`, the greedy decode against the label with an optional isti'adha and basmala first):
   - within PER 0.15: kept as labelled;
   - else relabelled to the best run of words in the label's surah (`best_span`), when that run is within 0.15 and at least 0.1 better;
   - else kept as "unsure" up to 0.4 (heard differently: a likely real deviation);
   - else dropped.
3. **Splits**, fixed in `/lab/data/realuser/splits-v1.json` before anything was scored: per source, a seeded half of the speakers is exam. A speaker is a sobolev recording (the dataset has no speaker ids), a RetaSy `reciter_id`, or a tlog speaker cluster.
4. **Sessions.**
   - sobolev: a recording's chunks in chunk order.
   - RetaSy: one reciter's clips in mushaf order, one pass per repeat of an ayah. A pass under 5 s joins the one before it.
   - tlog: one sampled chain.
   - Clips are level-matched to the session's median clip (at most ±10 dB) and joined with 0.5 s of silence.
   - At most 6 min per speaker. A group under 5 s (`MIN_S`) is dropped.
5. **Labels** (`check_take` from the recordings pipeline, then `session`):
   - Each session is a take of a clean script built from its clips' words, in said form at the clip edges. An optional isti'adha goes before each separate recording, and an optional basmala before a clip that starts a surah or a separate recording. `check_take` drops the optional items it doesn't hear; kept ones become non-Quran spans.
   - One span per clip. A backward move is a `repeat` span whose re-recited words are `repeat` non-errors.
   - No events. The BenchSession source is `recorded` (real device audio), and the manifest's `slices` are `{source: "realuser", origin}`.
   - `meta` records:
     - the clips;
     - the seams between them (contiguous or a jump; `resumesS`);
     - ayahs heard differently (`flagged`);
     - sobolev reviewer tags as candidate real mistakes (`candidates`).
   - All 580 sessions pass `validateSession` (`lab-validate-cli`, 0 failures).
6. **Seam flags.** `flags-v1.json` excludes a window around every jump seam, from 1 s before to 3 s after (61 windows in 33 sessions). There the joined audio reads as a skipped passage nobody recited: a chunk missing from the source, or another recording. It also holds one reviewed bad-audio window. See the triage below.
7. **Reviewed corrections** (`--action correct`): two words you heard as never said are confirmed `omitted_word` events. Those two sessions are no longer clean (`slices.clean` false). All 580 sessions still pass `validateSession`.

## Counts

| source | found | joined | practice sessions / speakers / recited min | exam sessions / speakers / recited min |
|---|---|---|---|---|
| sobolev | 451 chunks | 438 | 27 / 27 / 22.5 | 28 / 28 / 26.3 |
| RetaSy | 368 rows | 145 | 31 / 29 / 4.9 | 39 / 39 / 6.0 |
| tlog | 504 clips (sampled) | 502 | 239 / 28 / 36.0 | 216 / 27 / 30.2 |
| all | | 1,085 | 297 / 84 / 63.3 | 283 / 94 / 62.5 |

Audio minutes are higher than recited minutes (tlog clips have lead and tail silence): 78.4 on practice and 76.3 on exam.

### What was dropped and why

- **sobolev (13 of 451).**
  - 12 chunks hold no text near their label, the best span or any ayah (PER over 0.4).
  - 1 recording is under 5 s even when joined.
  - **Deviation:** chunk labels are loose. A chunk often holds several ayahs or part of one: a chunk labelled 113:4 holds 113:1–4, and 1:7 is split over two chunks. Checking against the label alone dropped 126. Relabelling to the heard span keeps all but 12, with 132 chunks relabelled.
  - 36 joined chunks are "unsure".
  - Chunks start and end at speech level, with no edge silence.
- **RetaSy (223 of 368).**
  - 53 are duplicates of identical audio.
  - 23 hold no text near their label.
  - 147 are under 5 s even joined, nearly all from reciter `Unknown`.
  - **Deviation:** 160 clips have `reciter_id` "Unknown" and no other speaker field, so they are from different people. Each is its own speaker and is never joined with others, so only those of 5 s or more survive.
  - 44 joined clips are "unsure": short learner clips with clipped starts and pronunciation noise. The labels ignore tajweed.
- **tlog (2 of 504).** 2 hold no text near their label. 36 are "unsure".

## Report

`lab-cli` reads `slices.source: "realuser"` (and `slices.origin`).

- These sessions are left out of `splits`, the headline, and the studio, strict and recorded slices, so every planted-set number is unchanged.
- The report gets `slices.realuser` (the usual totals) and `realuser.{practice,exam,all}`, with:
  - false warnings, per recited minute and per session;
  - `firstSeconds`: warnings in a session's first 10 s (`EARLY_S`) against the recited minutes there;
  - `rest`;
  - `byClass` and `byOrigin`.
- The ledger keeps the realuser numbers per split (`from_report`), and an experiment's verdict prints the realuser false warnings beside the main line's.

## Main line on the practice half

`spliced-clean-int8`, main line at 35e6fda (engine code unchanged since 083ba6e), free recitation, no context.

| | false warnings | recited min | per recited min | per session | first 10 s | rest |
|---|---|---|---|---|---|---|
| as built, no flags | 13 | 63.3 | 0.205 | 0.044 | 5 (over 32.3 min) | 8 (over 31.0 min) |
| seam flags only | 4 | 63.3 | 0.063 | 0.013 | 1 | 3 |
| **after review** (seam flags, 2 real omissions relabelled, 1 bad-audio window) | **1** | 63.3 | **0.016** | **0.003** | 0 (over 32.3 min) | 1 (over 31.0 min) |

- After review, by source: sobolev 1 (22.5 min), RetaSy 0 (4.9 min), tlog 0 (36.0 min).
- The two relabelled omissions are caught (2 of 2, none misclassified).
- The exam half is numbers only and isn't reported here.

## Triage of the practice false warnings

Following `docs/agents/audio-review.md`. Record: `docs/benchmarks/lab-realuser-v1-review.json`. Corrections, with your verdicts verbatim: `docs/benchmarks/lab-realuser-v1-corrections.json`, applied by `--action correct`.

- **9 at a jump seam (automatic, bad splice).** Each warning is raised within 1.4 s after a seam where the joined text jumps ahead or elsewhere, and 8 of the 9 fire on oracle evidence too.
  - sobolev: three chunks missing from the dataset (108:2, 79:43–44, the start of 83:14).
  - RetaSy: reciters who lack the ayah in between (1:3 twice, 110:2, words of 2:255, 1:4 and 112 to 114).
  - No reciter skipped anything. The seam flags remove exactly these 9.
- **4 listener cases.** Each was first heard by the independent model, from the audio window only: Soniox `stt-async-v5`, the judge with rule `reach` from `lab-triage-v1.md` "Soniox margin v2". It is calibrated on your lab-v1.x listens, not recalibrated on real users' phone audio. Then you listened (31 Oct):
  - **sob-028, omitted خسر (103:2).** You: «خُسْرٍ is never said». A real omission, now `omitted_word` at 103:2:4, and caught. Soniox also heard it missing. My provisional reading of a cut chunk was wrong.
  - **sob-032, omitted بالخنس (81:15).** You: «ٱلْخُنَّسِ is never said». A real omission, now `omitted_word` at 81:15:3, and caught. Soniox agreed: opinion mismatch, judge too close to call.
  - **sob-049, wrong word at وإنّ (92:13).** You: said as written, «there's no long 'n' at the end - a tajweed mistake». The false warning stands as a listener fault: the missing ghunna is a pronunciation slip, which is not a supported class. Soniox agreed.
  - **tlo-281, omitted تتقون (2:21).** You: bad audio. Its window (40.94–47.1 s) is flagged. Soniox had heard تهتدون.
- So Soniox matched your listens on 3 of 4 as to whether the word is there. The odd one out is tlo-281, which you called bad audio.
- The queue's tally reads "fail" (3 bad of 4, maxBad 1). That gate is for test-set spot checks; on a queue of failures, bad items are the expected outcome.
- `realuser-v1/review/flagged.json` holds the 49 practice ayahs heard differently from their label and the 11 practice reviewer tags, for later labelling. Exam sessions are never queued.

## Limits

- **Exam labels are unreviewed.** A real mistake in an exam session counts as a false warning there, so exam rates lean high. On practice, 2 of 4 non-seam warnings were real omissions.
- **Speakers are approximate.** sobolev recordings may share a speaker across splits, since the dataset has no speaker ids. A tlog cluster is a voice group, not one person.
- **The seam windows** hide any genuine engine warning within 1 s before or 3 s after a jump seam.
