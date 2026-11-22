# 19 Real-user clean set

Status: resolved
Type: task
Blocked by: 20 (lab-v1.5 registration)

The test set is professional reciters with planted mistakes, so it barely measures false warnings on ordinary people reciting normally, the number users feel. This ticket adds a phone-only `realuser` slice of real non-professional recitation, scored for false warnings per recited minute. It has no planted mistakes.

## Sources

All of these are on Hugging Face. Licences are cleared for internal evaluation.

- `sobolev210/quran-recitation-errors`: app recordings cut into one chunk per ayah (`chunk<k>.wav` under `temp_audio_chunks/chunks_recording_<id>/`), with word-level error tags from 5 reviewers in `metadata.jsonl`. Use Hafs only: 451 chunks from 57 recordings, 405 of them with no tags. A recording's chunks are consecutive ayahs, so join each recording back into one session. A chunk with a tag is a candidate real mistake, not clean audio: keep it in the session and record its tag. `…-errors-test` holds the same metadata.
- RetaSy `quranic_audio_dataset`: already on the volume (`/data/public/retasy/`, ticket 14). Use the `correct` clips that are Quranic and Hafs, about 360 unique. They are single ayahs, so join the clips of one `reciter_id` in mushaf order into one session where a reciter has several. `Aya` is text, so map it to `surah:ayah` first. The labels ignore tajweed and are noisy.
- tlog-v2 test split, `clean-match`, held out by speaker cluster (`docs/benchmarks/student-v1.md`; `tools/jumps_build.py` already stitches these clips). Real Tarteel app users on phones, not used to train the listener. Labels come from filenames and aren't verified. Take a sample of about 1–2 h.

## Approach (a suggestion)

The recordings pipeline (ticket 18) already turns a clean script plus device audio into a `recorded` BenchSession. It does the teacher decode, alignment, ayah checks and the `listen` verdict for ayahs heard differently from the script. Treat each joined source session as a take of a clean script built from its ayahs (`tools/lab_scripts.py::script`), then `manifest` and `join` it into the current test set as its own slice.

Things that need handling:
- Takes under `MIN_S` (5 s): join clips first.
- Seams between joined clips: record them like `resumesS`.
- An istiʿādha or basmala the user may or may not say: the optional items.
- Ayahs the checks flag as heard differently: these are likely real user mistakes. Queue them for listening, don't drop them.

## Acceptance

- Sessions from all three sources validate (`validateSession`) and score through `lab baseline` as a `realuser` slice with its own split. Hold out about half the speakers per source as exam before any scoring.
- The slice's report gives false warnings per recited minute and per session. Most sessions are short and start cold, so it also splits warnings in the first 10 s from the rest.
- Every false warning on the slice goes through the audio-review procedure (`docs/agents/audio-review.md`), so warnings on real mistakes are relabelled as mistakes, not counted as false warnings. The user is asked only about unresolved items.
- A write-up `docs/benchmarks/lab-realuser-v1.md` gives counts per source (sessions, speakers, recited minutes), what was dropped and why, and the main line's numbers on the practice half.

## Comments

Progress (30 Oct). Built `realuser-v1` (580 sessions) on the volume; code in `tools/lab_realuser_core.py`, `modal/lab_realuser.py`, tests `tools/test_lab_realuser.py`.

- **Clips.** sobolev 451 Hafs chunks (47 min), RetaSy 315 unique Quranic `correct` clips that are Hafs or give no qiraah (19.5 min), tlog 504 clean-match test clips in 457 sampled chains (90 min). Each clip is decoded by the teacher and checked against its label before joining. A clip within PER 0.15 of its label is kept as labelled. One further away is relabelled to the best span of words in its surah, when that span is within 0.15 and at least 0.1 better. Up to 0.4 it is kept as "unsure", a likely real deviation. Past that it is dropped.
- **Deviation: sobolev chunk labels are loose.** A chunk often holds several ayahs or part of one. For example, a chunk labelled 113:4 holds 113:1–4, and 1:7 is split across two chunks. Dropping on the label alone lost 126 of 451 chunks. Relabelling to the heard span keeps all but 12, and 132 chunks are relabelled.
- **Deviation: RetaSy `Unknown` reciter.** 160 clips have `reciter_id` "Unknown" and no other speaker field, so they are from different people. Each is its own speaker and is never joined with others. With the 5 s minimum (`MIN_S`), 147 RetaSy clips are dropped as too short even after joining.
- Ayat al-Kursi clips in RetaSy are fragments of 2:255; the Aya text is mapped to the run of words it matches.
- **Speakers per source:** a sobolev recording (no speaker ids), a RetaSy `reciter_id`, a tlog speaker cluster. Splits are in `/lab/data/realuser/splits-v1.json`: per source, a seeded half of the speakers is exam. They were fixed before anything was scored.
- **Sessions.** sobolev: a recording's chunks in chunk order. RetaSy: one reciter's clips in mushaf order, one pass per repeat of an ayah. tlog: one cluster's contiguous clips. Clips are joined with 0.5 s of silence and level-matched (±10 dB at most). Each seam is recorded as contiguous or a jump (`meta.seams`, `meta.resumesS`).
- **Optional items.** An optional isti'adha goes before every separate recording, and an optional basmala before a clip that starts a surah or a separate recording. `check_take` drops the ones it doesn't hear; the ones it keeps become non-Quran spans.
- A backward move is a `repeat` span with `repeat` non-errors. sobolev reviewer tags (46 chunks: Letters 40, Tajweed 11, Wording 2, Tashkeel 1) are kept in `meta.candidates`. Ayahs heard differently are in `meta.flagged`.
- All 580 sessions pass `validateSession` (`lab-validate-cli`, 0 failures).

| source | practice sessions / speakers / min | exam sessions / speakers / min |
|---|---|---|
| sobolev | 27 / 27 / 22.8 | 28 / 28 / 26.6 |
| RetaSy | 31 / 29 / 5.4 | 39 / 39 / 6.6 |
| tlog | 239 / 28 / 50.2 | 216 / 27 / 43.1 |

Progress (30 Oct, later). Scored and triaged on practice. Write-up: `docs/benchmarks/lab-realuser-v1.md`.

- **Scorer.** `realuser` sessions stay out of `splits`, the headline and the other slices, so every planted-set number is unchanged. The report adds `realuser.{practice,exam,all}`:
  - false warnings per recited minute and per session;
  - the first 10 s against the rest;
  - by class and by origin.
  - The ledger keeps these per split (3fab08f).
- **Main line, practice half:** 13 false warnings over 63.3 recited min in 297 sessions (0.205 / min, 0.044 / session; first 10 s 5, rest 8).
- **Triage** (`docs/agents/audio-review.md`, record `docs/benchmarks/lab-realuser-v1-review.json`):
  - 9 sit just after a jump seam. The joined audio skips a chunk the dataset lacks, or an ayah the reciter has no clip of, and 8 of the 9 fire on oracle evidence too.
  - **Deviation:** jump seams are excluded by construction. `realuser-v1/flags-v1.json` covers 1 s before to 3 s after every jump seam (61 windows in 33 sessions). With it, 4 false warnings remain (0.063 / min).
  - The 4 listener cases were heard by Soniox (audio only), and are queued at `realuser-v1/review/false-warnings.json`:
    - 2 unresolved: a likely real wrong word in tlog (تهتدون for تتقون, 2:21) and a possible omission in sobolev (بالخنس, 81:15);
    - 2 spot checks: a chunk cut inside 103:2, and 92:13 heard as written.
- **Waiting on:**
  - your 4 listens;
  - which test set the slice joins (`lab-v1.4` now or `lab-v1.5` from ticket 20) before `lab baseline` registers it.

Review applied (31 Oct). Your four listens (`/lab/reviews/realuser-v1/me.json`), recorded verbatim in `docs/benchmarks/lab-realuser-v1-corrections.json` and applied with `--action correct`:

- Two real omissions are relabelled as `omitted_word` and caught: خسر at 103:2:4 and بالخنس at 81:15:3.
- tlo-281 is bad audio, and its window is flagged.
- sob-049 is said as written (a tajweed slip), so it stays a false warning.
- All 580 sessions still validate.
- **Main line, practice, after review:** 1 false warning over 63.3 recited min in 297 sessions (0.016 / min, 0.003 / session; first 10 s 0, rest 1). Every practice false warning has been through the audio-review procedure.
- Remaining for acceptance: register through `lab baseline` once the slice is joined into the current test set. That waits on your choice of `lab-v1.4` now or `lab-v1.5`. lab-v1.5 is built, and its registration waits on ticket 20's spot check.

Decision (31 Oct): the slice joins **lab-v1.5** once ticket 20 registers it. That is the last step:

1. `modal run modal/lab_realuser.py --action join --base lab-v1.5 --name lab-v1.5u` copies the base. `lab-v1.5u/flags-v1.json` gets the base's latest flags plus realuser-v1's 62 windows.
2. Decode nothing new: both listeners' evidence is copied with the sessions.
3. `python3 tools/lab.py baseline --manifest lab-v1.5u/manifest-v1.json --exclude lab-v1.5u/flags-v1.json`.
4. Add the joined set's practice realuser numbers to the write-up. They should match realuser-v1's practice numbers: 1 false warning in 63.3 min.

## Joined (5 Nov)

- `lab-v1.5u` built from `lab-v1.5` + realuser-v1: 200 + 580 sessions, flags 4 + 62 carried. Main-line listener evidence
  (`exp-listener-stack-be988c-6a17f391`) decoded for all 980 files from main `2b50d856`.
- `lab baseline` on `lab-v1.5u`: practice 30 (20 + 10), exam 113 (44 + 69); autolab switched with `autolab.py test-set`.
- Realuser, main `2b50d856`: practice **7 false warnings** in 63.3 min (wrong_word 4, skipped_passage 3); exam 6 in 62.5 min.
  This does not match the 1 in 63.3 min recorded above. That figure was on an older main line, so the folds since then
  are the likely cause; not yet confirmed by scoring the older main on this set.
