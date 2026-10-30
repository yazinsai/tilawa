# 19 Real-user clean set

Status: ready-for-agent
Type: task
Blocked by: none

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
