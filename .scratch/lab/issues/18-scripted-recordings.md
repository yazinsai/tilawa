# 18 Scripted-error recordings

Status: ready-for-human
Type: task

Picks up the recording half of `../../v1-roadmap/issues/05-scripted-error-recordings.md`, aimed at memorization errors: real wrong words, added words and self-corrections, which splices can't fake (an inserted word spliced in has audible seams). Deletions stay spliced in the built sets. Write-up: `docs/benchmarks/lab-recordings.md`.

## What was built

- `tools/lab_scripts.py`: deterministic script batches from the reference index. Plants are about one per 2–3 ayahs: twin, same-context and look-alike wrong words, added words, omissions, and 1–2 self-corrections per script. A few scripts are clean. Planted ayahs are re-phonetized as said. Batch `rec-v1` is in `docs/benchmarks/lab-scripts-v1.json`: 10 scripts, 43 estimated minutes, 94 mistakes, 9 self-corrections, 2 clean.
- `/record` on the lab review app (`modal/lab_review.py`, `modal/lab_web/record.*`). Each script shows steps, the marked Arabic page, the recorder (level meter, device pick, listen back, upload with progress) and the takes, with a verdict and per-plant status.
- `modal/lab_recordings.py` and `tools/lab_recordings_core.py`. They turn each take into 16 kHz mono WAV, decode it with the teacher, align it to what was to be said, check each plant and ayah, and give a verdict (ok, listen, redo). A take that isn't redo becomes a BenchSession (`source: "recorded"`). They also write the spot-check queue `recordings/review/takes.json` and a phone-only manifest, and `--action join` makes a new test set with the takes as a `recorded` slice.
- `lab-cli` reports `slices.recorded`. `lab_decode` skips variants a session has no audio for (recorded takes are phone only).

## Acceptance

- [x] Scripts with labels known by construction and `forms_other_ayah` tags, plus tests (`tools/test_lab_scripts.py`).
- [x] Takes that don't follow the script are told to redo, with the reason per plant. Unsure plants and unscripted changes go to a human (`tools/test_lab_recordings.py`; end-to-end on the preview app, see the write-up).
- [x] Sessions pass the core validator. The review queue passes `validate_queue`.
- [x] A joined test set validates and scores with a `recorded` slice.
- [ ] Record batch `rec-v1` (human). Then hear `recordings/review/takes.json` at `/review?queue=recordings/review/takes.json`.
- [ ] Decide splits: put held-out speakers in `recordings/splits.json` as exam before their takes are scored anywhere.
- [ ] Join into the current test set, decode the candidate, score, and add a scoreboard row.

## Open

- From ticket 05, not done here: the native-language field, a consent and deletion code, and recruiting women and non-Arab reciters. The page asks only for a name.
- The pronunciation-class scripts (letter pairs, shadda) aren't generated. The pool and plant kinds are memorization only.
- On the e2e take, the checker misses the wrong word at 77:18 (لِّلْمُكَذِّبِينَ from 77:15 for بِٱلْمُجْرِمِينَ) even on oracle evidence. That is a checker case, not a label problem. Look at it once real takes show the same pattern.
- A splice-made omission in the e2e audio came out unsure (the cut left a long vowel). Real omissions should be cleaner, so watch the unsure rate on the first real batch before trusting the 0.25 local-PER gate.

## Comments

Built and checked end to end on the preview app (`tilawa-lab-record-preview`) with speaker `test`:
- performed take: listen, with the splice omission unsure;
- as-written take: redo, all 5 plants not performed;
- clean take: ok;
- browser take with a fake microphone: listen, last ayah cut off.

The test data was deleted afterwards. No scoreboard row, because no metric moves until a joined set is scored.

2026-10-30, fixing a passage. This answers "for takes that need to be recorded again, can I just re-record the bit that failed?" A redo or listen take now lists its failed ayahs: plants not heard or unsure, ayahs that sound different, words not found. It offers **Re-record ayahs A–B**, whole ayahs only, defaulting to the failed passage, and the range can be widened. The reciter records just that passage, with pause and resume, and uploads it.

- Processing checks the patch against the script's A..B and joins it into the base take at measured pauses on both sides of each join: 80 ms below room tone + 3 dB, or −30 dB against speech, capped at −12 dB. It level-matches the patch and refuses past 6 dB.
- The composed take is a new take number and is checked end to end. The base take and the patch are kept.
- A refusal says what to record instead. "No clean pause before A" offers A-1..B.

End to end on the preview app, with synthetic audio and one browser fix:

- a redo take fixed at 77:3 became ok;
- a fix of a fix became ok;
- both refusal kinds came back as designed;
- the browser fix became listen, because of the known omission artifact at 77:14.

The joins measured −83 dB. The test data was deleted. Details are in the write-up under "Fixing a passage". No scoreboard row: labels and metrics don't change; only how a take gets made does.
