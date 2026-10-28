# 03 Spliced session builder (test set v1)

Status: ready-for-agent
Type: task
Blocked by: 02

Spec: `../spec.md` decisions 3–6 and 8. Reuse `tools/session_build.py` and `tools/splice.py` where they fit.

## Acceptance

- Disambiguation point computed from the reference index for any start word. Normal sessions start where it falls within 5 words; the strict slice starts inside long shared runs (Al-Rahman refrain and similar) and is tagged.
- About 40 sessions of 3–8 min across the practice and exam reciters. About 20% start at an ayah 1; about 25% are fully clean.
- Planted mistakes (wrong, omitted, added, skipped passage; mutashabihat swaps tagged) at about 1 per 2–3 ayahs, 150 or more in total. Each is placed only at a boundary with no assimilation across the words, with a 10 ms crossfade. Wrong and added words come from the same reciter.
- About one planted repeat or restart per session, labelled as a non-error.
- Labels in the existing session format (`sessions.ts`), with a splice record per planted edit (source times, crossfade, boundary type).
- A studio WAV and a phone-degraded WAV per session (reverb plus licensed noise, fixed seed).
- Deterministic: the same seed and inputs give byte-identical output. Runs on Modal.

## Comments

- Disambiguation points are ready: `packages/core/src/disambiguation.ts` (`disambiguationWords(index)`), table at `/lab/data/reference/disambiguation-v1.json` on the `tilawa-lab` volume (`words[i]` = further words needed from reference word i, -1 never; regenerate with `src/bench/disambiguation-cli.ts`). 85% of start words are unique within 2 further words, about 5% need more than 5; the longest runs are 23:5 = 70:29 (27 words), 4:43 = 5:6, 27:80 = 30:52, 2:47 = 2:122. Exact phonemes: a mid-ayah case ending can make a point one word early for a reciter who pauses there (2:255 vs 3:2).
- Review queue: emit the v1 queue (20 splices with `before`/`after` source clips, plus 20 random cases: clean stretches, repeats/restarts, phone-degraded) at `/lab/data/<testSet>/review/v1-check.json` in the format of `docs/benchmarks/lab-review.md`; validate with `tools/lab_review_core.py::validate_queue`. Item ids must be unique and stable per test set (verdicts are keyed by id).
