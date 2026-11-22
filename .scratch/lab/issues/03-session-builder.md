# 03 Spliced session builder (test set v1)

Status: resolved
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
- Done (lab-v1, `docs/benchmarks/lab-v1.md`, counts in `lab-v1.json`). `/lab/data/lab-v1/`: 40 sessions (20 practice / 20 exam, 31 reciters, read 135 left out), 186.8 min, 182–415 s each. Slices: 8 ayah-1, 6 strict (26 ×3, 54 ×2, 4 ×1), 10 clean. 370 planted: 114 wrong, 15 mutashabihat swaps (wrong_word + forms_other_ayah, 6 sessions), 78 omitted, 110 added, 53 skipped passages; 0.37 per ayah in sessions with mistakes. One repeat or restart per session (16 / 24), as nonErrors. Studio and phone WAVs per session; phone = held-out OpenSLR-28 simulated RIR + freesound-commercial shards 100–103 at 8–22 dB SNR. All 40 session JSONs and the manifest pass core `validateSession` (`src/bench/lab-validate-cli.ts` via `tools/lab.py run`). Render checks per session (`<id>.checks.json`): exact lengths, pieces equal source × gain, seam levels within 0.17 dB of source, no crossfade clicks, and all 9541 checkable said words found at their labelled output time (zero lag, corr > 0.999). A forced rebuild reproduced every WAV hash. Review queue `/lab/data/lab-v1/review/v1-check.json` (20 splices + 20 cases) passes `validate_queue`. Manifest `evidence` is `sessions/<id>.{listener}.phone.evidence.json`; `sessions-cli --manifest` now fills `{listener}`.
- Deviations: `session_build.py` / `splice.py` not reused (planning is sample-exact against the alignments and boundaries come from ref-v2 phoneme forms). The Al-Rahman and Al-Mursalat refrains are disambiguated within 4 words, so strict starts (6+ words) come from 26/54/4. Restart = back to word 1 of the current ayah after a pause. No scoreboard row: nothing reads lab-v1 yet (ticket 08 adds the scorer).
- Scorer contract (ticket 08, check before resolving): planted `wrong_word` and `added_word` events carry `said` phonemes (the scorer throws otherwise); omitted/skipped words are absent from truth `words`; an `added_word` location is the word after the insertion; evidence lives at `evidence/<listener>/<id>.<studio|phone>.json` beside the manifest; the builder tags strict sessions itself (refrain starts in 55 need only 4 words, below the scorer's >5 fallback).

lab-v1.2 (`docs/benchmarks/lab-v1.2.md`): lab-v1.1 rebuilt with three rules from your listens to its both-mismatch cases. Each is enforced in the planner and in the seam gate.

- A linked cut sits at least 6 dB below the ayah level in its source, or 8 dB after a vowel or long vowel.
- A wrong or added word lasts at least 0.26 s.
- A cut before a weak onset after a pause (hamza including wasl, ع, ح, ه, خ, غ) needs 80 ms of measured silence, near room tone. Otherwise the cut is moved earlier into silence or refused.

On lab-v1.1 the rules refuse 172 of 360 splices, including all 3 bad ones and one of the 4 you passed (e10-x6, a long vowel at −6.3 dB). Mutashabihat swaps now carry up to 4 shared twin words to reach a clean seam; without that, the depth rule left 1. The set has 40 sessions, 180.9 min and 351 planted mistakes: 45 wrong, 25 mutashabihat, 30 omitted, 170 added and 81 skipped, plus 40 repeats and restarts. It converged in 5 loop iterations with every studio check and all 631 seams passing. No scoreboard row: nothing in the scoreboard reads lab test sets yet.

lab-v1.3 (`docs/benchmarks/lab-v1.3.md`): lab-v1.2 rebuilt with three rules from your listens to its triage queue (7 bad in 100). Each is enforced in the planner and in the seam gate, which re-reads each side's source boundary from the alignment (seam sides now carry `piece: host|donor`).

- A donor with a pause after it in its source (waqf form) goes only where the host also pauses right after the insertion. Otherwise an occurrence linked on its following side is used. This covers added and wrong words and the last word of a twin span.
- Donors come from the host's own recording.
- A repeat or restart breaks off at a pause boundary, or the site is refused.

On lab-v1.2 they refuse 90 of 391 splices (56, 6 and 28). That catches 6 of the 7 bad ones. p15-x1 is not caught: its donor lands on a real host pause, and a mirror rule would refuse about 35 added words you passed. The rules also refuse 27 of your 86 ok splices. Added words and skipped passages are each capped at 30% of the planted mistakes. Skips are included because with only added words capped they reached 45%. The set has 40 sessions, 186.3 min and 335 planted mistakes: 98 wrong, 21 mutashabihat, 62 omitted, 94 added (28.1%) and 60 skipped, plus 29 repeats and 11 restarts. It converged in 9 loop iterations with every studio check and all 588 seams passing. No scoreboard row: nothing in the scoreboard reads lab test sets yet.

lab-v1.4 (`docs/benchmarks/lab-v1.4.md`): lab-v1.3 rebuilt after your listens to its triage queue (9 bad in 116, all wrong or added words). Each rule is enforced in the planner and in the seam gate (`insertLinked`, `reenterLinked`).

- Inserted audio (wrong and added words, not twin spans) meets the host only at seams that are a pause on both sides, in the host and in the donor's recording (`Donors.pause_spans`). The donor stretch runs pause to pause and reads as the host's words except the planted one, which alone is labelled. At most 6 shared words either side.
- A repeat re-enters at a pause too.
- Review clips cover every seam ±1 s (`meta.reviewSeamPadS`; earlier labels unchanged).

On lab-v1.3 the rules refuse 211 of 375 splices: all 98 wrong words, 84 of 94 added words and all 29 repeats. That includes all 9 bad splices, and 50 of the 56 inserted splices you passed. The set has 40 sessions, 180.2 min and 252 planted mistakes: 5 wrong, 19 mutashabihat, 135 omitted, 43 added and 50 skipped, plus 24 repeats and 16 restarts. All 96 inserted-audio seams are pause to pause. It converged in 6 loop iterations with every studio check and all 359 seams passing. `spliced-clean-int8` was decoded on both variants.

Bias: every inserted mistake sits at a pause, and wrong words nearly vanish (5). Deletions and twin swaps still cover linked speech. No scoreboard row: nothing in the scoreboard reads lab test sets yet.
