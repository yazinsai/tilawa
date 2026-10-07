# Acceptance eval v2: help clips (ticket 07)

Report v2, commit 3e91183, reference ref-v2. Held-out speakers only (eval split): 413 clips, 25 unscored by the import or the audit (ticket 12), 388 scored from 29 speakers. Labels are the final post-audit `labels.jsonl` with version swaps applied. 6 letter-confusion substitutions are pronunciation errors, so they count as negatives here. Repeats are non-errors: a confirmed event on one is a false positive.

Teacher evidence is the offline Muaalem decode (30 s windows), replayed in 1 s chunks. It is not a streaming student.

Two arms, both core engine code. **Free recitation** is the default: the checker follows the tracker lock (Decision 3). **Declared passage**: the session passes the prompted passage (`CheckerOptions.passage`); the checker aligns against it and uses the tracker lock only to follow a deliberate jump (a lock of 4+ words more than an ayah off the passage whose words don't also fit the passage). At session end a declared session confirms a suspected wrong or added word once a later word was recited right; free recitation still ends it `uncertain`.

## Memorization

| Measure | Free (all) | Declared (all) | Free (audit-verified) | Declared (audit-verified) | Bar |
|---|---:|---:|---:|---:|---|
| Clips / speakers | 388 / 29 | 388 / 29 | 59 / 21 | 59 / 21 | |
| Hits / false positives / misses | 32 / 26 / 32 | 57 / 34 / 7 | 7 / 4 / 4 | 11 / 4 / 0 | |
| Precision | 55.2% | 62.6% | 63.6% | 73.3% | ≥ 90% |
| Recall | 50.0% | 89.1% | 63.6% | 100.0% | ≥ 80% |
| Clean clips / minutes | 225 / 46.5 | 225 / 46.5 | 14 / 3.6 | 14 / 3.6 | |
| Clean false warnings (clips with one) | 8 (7) | 13 (12) | 0 (0) | 0 (0) | |
| False warnings per clean clip | 0.036 | 0.058 | 0.000 | 0.000 | |
| False warnings per 5 min (short clips) | 0.86 | 1.40 | 0.00 | 0.00 | ≤ 1, phase 2 only |
| False positives on repeats (23 clips) | 0 | 0 | 0 | 0 | |
| Suspected skip labels raised | 5/26 | 5/26 | 2/7 | 2/7 | |

### Per class (all scored clips)

| Class | Positives | Free hits | Free FP | Free precision | Free recall | Declared hits | Declared FP | Declared precision | Declared recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| wrong_word | 29 | 1 | 9 | 10.0% | 3.4% | 24 | 16 | 60.0% | 82.8% |
| omitted_word | 35 | 31 | 10 | 75.6% | 88.6% | 33 | 11 | 75.0% | 94.3% |
| added_word | 0 | 0 | 4 | 0.0% | — | 0 | 4 | 0.0% | — |
| skipped_passage | 0 | 0 | 3 | 0.0% | — | 0 | 3 | 0.0% | — |

`skipped_passage` labels are all `suspected` (a skipped ayah may be a deliberate jump), so they are not confirmed positives; they appear in the suspected-skip row above.

## Short vowel

30 positives at 27 speaker × ayah sites. The checker has no pronunciation classes yet, so checker recall is 0 and precision is unmeasured: `short_vowel` is `unsupported` as a feedback class. The teacher-level E1 numbers on the same clips:

| class | positives | sites | in greedy | canonicalized | posterior prefers error | negatives | greedy FP | posterior FP | verdict |
|---|---:|---:|---|---|---|---:|---|---|---|
| short vowel | 30 | 27 | 19/30 (63.3%) | 3/30 (10.0%) | 28/30 (93.3%) | 6755 | 8/6755 (0.1%) | 78/6755 (1.2%) | viable |
| ح/ه | 0 | 0 | 0/0 | 0/0 | 0/0 | 562 | 2/562 (0.4%) | 4/562 (0.7%) | unsupported |
| ع/ء | 0 | 0 | 0/0 | 0/0 | 0/0 | 930 | 7/930 (0.8%) | 18/930 (1.9%) | unsupported |
| ص/س | 7 | 3 | 3/7 (42.9%) | 1/7 (14.3%) | 5/7 (71.4%) | 391 | 11/391 (2.8%) | 12/391 (3.1%) | unsupported |
| ض/د | 0 | 0 | 0/0 | 0/0 | 0/0 | 268 | 4/268 (1.5%) | 6/268 (2.2%) | unsupported |
| ط/ت | 0 | 0 | 0/0 | 0/0 | 0/0 | 473 | 3/473 (0.6%) | 5/473 (1.1%) | unsupported |
| ق/ك | 0 | 0 | 0/0 | 0/0 | 0/0 | 618 | 5/618 (0.8%) | 10/618 (1.6%) | unsupported |
| ذ/ز | 0 | 0 | 0/0 | 0/0 | 0/0 | 264 | 1/264 (0.4%) | 1/264 (0.4%) | unsupported |
| ث/س | 0 | 0 | 0/0 | 0/0 | 0/0 | 35 | 3/35 (8.6%) | 3/35 (8.6%) | unsupported |
| missing shadda | 0 | 0 | 0/0 | 0/0 | 0/0 | 804 | 41/804 (5.1%) | 82/804 (10.2%) | unsupported |

Scored 253 clips; skipped 6 (decode too far from the canonical, or unmappable vowel site). Real acted errors, next to E1 (`e1-teacher-bias-v1`) and the E1b splice upper bound (`e1b-spliced-v1`).

## Tracker gate on this audio

Each clip is one start. No word onsets are labelled, so a start is timed from the first decoded phoneme. A correct lock is any lock on a prompted ayah; a false jump is a non-local lock change to an ayah more than 2 ayahs from the prompted passage.

| Measure | All scored clips | Clean clips | Bar |
|---|---:|---:|---|
| Correct ayah within 5 s of onset | 308/388 (79.4%) | 191/225 (84.9%) | ≥ 95% |
| Never correctly locked | 48 | 15 | |
| Wrong lock before the correct one | 21 | 0 | |
| False jumps / jumps | 0/0 | 0/0 | |
| Clips with any lock outside the passage | 21/388 (5.4%) | 0/225 (0.0%) | |
| Clips with a false jump | 0/388 (0.0%) | 0/225 (0.0%) | ≤ 2% |

Error clips are user-like but adversarial for the tracker: a substitution that forms another ayah is a correct lock on what was said, scored here as wrong. The clean-clip column is the fair gate number. Clips are one short passage, so the tracker almost never changes lock after its first one; a wrong first lock shows up in the outside-lock rows, not as a jump. Ambiguous tlog clips are not included: the bucket review (ticket 06) found the sampled ones are wrong-passage or no-recitation, with no passage truth to score against.

## E0 attribution of misses (free arm)

`tracker`: the declared-passage arm catches it, so the free arm lost it to the lock. Otherwise the greedy decode is aligned to the prompted passage: `evidence` when at least half the planted word's consonants decode as written (the teacher shows no error), `checker` when the decode shows the error and the checker still didn't confirm it.

| Class | Evidence | Tracker | Checker | Verified: evidence / tracker / checker |
|---|---:|---:|---:|---|
| wrong_word | 5 | 23 | 0 | 0 / 4 / 0 |
| omitted_word | 0 | 2 | 2 | 0 / 0 / 0 |

Checker misses, by what the declared-passage checker did at the site:

| Class: outcome | Misses |
|---|---:|
| omitted_word: raised as wrong_word | 1 |
| omitted_word: not raised | 1 |

## Limits

- Clips are single ayahs or short ranges (mean 10.7 s). Per-5-minute false warnings from short clips are not the acceptance bar; phase 2's long sessions are.
- Matching needs exact class and word, one-to-one, on each event's final state. An error still pending at clip end is a miss: short clips end before the 2.5 s confirmation delay.
- The evidence check is a heuristic on the greedy decode, not an as-said label. `checker` includes errors the decode shows only partly.
- Deliberate jumps are covered by a synthetic test only: the help clips have none, so this eval shows jump detection doesn't misfire on substitutions, not that it catches real jumps.
- End-of-session finalisation is off in free recitation. Measured with it on, the free arm went from 32 / 26 / 32 to 34 / 31 / 30 (hits / FP / misses) and clean false warnings from 8 to 11, and a clean tlog fixture gained two confirmed errors. In the declared arm it is worth it: off, 43 / 30 / 21 and 10 clean warnings.
- Pronunciation classes other than short vowel have fewer than 15 eval sites (ص/س 4, madd 1) and are `unsupported` here.
