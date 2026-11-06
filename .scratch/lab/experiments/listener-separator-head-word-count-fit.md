# Separator-head closed word-count fit

Status: implemented; three occurrence-matched phone target recoveries. Official
stored-evidence replay is unchanged; leave the change for the supervisor's fresh decode
and fold gate.

## Prediction

The card predicts three of nine listener-charged seam additions recovered, with
about one loss: listener −2, overall −1 on lab-v1.5u. The registered card and its
listener selection are unchanged.

## Mechanism

Use the retained `word-sep-v2` phone-plus-separator CTC head from
`listener-ctc-word-separator-seams`. Compare two closed forward sums over the
primary tracker's step span: entry onto the preceding word through entry onto
the second following word. H_N reads the preceding word, separator, word 1,
separator, word 2. H_N+1 inserts a free-phone slot between two separators before
word 1, requiring at least two collapsed consonant emissions. Both choices
include the same connected/pausal text forms and separator-bearing contiguous
word and right-side repeat arcs. No word identity is supplied for the extra slot.

This first stage measures the head's full-vocabulary posteriors, not its greedy
boundary mask and not reconstructed top-k phone rows. Primary tokens and the
tracker are read unchanged. Labels only select diagnostic targets and exclude
planted incidents from the clean comparison; they never place the scoring span.
The card's threshold is the maximum clean phone/studio seam margin, with a strict
inequality. A target below even a partial clean ceiling cannot beat the complete
clean maximum.

The implementation carries full auxiliary rows on the first primary token of
each listener push, retaining blank-only rows until a token can carry them.
Cold-gap outputs discard their auxiliary rows; the original full-stream rows
are preserved through gap replacement. The checker reads these stored rows
on the primary tracker clock. A qualifying slot goes through the existing
fragment/back-run/prefix and non-Quran guards and the folded added-word issue
path. Its calibrated count judgment supplies certainty instead of requiring
the primary string to contain an extra word. Main logits, token selection and
tracker inputs are unchanged.

## Validation

The probe's CTC sums have unit checks against exhaustive small-alphabet frame
paths. Other checks require both extra separators, distinguish a held consonant
from two emissions separated by blank, explain separator-bearing repeats, and
abstain on missing rows or impossible complete text.

Practice score request 1 of 3 was submitted at 2026-11-06T21:26:47Z after the
three occurrence-matched phone recoveries. It completed against implementation
commit `4d01bbee`: overall 24 wrong calls (13 missed, 11 false), listener 23
(13 missed, 10 false), tracker 1 and checker 0. Every change was zero, verdict
noise. Two requests remain unused.

The parent teacher-insert experiment documents that `serve_practice` calls
`lab.py run` on the card's existing listener evidence. This card still names
`exp-listener-teacher-insert-marginal-gate-ad6eb8c0`; its stored evidence predates
the word-count output. The new detector abstains without complete auxiliary
rows, so this compatibility replay does not measure the exported head's
population result. The supervisor's post-implementation fresh SDK decode
reads `.autolab/model.json` and will supply those rows. No narrowing follows
from the unchanged stored-evidence result.

Run 19 independently audited all 497 stored evidence files for the card's
listener in the supplied sessions directory: **zero word-count packets**.
The exported bundle's model SHA and 1.955775473547149-nat cutoff matched the
validated export, and both existing teacher asset files were present. This
read-only audit did not decode or score sessions. Its initial invocation could
not read the experiment card because the card is absent from the Modal
snapshot; the corrected invocation passed the unchanged selection explicitly.

Both package builds and the unchanged six-shard test harness passed in runs
13 and 14: 764 core tests and 27 SDK tests, one optional SDK integration test
skipped. The existing harness accepted its documented Vitest RPC deadline
after every test in that shard passed. Run 15 passed the additional direct
checker test with the other eight new tests: all 756 existing core tests plus
nine new core tests have passed. The two added SDK tests check unrounded row
transport with unchanged primary output and a hard failure on a missing head
or missing calibration. Initial probe path and Float32/Float64 interface
failures were corrected before these successful runs; they yielded no scores.

## Acoustic measurements

Modal run 7 decoded phone and studio audio for eight selected sessions with
the retained original `word-sep-v2` export, SHA-256
`27c85f6ffcfc5dcbd691a4be5f8cfef71ca4de1598bef87089ee244b57b269a0`.
All eight shards exited zero, producing 544 eligible seam margins. The head's
full fp32 log probabilities are captured before top-k truncation or rounding.
The primary baseline remains
`exp-listener-teacher-insert-marginal-gate-ad6eb8c0`.

The five targets supplied in the task have these log-forward-sum differences
(`log P(H_N+1) - log P(H_N)`, nats):

| Target | Phone margin | Studio margin |
| --- | ---: | ---: |
| p33, 26:208:1 | −3.602 | 9.126 |
| p56, 32:21:1 | 7.441 | 10.142 |
| p31, 4:170:12 | 0.096 | −7.720 |
| p07, 37:36:1 | 14.495 | 20.368 |
| p79, 20:40:25 | 32.062 | 40.073 |

Modal run 9 independently selected all 211 supplied help-train takes labelled
`kind: none`, `split: train`, `scored: true`, with no events, repeats, unsupported
labels or audit exclusion. All 12 shards exited zero. The unchanged primary
listener and free tracker produced 136 eligible seams. Their maximum margin
was **2.0151743213 nats**, on a natural phone recording; the highest laptop
margin was −0.2773876892. This sample uses training speakers only. It is a
lower bound on the complete clean gold-excluded phone/studio maximum, because
the original continuous training corpus and its studio/augmented copies are
not supplied here. It is not a practice-fitted cutoff.

The supplied five have exactly three phone margins above that lower bound.
Additional probes at p48 12:76:1 and p57 51:18:1 / 51:20:1 measured −5.200,
0.393 and −10.540 respectively, all below it. Run 10 confirmed that the five
supplied locations and these three additional locations had no baseline
confirmed added-word warning at the same word. This is selected target
analysis; it does not enumerate or score the entire practice population.

For context only, the selected practice sessions contain 440 seams with no
overlapping planted event or labelled repeat. Their largest margins were
0.0708934281 on phone and 7.6285187815 on studio. These values are not used
to construct or adjust a threshold.

## Attached model and native replay

Run 14 attached the same frozen head to the unmodified primary int8 graph.
Nineteen head nodes add 183,263 bytes and no cache states. The model is
67,943,007 bytes, SHA-256
`e08cc19ff33c46ee2a09f08a5c5e68390869d594cc345b0dc088c6ec2c699574`.
Its separate full-vocabulary output is `word_count_log_probs`; primary
`log_probs`, tokens and cache outputs retain their original graph and weights.
The unchanged folded teacher graph and external weights were copied from
`listener-teacher-insert-marginal-gate/bundle` into the bundle as well.

Run 15 recalibrated the attached head on the same 211 clean help-train takes:
136 seams, maximum **1.9557754735 nats**. Run 16 included all 30 usable
training-speaker repeat takes with no planted or unsupported errors. They
yielded only one eligible seam, margin −5.5039574292. The combined maximum
therefore remains 1.9557754735 over 137 seams from 241 recordings. Repeat
coverage is limited by the short takes and the free tracker's acquisition and
consecutive-step requirements. There is no practice-fitted cut.

Run 15 used the initial, slightly stricter fp32-derived cut of 2.0151743213.
It confirmed all three passing phone targets through the ordinary checker:

| Target | Attached-head margin | Slot phones | Preliminary confirmation |
| --- | ---: | --- | --- |
| p07, 37:36:1 | 14.798 | سَء | yes |
| p56, 32:21:1 | 7.662 | نقِۦۦ | yes |
| p79, 20:40:25 | 32.130 | فَصصَفَاا | yes |

These are identity-free phones, not lexical claims about the inserted words.
The replay augments the card's recorded primary evidence with fresh full-stream
head rows. All tracker events compare equal, and native primary output checks
passed. It is not a fresh complete SDK decode of the practice population.
Run 17 exported the combined attached-head calibration and passed all nine
new core tests. The unchanged scorer occurrence matcher confirmed all three
phone recoveries: p07 37:36:1, p56 32:21:1 and p79 20:40:25 changed from
unmatched to matched. The same occurrences were also matched in studio; p07
was already matched there. Previously matched additions in these sessions
remained matched. All six primary output comparisons passed over the complete
recordings (290.84, 194.36 and 217.8 seconds per variant), and every replay's
tracker event sequence remained identical. This establishes selected-case
recoveries, not a population gain or a clean-session safety result.

The selected bundle is `/scratch/listener-separator-head-word-count-fit/model-word-count`,
declared through `.autolab/model.json`. The model's SHA-256 is unchanged from
run 14, and its final manifest cut is 1.955775473547149 nats.

## Limits

The empirical envelope uses supplied gold-excluded training-speaker phone and
laptop audio. The original continuous studio training bank and its augmented
copies are unavailable, so this sample does not establish that envelope for
every clean studio or degraded-phone seam. The head's slots sometimes borrow
a neighbour's tail, and the fixed three-word contest can favour an extra slot
when a written pronunciation fits poorly. The existing guards and population
fold gate remain necessary.

Only a declared, calibrated model activates this path. Existing stored practice
evidence has no auxiliary rows; fresh supervisor decoding is needed to measure
the bundle's population score, studio effects, new clean-session warnings and
won/lost tradeoff. No fold, scorer/tooling change, supplied manifest/evidence
edit or experiment-card edit has been made.
