# Channel confirmation after pausal-final manner rekey

Status: implemented and committed; both targets recovered in full phone-session replays. Left for the
supervisor's practice score and fold gate. No practice-score call was made by this builder.

## Prediction

Experiment `listener-manner-rekey-channel-confirm`, listener, explore, lab-v1.5 practice with listener
`exp-listener-heard-word-must-be-lexical-41570706`. Predicted listener −2 and overall −2. The card is unchanged.

The two intended wins are the missed wrong words at p78 11:11:10 and p50 37:84:5. The card's premise
requires an attested rival of the heard final manner to beat the written pausal word on both slots.
The predicted gain alone is below the fold gate.

## Mechanism

Reapply `d9c192a0`'s final-slot manner rekey. Its manner table, manner/confidence helpers, slot predicate
and both rekey paths match the original source byte for byte. Existing formula-slot decisions,
connected-madd decisions and ledger/seam merging remain on current main.

After either rekey path fires, compare the slot's own greedy tokens before the labelled pause against
the written isolated pausal form and attested isolated mushaf pronunciations. Both sides use the existing
waqf-edge strip and held-phone collapse, then the unchanged folded `LISTENER_CHANNEL` and `spanLogP`.
Rivals need at least one consonant and the heard final manner; the written word's forms and words or
prefixes in its ±3 neighbourhood are excluded with the existing lexical seam test. No candidate is
generated from practice labels or references.

Keep the rekey accusation only when the best rival has strictly higher channel log probability. Ties,
no rivals and a disabled channel drop it. This filters both the direct wrong-word path and the
speech-holding omission refile. Surviving accusations retain the original posterior SUPPORT and weak
evidence gates. No numeric decision constant or channel count changes. The decoded string and tracker
remain unchanged. Vocabulary and repeated slot comparisons are cached.

## Validation

Modal run 2 built core and SDK and passed all six test shards: 736 core tests and 16 SDK tests, with
one SDK test skipped. One core shard hit the documented Vitest `onTaskUpdate` RPC deadline after all
tests passed; the supplied `autolab_tests.sh` wrapper accepted that exception and every shard exited 0.

Unit coverage includes both original rekey paths and their pause, manner, confidence and posterior
controls, plus strict channel wins, ties, written-word explanations of manner contradictions, candidate
manner, own forms, neighbours and prefixes, one-consonant rivals, and waqf/held-phone normalization.

Modal run 1's default true-position trace confirms p78 11:11:10 at 147.36 s with unchanged confidence
0.5997 and observed `مَںں`. It rekeys p50 37:84:5 to a suspected wrong word at 52.24 s with observed
`مِل`; that trace's short time window does not settle its eventual confirmation.

Modal run 7 applied the shipped comparison to those exact observed strings, using the original manner
table and strip. Both have an eligible rival of the heard final manner that is strictly more probable:

| Slot | Observed | Written log P | Best rival | Rival log P | Rival advantage |
| --- | --- | ---: | --- | ---: | ---: |
| 11:11:10 | `مَںں` | −16.128257 | `مَا` | −2.096470 | 14.031787 |
| 37:84:5 | `مِل` | −20.731546 | `مِثل` | −4.200387 | 16.531159 |

The first observed string becomes `مَ` after the inherited waqf-tail strip. These rivals attest the
required manner and win the channel comparison, but neither identifies the planted word من. The
card's channel-evidence premise therefore survives this check.

Modal run 8 replayed both complete phone sessions with `runEngine` and `runTruePosition`, then repeated
them with only this experiment's checker changes reversed to baseline `0ce594b9`. It verified that
reversing the changes reconstructed that baseline exactly. The scorer, labels and evidence were
unchanged. The actual streamed slot comparisons matched the numbers above in both arms.

| Target | Main | Experiment, both arms |
| --- | --- | --- |
| p78 11:11:10 | wrong word ends uncertain at 147.36 s | wrong word confirmed at 147.36 s |
| p50 37:84:5 | omission ends uncertain at 66.12 s | wrong word confirmed at 84.76 s |

The whole-session missed/false counts were identical between product and true-position arms:

| Session | Main missed / false | Experiment missed / false |
| --- | ---: | ---: |
| p78 | 3 / 1 | 2 / 1 |
| p50 | 2 / 1 | 1 / 1 |

Both intended mistakes are won, with a net reduction of two calls across these sessions and unchanged
false-warning counts. These are target-session checks, not a practice-set score. The full replay also
passed the final 21 manner/channel unit tests. Run 2's full suites remain the build validation.

The card names two targets and requires both to have winning channel rivals. The generic builder
stop instructions separately name a three-target minimum. Clarification was requested; with no answer,
the builder used the card's specific two-target premise, as stated during the run, and retained the
change once both targets were recovered. It adds no third target or rider and requests no fold.

Diagnostic attempts 3 and 4 never ran because the supervisor excludes local `.autolab` and `.scratch`
helpers. Attempt 5 exceeded the Modal command-size limit; attempt 6 used a tsx eval import path that
could not resolve. Runs 7 and 8 corrected these transport issues using direct commands, explicit TS
imports for the short check and a small reverse patch for the full comparison. None of these attempts
was a practice score or changed the submitted engine implementation.

## Limits

This is conditional evidence from the already folded channel table, not an independent acoustic
measurement. The best rival need not identify the planted word; phone blur can erase that identity.
The filter cannot rescue unaccused slots or bypass SUPPORT. A target-session replay establishes those
cases only; aggregate practice results and the reader, studio, clean-session and incident-loss gates
require the supervisor's scoring.

The p50 warning is deferred about 34.5 seconds from the planted word's onset; retaining its accusation
does not improve that inherited confirmation timing. Equal session false-warning counts also do not
prove identical warning keys or zero losses elsewhere. No studio or fully clean-session gate was
measured here. The preregistered overall −2 remains a prediction below the fold threshold.
