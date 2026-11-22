# Ledger skip by read-versus-skip fit

Status: implemented. One practice score: overall −5, checker −5 (expected −3 / −3). The supervisor runs the fold gate.

## Prediction

Test set lab-v1.5, practice, phone. The three refrain-adjacent skips stay caught and the ledger's false skips on recited ayahs do not come back: checker −3, overall −3.

## Mechanism

The ayah-absence ledger's candidates were the right skips. Its decision was not: a phone-loop filler can explain any degraded speech, so a recited ayah whose consonants the channel blurred lost to it.

The candidate is unchanged in kind. After the tracker locks, an ayah counts as settled only once its last word has been locked or the alignment has read through that word. A flash of the skipped ayah, a few hundred milliseconds on word 1, does not settle it. Between two settled ayahs at most a checker's window apart, each unsettled ayah whose letter text is not the text of either neighbour is a candidate. An empty ledger, which is every in-order recitation, does nothing.

The decision is the parallel-phrase forward sum, not a filler. On the frames from the left neighbour through the right one, `H_read` is the neighbour words, then the whole candidate ayah, then the right neighbour, in mushaf order. `H_skip` is that path with the candidate removed. Both paths have the alignment's contiguous repeat arcs: a word or an ayah may be said again, with no extra penalty and no phone loop. A skip at word 1 is emitted only when `H_skip` leads by `PHRASE_SWITCH` (4 nats). A read path the frames cannot carry at all loses to a finite skip. A wrong or omitted word already open on that word 1 is the same detection misfiled, so it is replaced.

Channel blur hits both hypotheses on the same frames. A recited ayah still fits `H_read`, because `H_skip` has to force those words into the neighbours or into blank. A refrain said twice is a repeat of the neighbour, which both paths can do, and only `H_read` still has to place the missing ayah.

## What was measured

Stored phone posteriors, truth span of the ayah before the skip through the ayah after it. Margin is `nll(H_read) − nll(H_skip)`. The switch is 4 nats.

| Case | Frames | `H_read` | `H_skip` | Margin |
| --- | ---: | ---: | ---: | ---: |
| 26:109 | 143 | impossible | 8.6 | infinite |
| 55:29 | 334 | 856.7 | 8.4 | 848.3 |
| 77:33 | 216 | 394.9 | 7.0 | 387.9 |

All three clear the switch. The engine tracker does not settle those ayahs: it flashes word 1 for under a second and locks the following ayah. With the lock jumping that way, the engine confirms `skipped_passage` at 26:109:1, 55:29:1 and 77:33:1. On 77:33 the previous confirmed wrong word at that same word is the skip, and it is replaced.

Practice, one score, against the current main line (56 overall, checker 13):

| | main | this | change |
| --- | ---: | ---: | ---: |
| overall | 56 | 51 | −5 |
| checker | 13 | 8 | −5 |
| listener | 41 | 39 | −2 |
| tracker | 2 | 4 | +2 |

Misses 37 → 31, false warnings 19 → 20. The compare verdict was better. Leave-one-reader, clean sessions and studio are the supervisor's gate, not this score.

## Validation

- Unit: frames that are the neighbours alone prefer the skip by more than 4 nats. The same frames with the ayah present, and with that ayah blurred to 0.42 against blank, prefer the read. A lock that finishes the neighbours and jumps the ayah confirms the skip; a lock that reads the ayah confirms nothing there.
- Core and sdk tests: `bash tools/autolab_tests.sh` on six shards, every shard exit 0.

## Limits

The right neighbour is whatever ayah the lock settles next, up to the checker's window, not always the ayah immediately after the skip. The repeat arcs are what let a refrain copy inside that span explain itself. A jump wider than the window is not a candidate. Practice was scored once here; the fold gate has not run.
