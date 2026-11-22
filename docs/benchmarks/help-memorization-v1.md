# Help recordings: provisional memorization evaluation v1

This is a diagnostic run, not an acceptance verdict. The listening audit is incomplete. Verified clips are a selected audit subset, not a representative test set.

Teacher evidence is inferred offline with up to 30 s of audio context, then replayed in 1 s chunks. This does not measure a streaming student's latency or accuracy.

Held-out speaker split is preserved. Audited incorrect/unsure labels and unreviewed known prompt swaps are excluded. Only approved swaps are applied to this evaluation snapshot. Letter-confusion clips are pronunciation negatives for the memorization checker, not wrong-word positives.

Labels: 413; excluded: 26; missing evidence: 0. Reference: ref-v2.

| Measure | All usable labels (provisional) | Audited and verified only |
|---|---:|---:|
| Clips / speakers | 387 / 29 | 59 / 21 |
| True / false positives / misses | 32 / 26 / 32 | 7 / 4 / 4 |
| Confirmed-event precision | 55.2% | 63.6% |
| Confirmed-event recall | 50.0% | 63.6% |
| Clean clips / minutes | 225 / 46.5 | 14 / 3.6 |
| Clean false warnings | 8 | 0 |
| Clean warnings per 5 min (short clips) | 0.86 | 0.00 |
| Any expected ayah locked by 5 s (coarse proxy) | 274/387 | 40/59 |
| No correct lock anywhere | 48 | 9 |
| Suspected skip labels detected | 5/26 | 2/7 |

## Confirmed errors by class (all usable labels)

| Class | Positives | Hits | Misses | False positives |
|---|---:|---:|---:|---:|
| wrong_word | 29 | 1 | 28 | 9 |
| omitted_word | 35 | 31 | 4 | 10 |
| added_word | 0 | 0 | 0 | 4 |
| skipped_passage | 0 | 0 | 0 | 3 |

## Limits

Matching requires exact class and word location, one-to-one, using the final state per event id. Suspected skip labels are reported separately; a confirmed event at a labelled suspected-skip site is not a false positive. End-of-clip uncertain events are misses for confirmed labels.

27/32 confirmed-label misses occur in clips with no correct lock. Remaining misses cannot yet be attributed between listener and checker; that needs as-said phoneme labels or manual evidence review.

No word/ayah onset truth is available here, so the lock-by-5-s figure is only a clip-level proxy and false jumps are not scored. Long clean sessions are still required for the false-warning acceptance bar. Pronunciation feedback is not implemented and is not scored by this report.
