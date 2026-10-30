# Keep added speech on a contiguous insertion path

Status: implemented; full practice result pending. User decides fold or discard.

Base: main 5716d49, with engine source identical to stored baseline b544715. Test set lab-v1.4.v3, practice only.

## Prediction before scoring

One fewer phone wrong call: recover p15 38:4:6. No predicted reduction in checker attribution: its perfect-transcript arm already produces an added-word candidate but the acoustic margin is 8.67 nats, below the unchanged 10-nat requirement. Preserve clean recitation and the studio copy. The original broader hypothesis and -2/-2 prediction are retained in lab-experiment.json and the ledger; they were refined from the actual traces before the first full practice run.

## Data review

The added إِن is supported by both saved label-blind independent transcripts. The user listened to the audio and confirmed on 2026-10-30, “Both sound as labelled,” in response to a question explicitly covering intact words and acceptable joins for this clip and p19's وَمِن. No bad-data flag.

## Cause and change

The phone alignment maps the beginning of إِن into a substitution for the following وَقَالَ onset. The global repeated-sound discount then lets the added word's repeated ن sounds follow that substituted onset cheaply. The added word never becomes a contiguous insertion candidate. This is the mechanism introduced by the last folded noise discount.

Keep a separate insertion DP state. Repeated sounds may extend inserted speech, or lengthen a matching reference sound, but cannot borrow the held-sound discount just because a substituted onset precedes them. Traceback follows the retained insertion path. Judging, acoustic support, and thresholds are unchanged.

The bounded-excursion control passes on main already, so the discarded branch's excursion rule was not copied. This change does not solve p19's split boundary, the قليلون additions represented as wrong second visits to already correct words, or the skipped passage aligned as a repeat.

## Validation

A synthetic p15 regression fails on main and passes with the change; the bounded-excursion control and all existing checker assertions pass (53 assertions total). Actual practice case replays on Modal confirm p15's phone candidate with observed ءِننن and 13.65 nats of support; the oracle remains uncertain as predicted. The other targeted misses remain unchanged.

Red: additions-red-a8b1070-dirty-20261030T174910Z. Initial full unit runs passed all assertions but returned a Vitest fork-worker onTaskUpdate timeout. The complete build and 53 checker assertions pass with a thread worker in additions-unit-threads-a8b1070-dirty-20261030T175803Z (two shards, both exit 0). Actual case probe: additions-case-probe-a8b1070-dirty-20261030T175216Z, shards p15/p19/p10/p20.

All replays and tests run on Modal through lab run. No scorer, label, listener or tracker changes.
