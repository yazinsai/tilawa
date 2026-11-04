# Committed local steps exclude an ayah from the absence ledger

Status: implemented. The single target is removed in the p46 phone diagnostic, with every other checker event unchanged apart from issue sequence IDs. No full practice score was requested. Leave this below-gate rider candidate for the supervisor's score and fold gate, as specified by the card.

Test set: lab-v1.5 practice, phone, listener `exp-listener-heard-word-must-be-lexical-41570706`.

## Prediction

Tracker −1, overall −1, from the false skipped-passage warning at p46 `54:36:1` at 121.84 s. This is a below-gate rider candidate; the prediction does not claim a standalone fold.

## Mechanism

A local run can step through an ayah without opening a checker window on that ayah. The tracker exports a coverage note on an existing position event after two consecutive next-word entries inside the same ayah at zero edit cost. Each entry must take the ordinary step from the previous word's end to the next word's first phoneme, retain the run origin, and leave the path's edit count unchanged. A cost-bearing entry, a skipped word, a jump, or a new local search resets the unfinished pair. Costs inside a word do not revoke an entry the tracker already committed. The note comes only from the local path actually taken, not from a shadow or an alternative fit.

`stepped-ayahs.ts` records these entries, and `tracker.ts` attaches each ayah once to an existing position event as `steppedAyahs`. Event comparison excludes that metadata, so it creates no extra position event. Pending notes can also ride the existing final event. `checker.ts` remembers the notes and excludes those ayahs only from the absence ledger's candidate list. Tracker timing, lock/drop/relocation behavior, windows and word-level issue rules remain unchanged. Numeric thresholds and the forward sum are unchanged. The experiment card, scorer, test data and lab tooling were not edited.

## Premise check

An instrumented copy of the baseline tracker ran with the same `chunks`, `trackerInput`, free mode and phrases as `runEngine`. Its local before/after states on p46 show:

| time | committed entry | phoneme offset | step cost | edits |
| --- | --- | --- | ---: | --- |
| 117.12 s | 54:35:7 → 54:36:1 | 567550 → 567551 | 0 | 24 → 24 |
| 117.84 s | 54:36:1 → 54:36:2 | 567558 → 567559 | 0 | 24 → 24 |

Both retain the same run origin, 567382. These are two consecutive in-order zero-edit word entries inside 54:36 before 121.84 s, so the card's kill criterion does not fire. The implemented tracker exports `{surah:54, ayah:36}` at 117.84 s.

This evidence supports word entries, not two complete zero-edit words. At 119.32 s, entry into word 3 also drops word 2's three ghunna-tail phones and costs 0.6 edits. The initial draft required complete zero-edit words and consequently certified no coverage for 54:36; it was corrected to the next-word-step mechanism specified by the card, without changing any cost or threshold.

## Validation

The final six-shard Modal run, request 6, built core and SDK and passed 723 core tests and 16 SDK tests; the existing SDK `help-e2e` test was skipped. `tools/autolab_tests.sh` accepted the existing Vitest `onTaskUpdate` deadline error on the checker shard after all its tests passed. All six shard statuses were zero.

Seven new tracker coverage tests cover consecutive entries, a cost-bearing entry, jump landings, nonconsecutive words, separate runs, ayah boundaries, insertions/deletions/path edits, and an integrated synthetic read versus a genuine whole-ayah skip. The existing ledger tests gained a case proving that a noted ayah is excluded while an unrelated gap still warns.

The same run compared baseline and candidate p46 engine events and applied the unchanged `scoreSessionEvents` with `anyClass: true`:

| p46 phone | baseline | candidate | change |
| --- | ---: | ---: | ---: |
| missed mistakes | 1 | 1 | 0 |
| false warnings | 1 | 0 | −1 |
| wrong calls | 2 | 1 | −1 |

The target `54:36:1` ledger skip, suspected at 121.84 s and confirmed at 122.48 s on baseline, is absent. The planted `54:36:4` omission remains a misclassified hit on the same alignment `skipped_passage` warning, confirmed at 136.88 s. The false `54:37:2` wrong word remains unconfirmed. The genuine `54:40:1` skip remains confirmed at 146.04 s.

All position events are identical after removing only `steppedAyahs`. All checker events other than the removed target are identical after removing only issue IDs; removing the target necessarily shifts later issue sequence numbers. Output and logs are in `.autolab/modal-out-6/0/`. Temporary diagnostic source copies and their driver were removed before committing.

## Limits

Targets won: 1 of 1 listed on this card. These are case-level measurements, not a full practice result. No full practice score has been requested, and no practice-wide overall/component delta is claimed. The supervisor scores the final branch and applies the fold gate.

Coverage records committed zero-cost word entries; it does not certify every phone of an ayah, move the tracker to a preferred copy, or re-check the correctness of a committed path. The p46 diagnostic establishes no loss on that case only. Reader robustness, losses in other sessions, fully clean sessions and studio audio remain for the supervisor's gate.
