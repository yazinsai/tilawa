# Tracker-run written-word fit veto

## Prediction

Listener −1 and overall −1 on lab-v1.5u practice phone audio, with an upside of −2. The card names five target wrong-word false warnings and uses listener `exp-listener-stack-be988c-6a17f391`.

## Mechanism

At a pending wrong word on any confirmation path, use the local tracker's independently timed span instead of the checker's greedy slot. Require committed steps onto the previous, accused and next written words, consecutively and in one locked run without a hold or restart. Align that span's tokens against the full written word and the checker's rival with existing hard and soft charges. Veto only when the written word needs no more hard edits than the rival. No cost constant, decoded token, tracker decision, position event or checker window changes.

The starting worktree does not export the per-word committed-step record claimed by the card. The implementation adds a private `WeakMap` record, attached by object identity to an existing position event when the next word commits. It does not add serialized event fields. Only actual word-entry commits count; acquisition locks do not. Run starts, holds, phrase rewinds, relocation, backward movement, skipped words and changes of local-run origin break continuity.

The checker captures the tokens in those completed spans before its existing tracking and alignment run. At emission it downgrades only an otherwise-confirmed wrong word to uncertain. Existing soft, held and consonant charges were extracted unchanged into `word-fit.ts`; full-word fits allow no free prefix or suffix. Equal weighted costs choose the fewest hard edits, and the veto compares hard-edit counts, not weighted costs. A frozen pending accusation retains its already-established veto. Internal ownership of a wrong word's absorbed surplus is preserved, preventing its changed outward verdict from enabling another addition warning.

## Validation

Modal run 1 used temporary observational probes on the original engine. Every target had consecutive same-origin committed events and a nominal tracker span whose written-word hard-edit count was no higher than the rival's. This coarse check did not inspect every intervening local step.

Modal run 3 replayed the built rule and the original tracker/checker side by side on all five complete phone sessions, using the card's listener, the shipped reference, phrase asset and chunking. The stricter live continuity guards retain four of the five witnesses:

| Target | Tracker span (seconds) | Written / rival hard edits | Built target won |
| --- | --- | --- | --- |
| realuser-v1-sob-027, 84:25:1 | 92.32–93.16 | 0 / 0 | yes |
| realuser-v1-sob-049, 92:13:1 | 35.80–36.32 | 0 / 0 | yes |
| realuser-v1-sob-053, 88:2:2 | no qualifying committed span | — | no |
| realuser-v1-tlo-127, 2:29:16 | 8.64–9.28 | 0 / 0 | yes |
| lab-v1.5-p55, 26:174:8 | 99.64–102.76 | 0 / 1 | yes |

Every position event, including EOF, is byte-identical to the original engine. The complete checker output arrays differ only by confirmed-to-uncertain wrong-word verdicts: one change in each won session and none in the remaining session. All nine new unit tests passed on each diagnostic shard. Raw results are `.autolab/modal-out-1/{0..4}/premise.json` and `.autolab/modal-out-3/{0..4}/built.json`.

Modal run 4 built core and SDK and passed all six test shards: core 743 passed, SDK 16 passed and one skipped. One core shard encountered the known Vitest worker RPC deadline after every test passed; the repository's unmodified `autolab_tests.sh` accepted that case, and every shard exited zero. Unit tests cover span closure, private event metadata, soft entries, holds, restarts, backward movement, skips, missing committed neighbors, full-word boundaries and genuine rival consonants. The temporary probes, copied baseline sources and replay drivers were removed.

Four built target wins exceed the user's three-target continuation minimum. Practice score 1 is pending; no constants were tuned against practice.

## Limits

These are token and event-trace measurements, not an independent audio review or a correction of benchmark labels. Three target wins are in the separate clean-realuser slice; only p55 is a planted-session target win. The target result alone therefore does not establish a headline gain or fold eligibility. Native tracker event objects carry the private witness; oracle or serialized position events supply no witness and retain their existing behavior. The experiment's card, scorer, data, evidence and manifests are unchanged.
