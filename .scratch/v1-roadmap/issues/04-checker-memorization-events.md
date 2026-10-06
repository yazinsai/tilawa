# 04: Checker, memorization events

**What to build:** The checker as a separate consumer of tracker output. It emits versioned checker events for wrong, omitted and added words and for skips, with the confirm and resolve policy.

**Status:** done

- [x] Consumes the locked alignment and the evidence; emits `suspected | confirmed | resolved | uncertain` with error class and word location, using the versioned event schema.
- [x] Self-correction: a wrong word followed by a restart on the right word goes `suspected`, then `resolved`. A policy setting decides whether resolved errors count as mistakes; by default they don't.
- [x] Repeats produce no checker events.
- [x] Skip: a suspected omission is `resolved` if the words are recited before the next ayah ends, and `confirmed` otherwise.
- [x] Confirm delay is a policy setting: 2–3 s, or the next word onset.
- [x] Unclear evidence (low confidence or tracker not locked) yields `uncertain`, never `confirmed`.
- [x] Tests on corrupted teacher evidence (substituted, dropped and inserted words) plus clean fixtures. The tlog fixtures emit no confirmed errors.
- [x] False warnings per 5 minutes reported on the e2e sessions (professional audio, so it should be zero).

## How to run

```sh
pnpm --filter @tilawa/core test -- checker      # unit, synthetic, tlog and corrupted-teacher tests
pnpm --filter @tilawa/core e2e                  # prints checker false warnings per session and per 5 min
```

## Comments

- **Shape:** `createChecker(index, { policy, phrases })` returns `push(evidence, trackerEvents)`, `end(trackerEvents)` and `mistakes()`. `replayChecked` in `fixtures/` wires a tracker to a checker. The checker only acts while the tracker is locked. Lock events anchor it, and leaving `locked` or a lock outside its window ends every open error `uncertain`.
- **Alignment:** the tracker doesn't expose its per-phoneme path, so the checker runs its own. It is a weighted alignment of the phonemes since the anchor against a window (one ayah back, two ahead of the furthest word), with jump arcs to every word start. Back jumps are repeats and restarts. Forward jumps are skips. It re-aligns once per pushed chunk. The window commits at a word boundary above 240 phonemes, keeping the latest 120.
- **Judging:**
  - A word is fine if any visit covering at least half of it, with at least one consonant heard, got it right. A wrong visit followed by a right one is a self-correction, and a right one followed by a wrong one is a repeat that went astray.
  - A wrong word needs a cost above 0.3 of the word's weight, with 2 hard consonant edits (1 in a single-consonant word). Vowels, madd, qalqalah and ghunna marks, held or doubled sounds, and confusable pairs are cheap, because those are pronunciation, waqf endings or decoder noise, not a different word.
  - Unvisited words behind the head are an omission. Three or more words is a `skipped_passage`. A gap with evidence inside it is scored as a wrong word instead.
  - Inserted phonemes between two consecutive words, or a long run inside one word, are an added word.
  - Recognised phrases (the phrase asset plus the basmala) are consumed for free.
- **Confirming:**
  - Wrong and added words confirm once the next word has begun (3 matched phonemes into it), and, under the delay policy (default 2500 ms), once the delay has passed. If the session ends first, they end `uncertain`.
  - Omissions confirm when the reciter reaches the ayah after next, or at session end.
  - Errors decided while the tracker isn't locked, or with mean decoder confidence below 0.6, end `uncertain`.
  - Each error keeps one id through a class change, for example from wrong to added once the next word shows the attempt was an insertion.
- **Ambiguity:** an added word right before word W and a wrong attempt at W followed by W are the same audio. The alignment decides by resemblance. An attempt close to W re-aligns as W, wrong and then corrected, so it is `resolved`. An unrelated word stays an insertion, so it is `added_word`.
- **Results:**
  - e2e: 0 false warnings in 18.0 min of professional audio.
  - Getting there: the first version had 12. The fixes were basmala and phrase junk, memorization costs instead of phoneme costs, confirming only after a correct continuation, and scoring gaps that contain evidence.
  - tlog: 0 confirmed across all 62 fixtures.
  - Corrupted teacher evidence, on audio-verified clips spanning 5 or more words, with the word in the middle: wrong 4 of 5, omitted 6 of 6, added 4 of 5. Nothing was confirmed anywhere else.
  - Clips where the tracker locks only after the corrupted word, loses its lock over it, or never locks aren't scored. That is the tracker's part.
  - The added-word miss is an ambiguous span in 102:3–5, where "تعلمون" appears three times.
- **Known limits:**
  - Errors in the tracker's first span word aren't judged, because span edges are only accurate to about one word.
  - A self-correction pushed in the same chunk as the attempt is never seen as `suspected`. It goes straight to clean.
  - Waqf endings are only tolerated through the cheap vowel costs. The reference itself is continuous-reading.
