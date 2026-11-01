# Proclitic residue at skip edges

## Prediction

Experiment `checker-proclitic-skip-edges`, checker, lab-v1.5 practice only. The unchanged card predicts four fewer checker-attributed and overall wrong calls: repair the misplaced skip warnings at `26:102:7` in p24 and p73, recovering the paired missed skips at `26:103:1`. The p24 `26:131:2` warning is speculative and is not required for the prediction.

Kill the experiment if either primary seam lacks directly adjacent proclitic residue, or replay moves any currently caught planted skip or omission. No numeric thresholds are being tuned.

## Step 0

Baseline diagnostic run 3 used the card's listener, `exp-listener-context-free-gap-decode-62ac8b5`, phone evidence and oracle positions. Both traces support the proposed mechanism:

| Session | Article onset | Stem onset | Alignment before repair |
| --- | ---: | ---: | --- |
| p24 | 75.36 s | 75.60 s | A one-phone, non-counting visit to `26:102:7` holds ل; the immediately following stem counts as `26:103:8`. |
| p73 | 98.00 s | 98.20 s | The same split: ل is left at the source edge and the stem is bound to `26:103:8`. |

The lam is not an insertion step: it is a cut-off source visit that fails the existing word-coverage rule. There is no labelled pause between it and the stem. Requiring literal insertion steps would miss both cases. The reunited phones are an exact supported pronunciation of المؤمنين, distinct from the later مؤمنين. This is a trace finding, not an independent listening review or a label change.

## Mechanism

Build and cache surface prefix/stem candidates from each reference word's mushaf letters and supported pronunciations. Recognize ال (including sun-letter assimilation), و, ف, ب, ك and ل. Retain consonant doubling and collapse only the engine's existing held-sound variants. Both spelling stems and pronunciation tails must match; a leading letter alone does not establish a proclitic contrast.

Only an already proposed skipped passage can trigger repair. Join its unclaimed leading phones to the next counted word at the far edge. Require an exact source-word pronunciation, a different surface proclitic on the bound word, no labelled pause or phrase junk, and no intervening counted visit. Evidence already valid as a complete pronunciation of the bound word is ineligible.

Replace only the bounded alignment stretch, preserving every observed index once and in order. Rebind the complete word to the source edge and re-run the existing judgment and confirmation pipeline. Emission thresholds, tracker, listener, scorer and benchmark inputs are unchanged.

## Validation

- Run 7: TypeScript core build and all 18 focused tests passed. The p24 trace moved the confirmed warning from `26:102:7` to `26:103:1`, with the full unread ayah as its expected text.
- Focused tests cover all listed single proclitics, two sun-letter articles, exact identity, incomplete or mismatched stems, missing word text, a genuine skip including the edge word, absent article evidence, a labelled pause, clean recitation and an ordinary single-word omission.
- Run 9 replayed both primary seams. Both confirmed warnings moved from `26:102:7` to `26:103:1`, preserving suspicion and confirmation times (p24: 79.24/84.84 s; p73: 101.88/107.40 s). The expected skip text now includes the complete unread ayah.
- Practice requests 1 and 2 returned identical results, matching the prediction: overall **73 → 69** wrong calls (misses **48 → 46**, false warnings **25 → 23**); checker attribution **28 → 24**, listener **38 → 38**, tracker **7 → 7**. Two of the three permitted practice-score requests were used; the second was requested before the first returned. The code was not adjusted against either result.
- Run 10: core and SDK builds passed; **656 core tests and 16 SDK tests passed**, with one existing SDK test skipped. The checker suite hit the known Vitest `onTaskUpdate` RPC deadline after all its tests passed; the repository's supplied `tools/autolab_tests.sh` accepts that specific condition, and all six shards exited zero.
- Run 12 completed all 20 shards successfully: **400 unique comparisons** covering all 100 practice sessions, phone and studio evidence, and actual-tracker and true-position arms. Each comparison ran the same checker source with the new repair disabled and enabled. It compared confirmed warning locations, classes, expected/observed text, and suspicion/confirmation times; confidence and generated event IDs were excluded.
- The only eight changed comparison rows are the two primary seams across the two variants and two tracker arms. Each replaces the misplaced warning with the correct skip boundary and preserves its timing and the number of confirmed warnings. Every other confirmed call is unchanged. The speculative p24 `26:131:2` case is unchanged. All **26 fully clean sessions** retain identical confirmed calls.
- Neither kill criterion fired: both primary seams contain adjacent article residue, and no previously caught planted skip or omission moved. The phone result is four incidents won (two removed false warnings and two recovered misses), with no lost incidents. Retain implementation `4abfb2de` for the supervisor's fold gate.
- Operational notes: the earlier test run 8 eventually also passed. Run 11 failed before replay because the sandbox snapshot omits the experiment card; run 12 received the listener explicitly. Runs 13 and 14 only read saved replay progress. Temporary diagnostic drivers were removed from the proposed change; no scorer, lab tooling, card, manifest, label or evidence was edited.

## Limits

The segmentation is conservative surface analysis of available reference text and pronunciations, not a morphological parser. Stacked proclitics and degraded non-exact word forms can remain unrepaired. This implementation repairs the word immediately at a proposed gap's far edge; it does not search arbitrary interior words or change skip-placement costs. It does not infer unlabelled silence from CTC blank duration.

The card's assumption that studio audio would propose no eligible skips was incorrect: studio has the same two misplaced boundaries, and receives the same corrections. No other studio confirmed call changes. The replay comparison is diagnostic evidence, not an independent audio review or a replacement for the supervisor's fold gate.

The partial p73 true-position trace also contains a new `wrong_word` suspicion at 113.88 s that resolves at 114.12 s without confirmation. The practice score counts confirmed calls; this transient event remains a behavior to keep visible in review.
