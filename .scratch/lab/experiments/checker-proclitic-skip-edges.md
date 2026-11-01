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
- Practice request 1 returned the predicted improvement: overall **73 → 69** wrong calls (misses **48 → 46**, false warnings **25 → 23**); checker attribution **28 → 24**, listener **38 → 38**, tracker **7 → 7**. The code was not adjusted against this result.
- Run 10: core and SDK builds passed; **656 core tests and 16 SDK tests passed**, with one existing SDK test skipped. The checker suite hit the known Vitest `onTaskUpdate` RPC deadline after all its tests passed; the repository's supplied `tools/autolab_tests.sh` accepts that specific condition, and all six shards exited zero.
- An incident-level replay is pending. Earlier test run 8 and practice request 1 had no returned result when work resumed; test run 10 and practice request 2 retried those outstanding checks before the first score arrived. Run 8 subsequently also passed. Run 11 failed before replay because the sandbox source snapshot omits the experiment card. Run 12 passes the listener explicitly and compares confirmed warning locations, classes, text and times with the repair disabled/enabled on practice phone and studio evidence, through both the actual tracker and oracle positions. The temporary diagnostic drivers are excluded from the proposed commit.

## Limits

The segmentation is conservative surface analysis of available reference text and pronunciations, not a morphological parser. Stacked proclitics and degraded non-exact word forms can remain unrepaired. This implementation repairs the word immediately at a proposed gap's far edge; it does not search arbitrary interior words or change skip-placement costs. It does not infer unlabelled silence from CTC blank duration. Clean recitation with no proposed skip cannot trigger this path; empirical clean-session and studio behavior still require replay.

The partial p73 true-position trace also contains a new `wrong_word` suspicion at 113.88 s that resolves at 114.12 s without confirmation. The practice score counts confirmed calls; this transient event remains a behavior to keep visible in review.
