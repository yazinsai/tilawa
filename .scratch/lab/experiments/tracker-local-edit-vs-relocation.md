# Local-edit hypotheses against mid-session relocation

Status: killed at step 0. No engine change, no practice score.

Test set: lab-v1.5, practice only. Listener: spliced-clean-int8, phone. Branch base 364bb97.

## Prediction, before scoring

Three fewer tracker-attributed and overall phone wrong calls, from keeping a one-word mistake at the current anchor instead of relocating onto a look-alike verse. Clean sessions unchanged. The p33 26:222 win was already marked speculative on the card.

## Step 0

Run `pos-trace-364bb97-20261101T065657Z`. Same streaming path as the scorer: free tracker, phrases, 40 ms chunks, posteriors stripped before the tracker. Neighbourhood is the tracker's own jump threshold: a lock is out of neighbourhood when its ayah is in another surah or more than 2 ayahs from the word being recited (`NEAR_AYAHS`). A non-locked event is not a relocation; the checker drops its anchor on anything but `locked` and marks open issues uncertain.

| case | planted | committed lock outside the neighbourhood |
| --- | --- | --- |
| p46 54:36:4 فَتَمَارَوْا omitted at 120.31 | omitted_word | no |
| p66 9:3:21 وَإِن at 51.84 | wrong_word | no |
| p33 26:222:1 تَنَزَّلُ skipped at 230.09 | skipped_passage | no |

Zero of the three. The kill criterion is an out-of-neighbourhood relocation in fewer than 2.

### p46 54:36

The tracker stays on 54:36 through the omitted word (120.44, still 54:36:4 while the audio is already 54:36:5 بِٱلنُّذُرِ). At 121.84 it locks on 54:37:9 وَنُذُرِ, one ayah away, span restarted there. That is a local-window jump, not a global relocation: 54:37 is inside `NEAR_AYAHS`, so the pending-jump path cannot commit it. It then steps into 54:38 while the reciter is still on 54:37, goes `lost` at 125.04, and reacquires on the true 54:37:4. It never locks on 54:23 or 54:33.

The checker suspects an added word at 54:36:4 and resolves it, then suspects a skipped passage there and goes uncertain at 124.96, when the lock drops. No confirmed omission.

### p66 9:3

Locked on 9:3 through the wrong word (51.88, 9:3:21). At 52.72 the run restarts at 9:3:12, still the same ayah. `lost` at 56.48, on فَٱعْلَمُوٓا۟. Reacquisition is `ambiguous`: the shortlist at 58.64 ties 9:2:6–7 with 9:3:23–24 at cost 0.071, and also several وَٱعْلَمُوٓا۟ sites in surah 2. Those far words are the top ambiguous candidate, never a lock. From 59.04 the top candidate is 9:2 (one ayah away) until 61.40, when it locks on the true 9:3:28 وَبَشِّرِ. The wrong-word warning is suspected at 55.04 and uncertain at 56.32, i.e. when the lock drops, before 9:2 is even the top candidate.

### p33 26:222

Locked through 26:221. Recent cost climbs to 0.575, then `lost` at 229.20, just before the skipped ayah. Reacquisition shows single-word ambiguous and acquiring candidates (شَيَـٰطِينِهِم, يُلْقُونَ at 3:44 and 26:223 together). It locks on the true landing 26:223:2 at 231.36. The skipped-passage warning is suspected at 26:221:5 and resolved at 228.52, while the tracker is still locked on 26:221, before the loss. No parallel verse (45:7 أَفَّاكٍ أَثِيمٍ) ever becomes the lock.

## Why the proposed change does not apply

The card's mechanism is a global relocation: after a one-word mistake the current anchor pays the full mismatch, a parallel verse matches the rest, and `pending` commits because nothing prices "stayed put and edited". Step 0 does not show that commit. The two misses that move the position do it inside the local window or by dropping the lock, and the checker already treats a dropped lock as uncertainty. Gating only the global jump would not change these three traces. p33 was speculative and the trace agrees.

No tracker edit, no new tests, no practice score.
