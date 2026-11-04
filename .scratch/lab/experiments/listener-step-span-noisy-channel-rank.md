# Noisy-channel rank over tracker-step seams

Status: practice-scored once. Left for the fold gate.

## Preregistered prediction

Test set lab-v1.5, practice, listener `exp-listener-heard-word-must-be-lexical-41570706`. 2 of 4 seam additions won, about 1 loss: listener −2, overall −2.

## Mechanism

The killed noisy-channel ranker never scored, because its span needed a zero-cost word-1+word-2 greedy fit and that fit exists on 1 of 5 targets. The tracker steps through every one of these seams. This card takes the greedy tokens from the tracker's step onto the previous ayah's last word to its step onto word 2.

Over that span every hypothesis is the previous word plus word 1, and the candidates differ by one mushaf word inserted between them. The score is the Viterbi log probability under P(heard token | written phone), counted by aligning shipped decodes of 48 gold-excluded training clips (studio, and the same clips through the shipped room-and-noise augmentation) to their canonical phones. Additive smoothing is α = 1/2. Competitors are that written pair with nothing inserted, each word and prefix within ±3, a contiguous back-run ending at the previous word, non-Quran phrases, and a no-word insertion model.

A non-neighbour word is emitted on the folded added-word path (`lexicon`, so the 10-nat support margin is not consulted) only when its log-odds over the best competitor exceed the maximum the same score reaches on any clean gold-excluded phone or studio seam. That cut is 10.49 nats, from 117 such seams, set by the estimator and not by the practice set. A blurred consonant is a substitution at its measured rate. The decoded string and the tracker are unchanged.

## What was checked before the score

Tracker steps exist on all four targets. The planted phones sit in the step span. Under the estimated channel the log-odds clear the clean-seam cut on all four, and the planted string itself is the best non-neighbour on three:

| session | heard span | planted (collapsed) | best non-neighbour | log-odds |
| --- | --- | --- | --- | --- |
| p50 37:99:1 | `ءَسَلِۦۦنوَقِفُۥۥهُم` | `وَقِفُۥهُم` | same | 28.9 |
| p19 51:33:1 | `ُجڇرِمِۦۦنوَمِںںں…عَلَيهِم` | `وَمِن` | `وَمِں` | 36.8 |
| p33 26:208:1 | `ُمَتتَعُۥۥنفَرِۦۦموَمَاء` | `كَرِۦم` | same | 44.4 |
| p49 26:123:1 | `رَحِۦۦمكَرِۦۦمكَذَبَتعَ` | `كَرِۦم` | same | 42.8 |

p19's winner is the ghunna reading of the same onset (`ن` heard as `ں`), not the canonical `وَمِن` string. It still clears the cut, so the seam emits an added word at 51:33:1. The channel was counted on 5246 written phones (467 deletions, 27 insertions). A synthetic one-consonant blur of قبلهم before 44:10:1 confirms; the same seam with nothing inserted stays quiet.

Core and sdk tests passed (one shard hit Vitest's known RPC deadline after every test passed).

## Measured practice result

One practice score.

| | main line | this | change |
| --- | ---: | ---: | ---: |
| overall | 40 (26 missed + 14 false) | 42 (24 missed + 18 false) | +2 |
| listener | 36 (24 missed + 12 false) | 33 (20 missed + 13 false) | −3 |
| tracker | 1 (0 missed + 1 false) | 4 (2 missed + 2 false) | +3 |
| checker | 3 (2 missed + 1 false) | 5 (2 missed + 3 false) | +2 |

Listener misses fell by 4, in line with the four spans that clear the cut. Four new false warnings, two new tracker misses, and one new listener false warning leave the overall 2 worse. That is noise against a gate that needs overall −3. The cut was not moved after this score.

## Limits

The cut is the max over gold-excluded clips degraded with the training room-and-noise chain, not the lab phone simulator, so practice phone seams can sit above it. p19 emits `وَمِں` rather than the canonical `وَمِن`; the warning is still an added word at that word. Ranking runs once per ayah seam, on the first step onto word 2.
