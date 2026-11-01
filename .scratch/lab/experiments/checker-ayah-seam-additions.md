# Ayah-seam additions

Status: stopped at step 0. No engine change. Practice was not scored.

## Preregistered prediction

Test set lab-v1.5, practice only. The card predicted 6 of the 11 word-1 added-word targets won, checker −5, overall −4. The kill criterion was to stop before any rule if fewer than 6 of those 11 had window-trimmed gap phonemes carrying 2 or more consonants.

## Mechanism that was going to be tested

The checker does not open a fresh alignment at each ayah. The window already starts one ayah before the anchor and runs several ayahs ahead. `nextWord` walks the mushaf, skipping empty words only, so the last word of an ayah and word 1 of the next ayah are consecutive reference words. `judge` already opens an added-word issue for an insertion block between two consecutive visits of one run, and the folded joint-additions repair already refits a consecutive pair with an intervening insertion when both words stay explained.

The proposed change was that same fit over the seam span, sending residue that neither seam word owns to the existing added-word confirmation. It was not going to change the non-Quran phrase cost or use a phrase lexicon. That only prices residue. It does not move a diagonal match off a reference word that already owns the phonemes.

## What step 0 showed

For each target, decoded phonemes whose start time falls in the truth gap — after the previous ayah's last word, before word 1 — were labeled from the alignment path `judge` consumes. A phoneme is window-trimmed when that path drops it as lead, as an insertion before the first visit, as trailing speech, or never aligns it. A match or insertion on a reference word, or a between-word insertion block, is inside a window.

Three arms, one Modal run (`seam-gap-e1e61e9-dirty-20261101T063436Z`), eight shards, all exit 0. The tracer was temporary and is not in the tree. The per-target rows were keyed by session, verse and time; they are not kept. Re-aggregation of that run:

| Arm | Targets with ≥2 window-trimmed gap consonants |
| --- | ---: |
| Oracle evidence, oracle position | 0 / 11 |
| Phone evidence, oracle position | 0 / 11 |
| Phone evidence, tracker | 1 / 11 |

No gap phoneme was a non-Quran phrase. On the oracle arm the inserted word is in the decoded string in every case (3–5 consonants) and those consonants are diagonal matches: on word 1, on the previous word, or on a third reference word. One target holds three consonants in the between-word insertion block, which the current insertion state already prices; the warning confirms after the mistake's match window. The single engine window-trim is those same frames after a later tracker re-lock. True position on that phone evidence does not trim them, and oracle evidence matches them to earlier words of the previous ayah.

## Why there is no checker edit

0 of 11 on the checker arm is under the kill threshold of 6. The inserted consonants are labeled as Quran text inside a window, which the card treats as outside this change. Extending the joint fit across the ayah boundary would not take those matches off the words that own them. No session, reader, or verse literal was added to engine code. Practice scores used: 0.
