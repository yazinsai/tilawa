# Pausal-final manner, rekey an omission

Status: left for the fold gate. Not practice-scored here (two targets, predicted overall −2, below the gate on its own).

## Preregistered prediction

Test set lab-v1.5, practice, listener `exp-listener-heard-word-must-be-lexical-41570706`. Both phone wrong-word misses won: listener −2, overall −2. 2:284:28 at 8.9 nats is not in that count.

## Mechanism

Ayah-final words before a labelled pause only. The manner of the last consonant run in the slot, after the existing waqf-tail strip, is compared with the manner of the written word's isolated pausal final. The table is nasal (ن م ں ۾), liquid (ر ل), obstruent (stop or fricative, so ف/ك are one class) and glide (و ي). Same manner is never a contradiction.

Two paths, no new constant:

- A wrong word already raised on that slot keeps `SUPPORT` and the weak-evidence gate (twice `SUPPORT` when the changed phones average below 0.8). The manner mismatch replaces the lexical doubling and the 0.6 confidence cut.
- An omission already raised on that word becomes a wrong word at the same word when the phones from the previous word's last in-order visit up to the first labelled pause hold at least two non-blank tokens, one of them a consonant, and that final manner contradicts. The same two gates then decide it. A spliced-out omission has no speech before the pause and stays an omission. The rule does not open a new accusation.

The labelled pause is a pause token (ۜ ؙ) or a blank-owned gap of at least `minGapFrames`, the pause test the checker already uses. An earlier gap inside the previous word is outside the slot.

## What was checked before the score

True-position traces, phone, current checker:

| Target | Before | After | Heard | Manner |
| --- | --- | --- | --- | --- |
| `lab-v1.5-p78` 11:11:10 | uncertain wrong word, confidence 0.5997 | confirmed wrong word | مَںں | nasal against pausal ر |
| `lab-v1.5-p50` 37:84:5 | suspected omission, then uncertain | confirmed wrong word, observed مِل | مِل in the slot before the pause | liquid against pausal م |

11:11:10 already cleared the posterior margin (confirming it with `minConfidence` 0 did, before this change). The 0.6 cut was the only block. 37:84:5's alignment visits مِل on سليم but the visit does not count, so the word was an omission whose gap also covered إذ. The slot before the pause is only مِل. The rekey narrows the accusation to 37:84:5.

Unit tests cover the 0.6 bypass, a same-manner final (ل against ر, and ف against ك), a short gap that is not a pause, a non-final word, flat posteriors that do not clear `SUPPORT`, a contradicting omission, a same-manner omission, a spliced-out omission, a clean ayah end, and a ghunna ں standing in for ن.

## Limits

Stop and fricative are one obstruent class. A real stop-versus-fricative swap at an ayah end does not fire, and neither does ف/ك. 2:284:28 is ه against ر, which is a manner mismatch, and it stays under the 10-nat floor. The weak-evidence gate still asks for 20 nats when the changed phones are unsure, so a manner mismatch at one `SUPPORT` and low confidence does not confirm. Rekeying drops any later word that rode in the same omission; on 37:84:5 that later word is إذ, which was recited.
