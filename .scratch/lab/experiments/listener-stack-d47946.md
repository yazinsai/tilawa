# Stack manner rekey and the omission forward sum

Status: practice-scored. Left for the fold gate. The checker tree is the one that scored; this cut's parent is the channel-odds fold, so the experiment diff does not carry the ledger.

## Prediction

Test set lab-v1.5, practice, listener `exp-listener-heard-word-must-be-lexical-41570706`. All three phone misses confirmed: listener −3, overall −3, no new false warning.

## Mechanism

Two rules that each won a different incident and lost none.

An ayah-final word before a labelled pause confirms a wrong word the checker already raised when the last consonant run's manner contradicts the written pausal final. Nasal, liquid, obstruent and glide; same manner is never a contradiction. `SUPPORT` and the weak-evidence gate stay. An omission already raised on that word becomes a wrong word when the slot holds at least two non-blank tokens, one a consonant, and that manner contradicts. A spliced-out omission has no speech and stays an omission.

An omission that would end uncertain only because greedy clarity is under 0.6 is confirmed when the ayah-absence ledger's closed read-versus-omit forward sum, over the previous word's last confident phoneme through the onset of the word after the next, prefers omitting by `PHRASE_SWITCH` (4 nats). Support rejections, unlocked stretches and forced uncertain stay as they are.

`decide` uses both. A manner mismatch is sure without the 0.6 cut. The forward sum runs only while the issue is still an omission and clarity is the gate that failed. The rekey runs in `judge`, before `decide`, so a speech-holding ayah-final omission is a wrong word by the time the sum is considered.

This branch was cut before `checker-channel-odds-omission-place`. That fold is merged in unchanged: a one-word omission whose stretch the listener channel explains as the other adjacent word still moves, and the step splice still lands after the manner loop.

## What was checked

Full engine, phone, the card's listener. All three targets are hits, and the channel-odds omission on the same session stays a hit:

| Case | Outcome |
| --- | --- |
| `lab-v1.5-p14/phone/mistake:omitted_word/44:23:1/198.47` | hit, omitted word |
| `lab-v1.5-p50/phone/mistake:wrong_word/37:84:5/50.27` | hit, wrong word |
| `lab-v1.5-p78/phone/mistake:wrong_word/11:11:10/143.13` | hit, wrong word |
| `lab-v1.5-p14/phone/mistake:omitted_word/44:5:4/36.26` | hit, omitted word |

Readers `mp3quran-40`, `mp3quran-198`, `mp3quran-10905`. Leaving any one out leaves two of the three.

The first practice score ran before that merge. Main was already 36 (24 missed + 12 false) after the channel-odds fold; this tree, still on the previous line, scored 37 (23 missed + 14 false). Against that previous line (40 = 26 missed + 14 false) the same run is −3 misses and 0 new false warnings.

Second practice score, `4ac566bf`, with the fold merged:

| | main line | this | change |
| --- | ---: | ---: | ---: |
| overall | 36 (24 missed + 12 false) | 33 (21 missed + 12 false) | −3 |
| listener | 32 (22 missed + 10 false) | 29 (19 missed + 10 false) | −3 |
| tracker | 1 (0 missed + 1 false) | 1 (0 missed + 1 false) | 0 |
| checker | 3 (2 missed + 1 false) | 3 (2 missed + 1 false) | 0 |

Misses fell by 3 and false warnings did not move. Core and sdk tests passed, including the manner, forward-sum and channel-placement tests. One shard hit Vitest's RPC deadline after every test passed.

## Limits

The practice compare does not print studio, clean-session warnings or the incident keys. Phone false warnings are unchanged and the three wins are three readers. 2:284:28 stays under the 10-nat floor. The right edge of the forward sum is the onset of the word after the next, so a session that ends on the next word stays uncertain.

The first cut merged that fold in after the card commit, so `git diff` from the card's parent also contained `docs/benchmarks/lab-ledger.json`. The checker, both tests and the measured phone result are unchanged; only the history base moved.
