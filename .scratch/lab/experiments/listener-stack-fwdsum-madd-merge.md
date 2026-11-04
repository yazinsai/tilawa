# Stack the forward-sum omission, the connected madd, and the seam-skip merge

Status: practice-scored. Left for the fold gate. Engine tree is the three riders, byte-identical, on main `18d59c95c249a8d0d7556612674ebbde2a9c0418` (the channel-odds fold is in that main).

## Prediction

Test set lab-v1.5, practice, listener `exp-listener-heard-word-must-be-lexical-41570706`. Three phone incidents, three readers:

| Key | Rider | Reader |
| --- | --- | --- |
| `lab-v1.5-p14/phone/mistake:omitted_word/44:23:1/198.47` | forward-sum confirmation | mp3quran-40 |
| `lab-v1.5-p79/phone/mistake:wrong_word/20:43:1/160.19` | connected madd | mp3quran-10905 |
| false skip at 26:131:2 (p24) | ledger/seam merge | mp3quran-70 |

Listener −1, checker −2, overall −3, 0 lost.

## Mechanism

Three decision points in `checker.ts`. None reads or writes the others, and none goes through `judge`'s shared wrong-word return.

An `omitted_word` that would end uncertain only because greedy clarity is under `minConfidence` is confirmed when the ayah-absence ledger's closed read-versus-omit forward sum prefers omitting by `PHRASE_SWITCH` (4 nats). The stretch is the previous word's last confident phoneme through the onset of the word after the next. Support rejections, unlocked stretches, and forced uncertain stay uncertain. Confirmed omissions are not reopened.

A word-final long vowel before a hamza word, with no labelled pause and an attested shorter mushaf phrase, is a wrong word only when that span is strictly shorter than every earlier matched madd munfasil in the session. The bar is that minimum. No new constant.

After both skip emitters, a seam skip that starts after a ledger ayah is dropped when every word between them is empty or a counting visit of refrain or shared near-duplicate text. The ledger key stays. A unique word between them keeps both.

`checker-channel-odds-omission-place` is on this main. It still places 44:5 and 20:25. The forward sum does not move those; they were already confirmed.

## What was checked

`runEngine` via the lab scorer, phone, the card's listener, on the rebased tree. All three keys:

| Case | Outcome |
| --- | --- |
| p14 44:23:1 | hit, omitted word, confirmed 214.96 s. Session 0 missed, 0 false warnings |
| p79 20:43:1 | hit, wrong word, confirmed 164.08 s |
| p24 26:131:2 | absent. 26:130:1 still a hit at 256.76 s. Session 0 missed, 0 false warnings |
| p14 44:5:4 | still a hit (channel-odds placement) |
| p79 20:25:4 | still a hit (channel-odds placement) |

Readers mp3quran-40, mp3quran-10905, mp3quran-70. Leaving any one out leaves −2.

Practice score of `529160d4`, which is this branch on main `18d59c95`. The rebase only brings `modal/lab_web/progress.js` and the autolab brain; checker, tracker, and listener code are the three riders:

| | main line | this | change |
| --- | ---: | ---: | ---: |
| overall | 36 (24 missed + 12 false) | 33 (22 missed + 11 false) | −3 |
| listener | 32 (22 missed + 10 false) | 31 (21 missed + 10 false) | −1 |
| tracker | 1 (0 missed + 1 false) | 1 (0 missed + 1 false) | 0 |
| checker | 3 (2 missed + 1 false) | 1 (1 missed + 0 false) | −2 |

Misses fell by 2 and false warnings by 1, which is the three keys and nothing else in the totals. Core and sdk tests passed on six shards. One shard hit Vitest's RPC deadline after every test passed.

## Limits

The practice compare does not print studio, clean-session warnings, or the incident keys. The totals move by exactly the three predicted calls (two misses, one false warning), and the three sessions that hold the keys do not grow a warning. p79 still has one other miss; it is not one of these keys. The forward sum's right edge is the onset of the word after the next, so a session that ends on the next word stays uncertain.
