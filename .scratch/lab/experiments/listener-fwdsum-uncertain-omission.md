# Uncertain omission by read-versus-omit forward sum

Status: implemented. Left for the supervisor's score. Not practice-scored here: the preregistered change is −1, below the fold gate on its own.

## Prediction

Test set lab-v1.5, practice, listener `exp-listener-heard-word-must-be-lexical-41570706`. The planted omission `lab-v1.5-p14/phone/mistake:omitted_word/44:23:1/198.47` is confirmed. Listener −1, overall −1.

## Mechanism

An omission that is already proposed is confirmed or left uncertain by the mean confidence of the greedy tokens beside its gap. On p14 that mean is 0.58 because it includes 44:23:2's own blurred visit, not residue of فأسر.

The change runs only for an `omitted_word` that today would end uncertain because of that clarity gate: the tracker is locked (or the passage is declared), posterior support has not rejected it, and nothing forced the state to uncertain. Confirmed omissions, support rejections, and every other error class are untouched.

Over the frames from the previous word's last phoneme at or above `minConfidence` to the onset of the word after the next one, it runs the ayah-absence ledger's forward sum. `H_read` is the previous word, the flagged word, and the next word. `H_omit` is that path without the flagged word. Both use the ledger's contiguous repeat arcs and no filler. The omission is confirmed only when `H_omit` leads by `PHRASE_SWITCH` (4 nats). Otherwise it stays uncertain. The stretch is remembered if a later commit scrolls an edge out of the window, and those posterior frames are kept until the omission is decided, the same way the ledger keeps frames for an open skip. No new threshold. The omission is never created or moved.

A blurred neighbour sits in both hypotheses. A word that was actually absent has to be fitted into frames that do not contain it.

## What was checked

`runEngine` on lab-v1.5-p14 phone, the card's listener:

| | value |
| --- | --- |
| clarity | 0.582 |
| support | passes (not the blocker) |
| frames | 4961–5003 (42 frames, still buffered) |
| `nll(H_read) − nll(H_omit)` | 71.0 nats |
| decision | confirmed `omitted_word` 44:23:1 at 214.96 s |

71 nats clears 4. It is the only omission in the session whose clarity is under the gate; every other omitted word still confirms or stays on the old path. The kill criterion (the margin missing `PHRASE_SWITCH`) does not fire.

Unit tests: a low-clarity omission whose frames never carry the word confirms, and the confirmed event's confidence stays under 0.6, so it did not sneak through the old gate. The same omission with the word's phones painted into the blank gap stays suspected, then uncertain. Core and sdk tests passed on six shards (one shard hit Vitest's RPC deadline after every test passed).

## Limits

Practice was not scored. The fold gate wants an overall gain of 3; this card is the rider below that, and the supervisor scores it. The right edge is the onset of the word after the next, so a session that ends on the next word cannot run the sum and stays uncertain. A support rejection is still a rejection.
