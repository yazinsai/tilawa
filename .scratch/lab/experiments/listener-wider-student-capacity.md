# Wider student capacity

Status: killed. The engine diff is reverted. No model is deployed.

## Preregistered prediction

Test set lab-v1.5, practice, listener `spliced-clean-int8`. Listener −3 wrong calls, overall −2. Three of the eight phone wrong-word false warnings cleared, one loss.

## Mechanism

The eight warnings are stable consonant confusions of the shipped student (the fp32 parent confirms them, and no re-read or veto separated them from planted swaps). The bet was that a student at Zipformer-M width spends its capacity on dominant studio cues and keeps only coarse place features for band-limited consonants, so ن/م and ف/ك collapse on phone audio. A student at twice that width, same depth and 40 ms frames, distilled with the shipped CTC+KD recipe, augmentation and int8 export, and pulled toward the shipped student by a KL on clean-match frames, would keep the studio string and resolve some of the phone spikes.

## Early check

The teacher evidence, scored at the true position, does not confirm any of the eight target warnings. On the accused words the teacher has the written consonants where the student does not (إذ, يشرك, ثاني, غرورا exact or checker-clean; نزل keeps ن; ذواتا keeps ت). Window-edge phones made a skeleton diff look like six teacher misses; the checker does not treat those as the target warning. That is a capacity gap, not a data limit, so the distillation ran.

## What was trained

`wide` is Zipformer-M with encoder, unmasked and feed-forward width doubled (247.7M parameters, depth and 40 ms frames unchanged). Twelve epochs on clean-match, spliced CTC target, KD weight 2, the shipped augmentation, scratch learning rate 0.045 because the LibriSpeech init does not fit the wider tensors (0.015 was only that init arm), plus KL weight 1 to the frozen shipped full student on every clean-match frame. Final greedy PER on the clean-match test split was 9.92% versus reference and 6.18% versus the teacher, against 9.38% and 5.60% for the shipped run. The anchor term was about 0.1 nats at the end. Int8 export is 252 MB; fp32 ONNX agreed with PyTorch on 99.0% of frames of one clip and 0 phone edits.

## True-position replay

Int8 decode of the eight target phone sessions and every fully clean phone and studio session (56 wavs), checker at the true position, compared with the shipped int8 evidence:

| session | word | wide | shipped |
|---|---|---|---|
| p11 | 26:106:1 | clear | confirm |
| p23 | 7:71:16 | clear | confirm |
| p32 | 22:60:11 | confirm | confirm |
| p36 | 55:48:1 | confirm | confirm |
| p37 | 18:26:20 | clear | confirm |
| p43 | 22:9:1 | clear | confirm |
| p55 | 26:174:8 | clear | confirm |
| p70 | 35:40:33 | confirm | confirm |

Five of eight targets cleared. The kill line was six or more still confirmed. That line did not fire.

The same replay added warnings the shipped student does not raise, on sessions with no planted mistake:

- `lab-v1.5-p32/phone` wrong_word 22:72:20 at 235.08 s
- `lab-v1.5-p36/phone` wrong_word 55:48:2 at 287.40 s
- `lab-v1.5-p37/phone` wrong_word 18:29:20 at 184.88 s
- `lab-v1.5-p43/phone` wrong_word 22:13:3 at 213.84 s
- `lab-v1.5-p43/phone` wrong_word 22:15:3 at 237.68 s

The kill criterion says any warning added on a fully clean phone or studio session. That fired. The clean-match anchor was not enough: PER versus the shipped string moved by about half a point, and the phone decode picked up new wrong words while dropping five of the eight targets.

## Limits

No practice score. Studio clean sessions were decoded and did not show up in the new-warning list; the five additions are phone. The wider weights are still on the scratch volume (`student-runs/wide-clean`, `wide-clean-b-int8`) and are not what the branch decodes with.
