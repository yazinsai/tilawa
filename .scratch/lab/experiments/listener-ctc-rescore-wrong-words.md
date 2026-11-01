# Word-level CTC veto for wrong-word false warnings

Status: measured, not recommended. Overall practice change is −2, which is noise. The kill criterion fired: planted prophet-name swaps fall inside the veto margin on perfect evidence, and one of them is newly missed on studio audio. Do not fold.

- Experiment: `listener-ctc-rescore-wrong-words`
- Test set: lab-v1.5 practice (100 sessions). Exam was not scored.
- Main line: `abb918b`, baseline `main-abb918b-20261101T034859Z` (89 phone wrong calls = 64 missed + 25 false warnings)
- Scored implementation: `e687777`, run `exp-listener-ctc-rescore-wrong-words-e687777-20261101T044238Z`
- One practice score. The margin was fixed from a true-position diagnostic before that score and was not moved afterwards.

## Prediction and result

The card predicted listener −4 and overall −4. Before scoring, the phone true-position scan of every confirmed wrong word supported a smaller claim: three of the six target false warnings sit at 11.1–12.2 nats, the nearest caught planted swap on phone listener evidence sits at 15.7 nats, so a 14-nat word margin would remove those three warnings and no planted wrong word. That would be listener −3 and overall −3. The score did not come out that way.

| Practice phone | Main | Experiment | Change | Prediction |
| --- | ---: | ---: | ---: | ---: |
| Overall wrong calls | 89 (64 + 25) | 87 (65 + 22) | −2 | −4 |
| Listener-attributed | 47 (28 + 19) | 43 (27 + 16) | −4 | −4 |
| Tracker-attributed | 5 | 5 | 0 | |
| Checker-attributed | 37 (33 + 4) | 39 (35 + 4) | +2 | |
| Checker on perfect evidence | 51 (46 + 5) | 58 (53 + 5) | +7 | preserve |

Studio practice went from 58 (45 missed + 13 false) to 60 (47 + 13). Listener phoneme errors stayed 9,039. Tracker false jumps stayed 37. No new false warnings on either variant.

## Mechanism

A wrong word already has to beat the written words by `SUPPORT` (10 nats, doubled below peak confidence 0.8) over the word and one neighbour either side. That window still confirms when one phone-confusable consonant wins argmax for a frame or two and the written phoneme holds the rest: the neighbour labels mostly cancel, and the peak confidence stays high, so the wider margin is cleared by about a nat.

`wordWins` runs the same CTC forward, `ctcForward`, on the heard word's own frames ±2, once for the written phonemes and once for the heard ones. The warning confirms only when the heard sequence wins by `WORD_MARGIN` (14 nats). No posteriors, no observed string, or an empty frame span leaves the wider window in charge. The SDK exports the same forward over a raw log-prob matrix; the lab score moves because the checker applies it to the stored top-5 rows, not because the acoustic model changed.

This is not prefix beam search (free decoding, no reference) and not the folded peak-confidence gate (that gate only doubles `SUPPORT`).

## What the margin was set on

True position, phone, every practice session, last confirmation of each wrong word. Word-window gap = written NLL − heard NLL over the heard phonemes ±2 frames.

False warnings under 16 nats:

| Word | Written → heard | Gap | Result |
| --- | --- | ---: | --- |
| lab-v1.5-p45 26:105:1 | كَذذَبَت → ََمَت | 11.06 | target, removed |
| lab-v1.5-p37 18:26:20 | يُشرِكُ → نَصِرِكُ | 12.01 | target, removed |
| lab-v1.5-p23 7:71:16 | نَززَلَ → لَمزَلَ | 12.18 | target, removed |
| lab-v1.4-p08 2:23:12 | مممِثلِهِۦۦ → مممِثلِهقَلِۦۦ | 13.04 | counted as a free false warning; it was the catch of an addition |
| lab-v1.5-p43 22:9:1 | ثَاانِيَ → ذَاالِيَ | 14.89 | kept |
| lab-v1.5-p32 22:60:11 | لَيَںںںصُرَننننَهُ → ََصَُننَهُ | 15.16 | target, kept |

The other three targets are not in this band. p11 26:106:1 (ءِذ → يَاا) is 24 nats: ذ is absent from the top 5, so the written score collapses. p70 35:40:33 (غُرورًا → رَاا) is 79 nats on the short heard span and 12 nats on the wider window; the ±2 pad never sees the missing onset. 56 confirmed planted wrong words were all at or above 15.70 nats on this phone evidence (closest: p42 26:161:5 لوط → هود). No other-class hit sat under 14. Engine alignment on the boundary words reproduced the same gaps. 14 was the gap between 12.18 and 15.70, with p32 left on the far side rather than scraped.

## What the score actually moved

Removed, all listener false warnings, matching the three separable targets:

- `lab-v1.5-p23/phone/warning:wrong_word/7:71:16/231.48`
- `lab-v1.5-p37/phone/warning:wrong_word/18:26:20/110.40`
- `lab-v1.5-p45/phone/warning:wrong_word/26:105:1/68.68`

p11, p32 and p70 stayed.

New phone miss: `lab-v1.4-p08/phone/mistake:added_word/2:23:13/122.38`. It had been a misclassified hit. The catch was the wrong-word warning on the previous word (2:23:12, gap 13.04). Oracle evidence still catches the addition as an addition; true position no longer does, so the miss is charged to the listener. The same miss appears on studio. Net phone wrong calls −3 + 1 = −2.

New studio miss of a planted wrong word: `lab-v1.5-p45/studio/mistake:wrong_word/26:106:5/73.95` (نوح said as لوط). Phone already missed it. Studio listener evidence over that word is about 13.1 nats, under 14, so both the studio engine and the perfect-evidence arm drop it. Charged to the checker.

Perfect evidence lost seven planted swaps, all prophet-name mutashabihat, and no false warnings. Phone engine still catches the ones it caught before; two that were already phone misses (p13 26:161:5 and p45 26:106:5) move from listener to checker because the oracle arm now misses them too. That is the checker +2, with no extra phone engine miss.

| Plant | Said | Oracle gap |
| --- | --- | ---: |
| lab-v1.4-p01 26:124:5 هود | لوط | 12.51 |
| lab-v1.4-p10 26:124:5 هود | نوح | 12.51 |
| lab-v1.5-p13 26:161:5 لوط | نوح | 12.51 |
| lab-v1.5-p24 26:106:5 نوح | هود | 12.36 |
| lab-v1.5-p42 26:161:5 لوط | هود | 12.59 |
| lab-v1.5-p45 26:106:5 نوح | لوط | 12.36 |
| lab-v1.5-p61 26:161:5 لوط | هود | 12.39 |

Oracle rows are a 0.9 spike on the heard phone and 0.05 on blank, residual spread over the rest. Forcing one alien consonant through that residual is only a few nats, and these pairs differ by two consonants around a shared madd, so the word lands at ~12.5 nats. Phone listener evidence of a caught swap is higher (15.7 and up) because the written consonant is missing from the top 5 and sits on the 1e-6 floor. The false alarms worth vetoing (11–12 nats) occupy the same band as a real swap once the evidence is clean. That is the kill criterion.

## Limits

The ±2 heard-span window cannot see a dropped onset (p70) or a written consonant that is genuinely absent (p11). A per-phoneme margin does not open a gap on the phone scan either: the weakest caught plant was about 2.1 nats per written phoneme and the strongest easy false alarm about 1.8. The oracle gaps above are the truth span of the said word, not the checker's heard span, so they explain the seven misses without being a number to cut a new threshold on. The phone false alarms that are actually free (11.06–12.18) and those swaps (~12.4) leave no headroom, and 13.04 on p08 is a real catch rather than a warning to spend. The constant was not moved after the score. Two practice scores remain unused.
