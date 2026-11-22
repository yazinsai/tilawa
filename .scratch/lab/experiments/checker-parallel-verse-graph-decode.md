# Parallel phrase graph on the posteriors

Status: implemented, not practice-scored here. The supervisor scores the branch.

- Experiment: `checker-parallel-verse-graph-decode`
- Component: checker
- Prediction: checker −3, overall −3

## Prediction

Mutashabihat swaps that differ by a suffix or one interior word are missed because the string aligner prices them as an end-edit. Held-sound and waqf tolerance lend a neighbour's phones to the suffix, so the word never becomes a candidate. The bet replaces that comparison: for each anchored stretch, score the written phrase and every mushaf-parallel phrase by the CTC forward sum on the shipped posteriors. Suffix frames belong to the whole phrase and cannot be counted twice. No insertion or restart arcs.

## Mechanism

A phrase is four consecutive mushaf words inside one surah. It is parallel to another when the two share a letter skeleton on three positions and the interior word differs by a consonant. The index is built from the mushaf text, not from session or verse ids. Longer single-difference phrases contain one of these four-word windows.

On an anchored alignment that has matched the shared entry word and the shared exit word, the checker builds two phoneme strings over that span: the written words, and the same words with the interior word replaced by the parallel occurrence's phonemes. Each string is the CTC negative log-likelihood of the frames from the entry word through the exit word. The parallel path wins when it leads by `PHRASE_SWITCH` (4 nats) and the frames that are not blank-owned actually peak on a phone (`PHRASE_PEAK`). The interior word is then a wrong word. The exit match is the onset, because the next word is often the suffix the two phrases disagree about.

## What was measured

Shipped phone posteriors, truth word timings, nine targets. Every `said` string is the parallel word. Margins (`written − alt`, nats):

| Case | Margin |
| --- | ---: |
| p08 37:108:2 عليه→عليهما | 12.7 |
| p50 37:108:2 | 15.3 |
| p08 37:119:2 عليهما→عليه | 49.8 |
| p64 37:119:2 | 45.1 |
| p64 37:129:2 | 22.7 |
| p64 37:122:1 إنهما→إنه | 37.9 |
| p64 37:132:1 إنه→إنهما | 16.0 |
| p54 26:141:2 ثمود→عاد | 50.8 |
| p59 7:66:1 قال→وقال | 6.8 |

All nine prefer the parallel path by more than 4 nats.

Correct speech does not. Surah 37, unspliced, four readers excluded from the lab held-out set (reads 1, 4, 5, 12), shipped listener: 175 parallel phrases whose greedy decode was closer to the written phrase, worst margin −8.0, 95th percentile −15.4, none positive. Fully clean practice phone sessions: 760 spans, worst −0.7, none positive. Correct spans in the other practice sessions, away from a planted mistake: 1659 spans, three positive, worst +2.9.

Default-policy traces confirm the parallel word: p08 37:108:2 (عليه→عليهما), p64 37:119:2 (عليهما→عليه), p59 7:66:1 (قال→وقال), p54 26:141:2 (ثمود→عاد). The suffix has to count the shared exit word as the onset; exact hits on the next word fail when the extra ما lands on it. That spill still opens a string-alignment warning on p08 37:108:3, and with posteriors on it ends uncertain.

## Limits

The penalty is one number for every parallel, set above the phone false tail (+2.9) and below the weakest target (+6.8). Studio recitations of the same phrases sat 8 nats the other way, so the phone channel is what the 4 nats is for. Flat posteriors abstain; a window of confident blanks does not count as hearing the phrase. Vowel-only twins are not paths. This run did not practice-score, so new confirmed warnings away from the nine targets are not counted yet. Unit tests cover the index, both suffix directions, ثمود→عاد, a correct phrase, and flat or missing posteriors.

The phrase graph now sits beside the folded dropped-suffix joint fit. A phrase candidate does not replace an issue that fit already raised on the same word. The margins above are CTC sums on shipped posteriors; they do not depend on that fit. A span the string edit abstains on — a stray consonant the pair will not call, or a swap that is not a suffix — is still scored as one phrase.
