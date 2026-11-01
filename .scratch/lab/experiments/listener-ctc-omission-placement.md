# CTC forward score for omission placement

Status: killed by the pre-registered margin. No engine change. Practice was not scored.

Test set: lab-v1.5 practice, phone, listener `spliced-clean-int8`. Traces are the checker at the true position (`lab-trace-cli`). Forward scores are `PosteriorFrames.nll` on the stored top-5 posteriors. `listener-ctc-rescore-wrong-words` has not landed here; that call is this same forward.

## Prediction

Listener −4, overall −4. The two planted omissions are heard, but the warning is opened on the next word:

- `lab-v1.5-p14/phone/warning:omitted_word/44:5:5/38.92` — truth is 44:5 word 4 إِنَّا left out; كُنَّا was said.
- `lab-v1.5-p79/phone/warning:omitted_word/20:25:5/26.40` — truth is 20:25 word 4 لِى left out; صَدْرِى was said.

Each is one miss plus one false warning, both charged to the listener (oracle phonemes place the gap on word 4; listener phonemes place it on word 5). The proposed rule: when one heard segment has two adjacent written owners, score each word with the CTC forward on the segment frames ±2 and put the omission on the loser, unless the margin is under 1 nat.

## What the checker actually did

p14. كُنَّا (36.88–38.35 s) is decoded `ءُننننَاا`, frames 923–958. That visit covers إِنَّا (`ءِننننَاا`) at cost 3, one short-vowel substitution, and counts as said. كُنَّا has no visit. A consonant substitution of ك for ء would have cost 10, so the weighted edit prefers إِنَّا. The /k/ mass the greedy argmax dropped is real and sits in the ±2 pad: frame 922 is blank 0.636, ك 0.219, ء 0.060. Frame 923, the greedy hamza, is ء 0.553, blank 0.201, ق 0.114, ُ 0.037, ك 0.027. Frame 924 is ُ 0.531, ء 0.190, blank 0.139, ِ 0.108.

p79. صَدْرِى (25.07–25.89 s) is decoded `د ڇ ل ِ ۦ`. The visit that counts is only `لِۦ`, frames 639–643, and it matches لِى (`لِۦۦ`) at cost 0. `د` at frame 633 and `ڇ` at frame 634 sit between ٱشْرَحْ and that visit. Two phonemes are below the added-word minimum, so they stay unattached. صَدْرِى is the empty gap.

## Forward scores

Negative log-likelihood in nats, raw reference labels, half-open frame range. Lower is better. The true surviving word has to win by at least 1 nat.

| Case | Window | Omitted word | Surviving word | Margin |
| --- | --- | --- | ---: | --- |
| p14 visit | [923, 959) | إِنَّا 4.165 | كُنَّا 6.247 | إِنَّا by 2.082 |
| p14 visit ±2 | [921, 961) | إِنَّا 4.525 | كُنَّا 5.174 | إِنَّا by 0.649 |
| p79 visit ±2 | [637, 646) | لِى 5.799 | صَدْرِى 53.791 | لِى by 47.993 |
| p79 visit plus the two loose phonemes ±2 | [631, 646) | لِى 13.227 | صَدْرِى 27.607 | لِى by 14.380 |

On the specified window the true word loses both times. Padding the p14 segment by up to 7 frames before and 5 after never flips it; the best كُنَّا can do is still 0.65 nat behind. The hamza spike at 0.553 outweighs the /k/ frame at 0.219, and the kasra mass is close enough to the damma that the path sum does not recover كُنَّا. Collapsing the shared noon and madd runs leaves that 0.65 nat gap unchanged.

A window that is not the segment, [626, 648), does let صَدْرِى beat لِى by 1.577. It starts on the previous word's ح and runs through the trailing madd. That is outside the rule, and no analogous stretch saves p14. The 1 nat fallback would therefore move neither target. Nothing was changed in the checker, and the practice set was not scored.

## Limits

These two sessions only. Other omission pairs were not searched: a rule that cannot move these two could still move something else, and that was not measured. Studio audio was not scored. The posteriors are the top-5 rows the checker already stores, with the same residual spread as `PosteriorFrames.row`.
