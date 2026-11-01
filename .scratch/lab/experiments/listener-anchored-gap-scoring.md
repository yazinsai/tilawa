# Anchored gap scoring for omission warnings

Status: killed at step 0. No engine change. Practice was not scored.

## Prediction

Listener −4, overall −4. The p79 pair (omission of 20:25:4 reported on 20:25:5) is 2. The other 2 were speculative vetoes among the three unpaired `omitted_word` false warnings: p50 37:113:10, p65 4:12:40, p65 4:15:6. Margin reused from the killed placement card: 1 nat. No new constant was fit.

## Mechanism that was going to be tested

An omission is cut by the greedy visit. A phone-blurred word can leave its consonants in the gap and have its tail consumed by the neighbouring written word. p79 is that cut: 20:25:4 لِى (`لِۦۦ`) is the planted omission, and the decode of 20:25:5 صَدْرِى (`صَدڇرِۦۦ`) is `د ڇ ل ِ ۦ`. The visit on لِى takes `ل@639 ِ ۦ@643`; `د@633 ڇ@634` sit between ٱشْرَحْ and that visit; the checker then reports صَدْرِى omitted.

The pre-registered measurement holds. `PosteriorFrames.nll` on frames [626, 648), the stretch from ح of ٱشْرَحْ (frame 626) through the end of the صَدْرِى audio:

| hypothesis | nll (nats) |
| --- | ---: |
| صَدْرِى | 10.825 |
| لِى | 12.402 |
| لِى + صَدْرِى | 62.10 |

صَدْرِى beats لِى by 1.577 nat. The same difference is 1.577 on [626, 651), out to و of وَيَسِّرْ. On the tail segment ±2 frames, [637, 646), لِى wins by 47.993 nat. That is the killed card's ±2 window. Full text is 51 nat worse than keeping صَدْرِى, so a veto would not have fired here; a placement move to لِى would have.

## Step 0 windows

Same rule for each unpaired false warning: frames run from the last phoneme of the previous counting visit to the next visit's first reference match. The omission hypothesis is blank (nothing between those anchors). The full-text hypothesis is the written word. Positive margin means full text beats the omission.

**p50 37:113:10** `مُبِۦۦن`. Previous visit 37:113:9 ends at `ۦ@6876`. Next visit 37:114:1 (`وَلَقَد`) starts at `ء@6891`, which is not و. Window [6876, 6891): full 35.95, blank 5.80, full loses by 30.15. Extending the end through frame 6998 never flips it; the least-bad margin is −1.82. Scoring `للِنَفسِهِۦۦ` + `مُبِۦۦن` against `للِنَفسِهِۦۦ` from frame 6845 is still at best −2.03. The gap does hold weak م/ب mass (frame 6886 ب 0.27, م 0.13) and ن only after the next word has started. It is not a word-scale win for the written word.

**p65 4:12:40** `ثثُمُنُ` (pause form `ءَثثُمُن`). Previous visit 4:12:39 ends at `َ@2746`. The next visit starts at `ك@2749` (conf 0.33), which is not م of مِمَّا; the first reference match is `م@2779`. Window [2746, 2779): connected 30.53, blank 15.10, full loses by 15.43. The pause form loses by 22. Scanning the end out to frame 2858, the connected form's best margin is +0.23, under 1 nat, and that end is past the onset. `فَلَهُنَّ` + `ٱلثُّمُنُ` + `مِمَّا` against the two neighbours stays worse by at least 17 nat. Frame 2749 is ك/ت/س, not ث.

**p65 4:15:6** `فَستَشهِدُۥۥ`. Previous visit 4:15:5 ends at `م@5218`. Next visit 4:15:7 starts at `ع@5251`. The gap is `ِ@5240 د@5243 ُ ۥ ۥ`, the tail, shorter than half the word so the checker keeps an omission. Window [5218, 5251): full 25.71, blank 19.18, tail `دُۥۥ` 9.08. Full loses to blank by 6.53 and to the tail by 16.6. ف is 0.08 and ش peaks at 0.32; forcing the rest of the word through those frames never catches up (best scan to frame 5319 is still −6.49).

## Kill

The anchored full-text score fails the 1 nat test on all three unpaired false warnings. The gate requires a failure on two. Stopped before any checker change. Caught planted omissions were not swept: this clause already kills, and a veto that cannot fire on these three warnings is not the predicted −4.

p79's 1.577 nat placement margin is real and is specific to scoring the bare words on the stretch that still contains `د`. Putting the previous anchor phoneme ح into the label string reverses it (ح + لِى beats ح + صَدْرِى by 6.4 nat), because ح takes the only frame that has ص mass. The veto, which is what step 0 tests, does not clear the false warnings either way.
