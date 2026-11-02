# Surplus speech that spells a mushaf word

Status: practice-scored once. Left for the fold gate.

## Preregistered prediction

Test set lab-v1.5, practice, listener `exp-listener-context-free-gap-decode-62ac8b5`. 3 of 16 checker added-word misses won, checker −3, overall −2 (floor −1).

## Mechanism

An added word at an ayah start is often already a whole word in the greedy phones, and it is lost when that string is re-priced as a likelihood against the neighbours. The rule confirms an added word only when both of these hold, with no new constant:

- Surplus: word 1 and the previous ayah's last word each have a complete in-order visit (`lexical.matches` on a full visit), so the phones between those visits are not needed to read the text.
- Identity: those phones, with phrase-labelled spans removed and edge waqf marks stripped, equal one mushaf word of at least two consonants under soft costs only (same length after held-sound collapse, vowels may differ, no consonant edit). The span must not be a word, a prefix, or a contiguous run inside ±3 content words, and must not be a non-Quran phrase.

That match sets `issue.lexicon` and replaces the 10-nat support margin. Seams with no such span are unchanged.

## What was checked before the score

The previous card's step 0, on this same rule shape, found 10 of 16 checker added-word misses whose surplus span spells a non-neighbour word, and one phone seam with no planted addition (`lab-v1.4-p15` 38:7:1, تبع). That seam is on a session that already has planted mistakes, so it is at most one lost incident. This rebuild ships the rule and lets the gate weigh it.

Unit coverage, local vitest, 648 core tests passing:

- قبلهم inserted before 44:10:1 confirms under neutral posteriors, where the 10-nat margin would abstain.
- A neighbour repeat, a word-1 prefix, amin, a bare waqf mark, and a held meem on عظيم stay quiet.
- Speech that spells no word still does not confirm under neutral posteriors.
- A paused قبل outside ±3 of 51:18:1 is an added word. The old test expected silence because that repeat sits further back than the neighbourhood.

## Measured practice result

One practice score, worktree `3d79aab8-dirty` (the shipped diff). No second score.

| | main line | this | change |
| --- | ---: | ---: | ---: |
| overall | 66 (45 missed + 21 false) | 60 (39 missed + 21 false) | −6 |
| checker | 24 (22 missed + 2 false) | 13 (11 missed + 2 false) | −11 |
| listener | 38 (21 missed + 17 false) | 41 (24 missed + 17 false) | +3 |
| tracker | 4 (2 missed + 2 false) | 6 (4 missed + 2 false) | +2 |

False warnings are unchanged in total and per component. Checker misses fell by 11. Listener misses rose by 3 and tracker misses by 2. The compare verdict is better than the preregistered overall −2 and checker −3. Studio, clean-session, and leave-one-reader cuts are not in this compare; the fold gate has them.

## Limits

The match is exact after held-sound collapse, so a span that only shares a consonant skeleton with a longer tanween form does not fire. Step 0's `2:11:1` قليلا, which strips to `قَلِۦۦل`, is that case. Phone noise that spells a real multi-consonant word in a surplus span still fires; the scored run did not add a false warning on net. The listener and tracker miss increases are not narrowed here: the gate trades them against the checker wins.
