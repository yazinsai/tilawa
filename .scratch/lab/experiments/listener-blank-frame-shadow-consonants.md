# Blank-frame shadow consonants on proposed omissions

Status: implemented. Practice headline is −3 overall and −3 listener. The supervisor's fold still checks studio, clean sessions, and the reader holdout.

## Preregistered prediction

Test set lab-v1.5, practice, listener `exp-listener-heard-word-must-be-lexical-41570706`. Listener −2 wrong calls, overall −2, upside −3. The gain is the p14 44:5 placement (false warning on كنا plus the paired miss of إنا) and, if the same read holds, the p50 veto of مبين.

## Mechanism

An erased consonant on this listener does not become a token or a token's runner-up. It stays the top non-blank class on frames where blank wins, next to the tokens. Forward sums still price those frames as blank.

For an omission the checker has already proposed, take the stretch from the previous matched word's last confident phoneme to the next matched word's confident onset (the checker's existing confidence bar). On each frame in that stretch that no token owns and blank ranks first, record the top non-blank class. Rank only, no mass cut, so no new constant.

Two adjacent words that differ by a single onset consonant are one heard body. If that onset is the last shadow consonant before the body, the omission moves off that word onto the other one. A lone omission is dropped when those shadow consonants spell a contiguous slice of the flagged word of at least two consonants and two distinct classes, the same shape an added word already has to have, and none of those consonants is written on the word to either side. A neighbour's own letter is not evidence this word was said. The rule never creates an omission. The decoded string and the tracker are untouched.

## What was validated

On the stored phone posteriors, before the rule:

- p14, 36.84–36.88 s: blank wins, top non-blank is ك at 0.219, and no token owns the frames. The next token is ء at 36.92 s.
- p50, inside مبين: ب at 0.274 then ن at 0.403 on blank-won frames, with م just before ب. That is the prefix مب of مبن.
- p79, صدري: no ص on a token-free blank-won frame. The rule does not claim this pair.

Replay of the three sessions through the true-position arm and the free engine:

- p14 confirms 44:5:4 and not 44:5:5, on both arms. The planted omission is caught. The other planted omissions in the session stay caught, except 44:23:1, which this replay still misses.
- p50 emits no omission on 37:113. Every planted omission in the session stays caught.
- p79 still confirms 20:25:5 and still misses 20:25:4.

Unit tests cover the move, the same audio without the shadow consonant, the two-consonant veto, a reversed pair that must not veto, and a pair that borrows the previous word's consonant.

The first practice score, before the neighbour guard, was overall −2 (27 missed + 11 false warnings, from 26 + 14). Listener −2. Two planted omissions were vetoed because a neighbour's consonant completed a pair: وكلَّ at 36:12:9 (`موك`, the و already written on وآثارهم) and وذلكم at 41:23:1 (`كم`, written on the next word). The guard drops a slice that uses either neighbour's consonant.

After that guard, practice is 37 wrong calls (25 missed + 12 false warnings), from 40 (26 + 14). Listener 33, from 36. Overall −3, listener −3.

## Limits

A move can still swap a correctly placed omission if the missing word's onset is the runner-up in front of the surviving neighbour. p79 is unchanged: its onset is not a shadow name. A gap can still spell two consonants that belong to this word and to neither neighbour. Clean sessions are unchanged only while they propose no omission; the rule does not invent one.
