# Wrong-word rival locality

Status: at least three targets validated; practice score and full tests requested.

## Preregistered prediction

Experiment `listener-swap-rival-surah-locality`, listener, explore. Practice set `lab-v1.5u`, listener
`exp-listener-teacher-insert-marginal-gate-ad6eb8c0`. Predicted 2 of 4 target phone false warnings removed,
about one caught swap lost: listener −2, overall −1. The card is unchanged.

The task requires stopping if the built veto removes fewer than three registered targets, even though the card
predicts only two. The card's separate premise kill is fewer than two targets with rivals absent from both links.

## Mechanism built

At final confirmation, a non-graph wrong word retains its existing acoustic and lexical tests and additionally
requires its attested rival to have a memorisation link. A rival is linked if its mushaf word form occurs in the
current surah or in the existing folded parallel-phrase entry for the accused slot. Letter identity uses the
existing diacritic/alef folding; supported phone forms map back to all their attested word identities.

For a lexical candidate the named pronunciation stays the rival. For an ordinary string-path spelling the
existing whole-word lexical test supplies the rival. If the span is non-lexical, all whole attested forms with at
least two consonants compete under `spanLogP` and the unchanged `LISTENER_CHANNEL`, after held-sound collapse.
The written word, nearby words, prefixes and contiguous neighbourhood runs are excluded using the existing
lexical neighbourhood test. Locality is checked after ranking, so a local word cannot displace a better distant
rival. Missing word text, an empty channel for a non-lexical span, or an unidentified rival abstains.

Parallel-graph wins are exempt. No constants, model weights, decoded tokens, tracker events, or word boundaries
change. The rule changes only a pending confirmation to uncertain.

## Validation

Targeted engine and true-position replays compare the built checker against the branch's original checker on
the same stored phone evidence. Diagnostics record the original confirming path, heard span, chosen attested
rival, and both locality links. These are target diagnostics, not a practice score.

Modal run 4 passed all 11 focused unit and checker tests (1.09 s). The first test run passed 10 and failed one
fixture that accidentally placed the rival inside the existing ±3 exclusion; the corrected fixture places it
outside that neighbourhood and asserts its identity as well as its locality. No engine rule changed for that fix.

Unit tests cover same-surah rivals, cross-surah parallel completions, distant rivals, slot-specific parallel
links, non-lexical channel ranking, preservation of a named lexical rival, missing-data abstention, clean text,
and checker confirmation of local and parallel substitutions.

Modal run 6 used the exact card-listener evidence for the first three completed targets. Each warning
was present in the original checker and absent with the veto, in both the free-tracker and true-position arms.
Their best rivals were `يُرَا` (يرى), `يَخَافَا` (يخافا), and `بِطَرِۦقَتِكُمُ` (بطريقتكم), respectively. All
three rivals were absent from the current surah and the slot parallel entry. The fourth shard is pending.

Modal runs 3 and 5 did not execute the replay: the snapshot omits `.scratch` files and the
experiment card. Run 6 uses temporary core-test helpers and passes the card listener name explicitly. If that
evidence is absent, the helper records its fallback to stored `spliced-clean-int8` evidence; that is a diagnostic
comparison, not a fresh decode of the card bundle. One practice score is being requested because the built change has already won three targets. Results pending.

## Limits

This is a text-locality prior rather than additional acoustic evidence. Deliberate substitutions from unrelated
surahs can be vetoed; a phone blur that resembles a word in the current surah remains eligible. Homophonous
attested word forms retain every identity, so any local identity keeps the accusation. The existing parallel
index describes four-word phrases with an interior difference; it does not enumerate every hifz association.
