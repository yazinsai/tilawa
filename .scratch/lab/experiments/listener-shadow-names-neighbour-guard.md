# Shadow names, minus the neighbour's edge

Status: implemented. Left for the fold gate. No practice score from this worktree.

## Preregistered prediction

Test set lab-v1.5, practice, listener `exp-listener-heard-word-must-be-lexical-41570706`. The blank-frame branch won 4 and lost 2 (listener −2, overall −2). Dropping shadow names that belong to a neighbour keeps those wins and removes 1 or 2 of the losses: listener −3, overall −3, upside −4.

## Mechanism

Re-land the blank-frame read. On an omission the checker has already proposed, each blank-won frame no token owns, inside the stretch from the previous word's last confident phone to the next word's confident onset, contributes its top non-blank consonant. Rank only, no mass cut.

Two uses, both only on an omission already open:

- Two adjacent words that differ by one onset consonant. If that onset is the last shadow name before the heard body, the omission moves off that word onto the other.
- One word. The omission is dropped when a kept run spells a contiguous slice of it of at least two consonants and two distinct classes, and none of those consonants is written on the word to either side.

Before either use, two names are dropped. A blank frame whose class equals the token it touches is that token's tail or onset. A blank frame in the token-free run that touches a neighbour visit, whose class is that word's facing consonant (previous word: its last consonant; next word: its first consonant after a proclitic or a hamza), is that word's edge. A dropped frame is a barrier in the name string, so removing it cannot join the two sides into a new slice or uncover a new ending. No new constant. The rule cannot open an omission.

## What was checked

On the current main phone posteriors (the lexical listener and spliced-clean-int8 agree):

- p14, 36.84–36.88 s. Blank wins, top non-blank is ك (0.219 on the second frame). The run touches ء, not a ك token. عندنا's last consonant is ن. إنا's first consonant after the hamza is ن. ك stays. The placement test still moves 44:5 off كنا onto إنا, and does not move when that ك continues a ك token.
- p50, 275.40–275.44 s. م then ب on blank-won frames, touching neither an م token nor a ب token. The previous word ends in ه, the next word's first consonant after the proclitic is ل. مب stays, and it is a prefix of مبين. The veto test still drops that omission, and a reversed pair does not.

A smear of the previous word's last consonant between م and ب does not become مب. A pair that borrows و from the neighbouring word stays an omission. A ب that continues a ب token does not complete a veto.

Core and sdk tests on Modal.

## Limits

The guard only removes shadow names, so it can only undo a move or a veto the branch would have made. A planted omission whose two shadow consonants are themselves a neighbour's facing edge loses the veto; that is the leak being closed. p79 is unchanged: its onset is not a shadow name. Clean sessions are unchanged only while they propose no omission.
