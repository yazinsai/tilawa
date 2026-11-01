# Context-free decode of the pause between matched words

Status: implemented, not practice-scored in this worktree. The supervisor scores the branch.

## Preregistered prediction

Test set lab-v1.5, practice, listener `spliced-clean-int8`. Listener −4 wrong calls, overall −3. The gain was expected from word-1 added words lost after a waqf, where the streaming model's left context favours the next verse word.

## Mechanism

At a pause between two consecutive reference words that the in-context decode matched confidently, the session re-encodes only that audio plus 300 ms either side, from a fresh encoder state. Phones whose centres lie in the pause are kept when they are not a suffix of the left word or a prefix of the right word. If what remains is already an added-word shape (at least three phones and two consonants, the checker's existing bar), those phones and the cropped posterior rows replace the in-context gap. The checker's added-word rule, including non-Quran phrases and the local-window check, is unchanged.

A pause shorter than the 300 ms pad is neighbour audio, not a gap. A hole longer than 8 s is a break or a skip, and is left alone. Breath and a held waqf vowel do not form an added-word shape, so they are not spliced.

This is not listener-pause-gap-redecode, which re-reads the same in-context posteriors. It does not retrain.

## Step 0

Non-blank posterior mass on the nine word-1 gaps, cropped window versus the stored in-context decode: the cropped window was higher on 6 of 9. The kill line was no increase on at least 5, so the card stays open. The increases are small. Several of those gaps already contain phones of the inserted word; the failure is often the local-window check marking a suspected addition `uncertain`, not a blank greedy decode. Long pauses in the same sessions mostly stay blank under greedy. A few emit one phone.

## What was validated

Re-encoding the labelled span and splicing both phones and posteriors, then running the checker at the true position, confirmed additions on four of the nine spans. Posterior replacement alone confirmed one of those.

The engine does not see the label. It uses the pause between confidently matched words. A full free-recitation decode of five of the targets through that path:

- `37:123:1` confirms as an added word on the true-position arm and on the free-recitation engine.
- `51:29:1` confirms the same way on both arms.
- `26:164:1` is now heard as the inserted word and still ends `uncertain`.
- `37:36:1` and `26:208:1` stay `uncertain`.

So a recovered addition can pass the unchanged checker without knocking the tracker off the verse. It does not do so for every word-1 miss. No practice score was taken here.

## Limits

The local window can still reject a word the cropped decode heard. Gaps the in-context alignment swallows into a neighbour are not the span that gets re-encoded, so a badly absorbed insertion can stay missed. New false warnings on fully clean sessions were not counted; the pause sample above did not produce an added-word shape, which is not a guarantee over the practice set. Studio audio was not re-decoded here.
