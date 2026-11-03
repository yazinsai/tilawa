# Pause-onset glottal hamza

Status: implemented, not practice-scored. The supervisor scores the branch.

## Preregistered prediction

Test set lab-v1.5, practice, phone. The p14 pair flips: the missed omission `44:5:4` and the false omission on `44:5:5`. Listener −2, overall −2. `44:23:1` was not counted.

## Mechanism

A vowel that starts after silence is a glottal attack. When the phone channel has already erased the onset burst, the student decodes that attack as ء. In the checker alignment that ء is an exact match for a written hamza, so the heard body is given to the hamza word and the omission lands on its neighbour.

The decoder tags `pauseOnset` on a ء that is the first non-blank token after a pause. A pause is the checker's existing test: a blank run of at least `minGapFrames` (`GAP_PAD_S`, 300 ms) in which every frame is blank-owned (blank probability at least 0.5). While aligning, that ء costs `UNIT` against every word-onset consonant, the same charge a mismatched onset already pays, instead of matching ء for free. The rest of the phones keep their costs. The vowel then picks the word. A hamza that was actually spoken still matches on the phones after the ء, the token string is unchanged, and an omission is only moved when those costs change which of the two words wins.

## What the p14 trace showed

Oracle words on phone, listener `spliced-clean-int8`. `44:5:4` إِنَّا is absent (truth ends `44:5:3` at 36.26 s and resumes at `44:5:5` at 36.88 s). The decode of كُنَّا is `ءُننننَاا`.

| | |
| --- | --- |
| ء | 36.92 s, frame 923, conf 0.55 |
| blank run before it | frames 907–922, 16 frames, 640 ms |
| pause test | `minGapFrames` is 8; lowest blank probability in the run is 0.636 |

The ء is the first non-blank token after that pause. A later 640 ms blank sits inside the ghunna (between ن and ن) and is not a hamza, so it is not tagged.

## Validation

- Unit: with إِنَّا deleted and كُنَّا's ك heard as ء, the omission stays on `44:5:5`. The same audio with that ء flagged `pauseOnset` confirms `omitted_word` on `44:5:4` only. Flagging the real ء of a spoken إِنَّا while كُنَّا is deleted still omits كُنَّا. Flagging it when both words are spoken confirms nothing.
- SDK: a ء after a blank-owned run of `minGapFrames` is `pauseOnset`, including when the run crosses a decode window. A shorter run, a run whose blank probability is 0.4, a non-hamza, and a ء with no preceding blank are not tagged.
- Core and sdk tests: `bash tools/autolab_tests.sh` on six shards, every shard exit 0.
- p14 phone, full session, flag applied with the same pause test on the stored top-k (the decoder's argmax probability). 28 hamzas tagged. Confirmed calls, engine and oracle tracker: lost `omitted_word 44:5:5`, gained `omitted_word 44:5:4`, both at 50.08 s. Nothing else moved. `44:23:1` did not flip.

## Limits

Only p14 phone was replayed. The other tagged hamzas in that session did not move a confirmed call; studio and the rest of the practice set were not run. The flag is computed at decode time, so a stored evidence file without it behaves as before. Practice was not scored here.
