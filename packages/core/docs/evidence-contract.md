# Listener evidence contract

The tracker accepts `PhonemeEvidence` chunks. Every listener (the Muaalem teacher pass, the student models, and fixtures) must produce this shape.

## Inventory

- `inventory` is an id. The tracker rejects any chunk whose id differs from the reference index's `inventory.id`. The current id is `muaalem-v3_2-phonemes`.
- The inventory has 42 single-character `quran-transcript` phoneme tokens, listed in `inventory.tokens` of the index asset.
- Token id `0` is the CTC blank. Token ids `1..42` map to `inventory.tokens[id - 1]`, in the same order as the phoneme level of `obadx/muaalem-model-v3_2` `vocab.json`.
- A decoded phoneme outside the inventory is rejected.

## Frames

- Times are frame indices, never wall-clock times. `frameMs` is the frame stride; it's 20 ms for w2v-BERT and depends on the model for students.
- Frame indices are absolute within the session and increase monotonically across chunks. Every chunk in a session uses the same `frameMs`.
- `DecodedPhoneme.start`/`end` are the first and last frames of the token's collapsed CTC run.

## Top-k posteriors (optional)

- Each row of `ids`/`probs` holds the k most probable tokens for one frame, in descending order of probability. The blank may appear among them.
- Probability mass outside the top-k, `1 - sum(row)`, is spread uniformly over the tokens not listed in that row. Each of those tokens gets at least `1e-6` after normalisation, so that log-scores stay finite.
- If no blank appears in a row, the blank is treated as an omitted token.
- Rows cover frames `startFrame .. startFrame + nFrames - 1` and must line up with the decoded phonemes of the same chunk.
- Leaving out `posteriors` is valid. The tracker then uses decoded phonemes only.
- Either every chunk in a session carries posteriors or none does. The tracker rejects posteriors when `ids`/`probs` don't hold `nFrames x k` entries, when rows don't continue the previous chunk's frames, when an id falls outside `0..N`, when a decoded phoneme's frames fall outside the rows (a stride mismatch), or when fewer than half the decoded phonemes appear in their own frames' top-k (a different token order).

## Posterior reranking

- Retrieval and PER always use the decoded phonemes. Posteriors only rescore the shortlist that retrieval returned.
- A candidate's alignment is read as a label sequence: reference tokens where it matches, substitutes or deletes, and observed tokens where it inserts. The ranking cost is that sequence's CTC negative log-likelihood over every frame of the evidence, minus the decoded sequence's own, at 3 nats per edit. Insertions, a mid-word start and a basmala prefix add their decoded costs on top.
- Held-sound runs (madd letters, `ں`, `۾`, and noon/meem runs of 3 or more, which become 2) collapse before scoring on both sides, so reciters' lengths aren't charged. Length stays the decoded alignment's job.
- Fit uses the lower of the decoded and posterior costs, so posteriors can rescue a mangled decode but never fail a candidate that the decode fits.
