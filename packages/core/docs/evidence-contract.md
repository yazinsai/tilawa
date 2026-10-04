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
