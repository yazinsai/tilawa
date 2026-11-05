# Separator-head veto on ownership-free spelled seam words

Status: implemented; four targets retained; practice evidence replay unchanged;
supervisor model decode and fold gate pending.

## Prediction

Keep at least three of the four named excise-certify added-word targets, with
losses reduced from 18 to about three: listener −2, overall −1. The experiment
card remains unchanged. No threshold is fitted on practice.

## Mechanism

Re-land the final `listener-excise-certify-seam-waqf-adds` generator: entire
pause-bounded shipped greedy utterances, surplus-adjacent hum retained as nun,
exact non-neighbour spelling, nearby text-fragment exclusions, and forward-sum
read-versus-omit neighbour certification after cutting the candidate frames.
Candidate selection and its cache are unchanged.

After that generator selects its emission, require complete separator-head
coverage of its first-to-last token interval and reject it if any strictly
internal frame has greedy separator class `|`. Separators at the endpoints are
allowed. A veto cannot trigger a search for a different candidate. Missing head
coverage also vetoes the emission.

The frozen separator projection supplies an auxiliary binary output on the
shipped encoder features, sharing its existing caches. The SDK carries full-stream
head coverage on phoneme metadata through recording, evidence re-chunking and
cold-gap replacements. Cold-crop separator outputs are discarded rather than
mixed with the full-session frame clock. Phone strings, main posterior rows and
tracker inputs keep their original values.

## Validation

Modal run 6 passed the TypeScript builds and all 752 core and 17 SDK tests;
one optional SDK end-to-end test was skipped. Tests cover an internal-boundary
veto, endpoint boundaries, missing and discontinuous coverage, and identical
main phones/posteriors with auxiliary head output.

Modal run 10 replayed the retained original fp32 separator export
(`word-sep-v2`, model SHA-256
`27c85f6ffcfc5dcbd691a4be5f8cfef71ca4de1598bef87089ee244b57b269a0`).
All four original spans have no internal separator, and all four gated additions
confirm in the free engine replay:

| Target | Inclusive shipped token frames | Spelling | Internal separator frames |
| --- | --- | --- | --- |
| p19, 51:33:1 | 2552–2569 | وَمِن | none |
| p50, 37:99:1 | 3662–3692 | وَقِفُۥۥهُم | none |
| p31, 4:157:24 | 1386–1403 | وَءِن | none |
| p66, 9:5:15 | 2805–2824 | وَءِن | none |

The parent-span census and these replays use this card's baseline evidence,
`exp-listener-stack-be988c-6a17f391`, without changing its phones or rows. The
separator export survived under its parent's scratch directory, so no training
or practice-fitted parameter was needed.

Modal run 13 attached 19 head nodes to the shared mid-stack and final features,
adding 183,439 bytes and no cache states. The resulting model is 67,943,183
bytes, SHA-256 `0d8ab96b53327c00f82fa5cccad42c9e39a8e5534df3c781ac18bc9daa6b9d75`.
Modal run 14 passed all four exported-model replays: each candidate again has
no internal separator and each addition confirms in the free engine. Main
phonemes and top-k posterior output were byte-identical to the shipped model
on the first ten seconds of each of the four real recordings. The original
phone graph, weights and cache outputs are copied unchanged into the export;
the 19 auxiliary nodes read features without feeding back into them.

The model is exported under this experiment's scratch directory as
`model-separator`; `.autolab/model.json` selects it for supervisor decoding.
Temporary scripts containing target identifiers were removed before scoring.
The only change outside the seam/head paths is an explicit string type
predicate in `reference.ts`, needed to compile the existing reference-form
filter without changing which values it retains. Initial diagnostic failures
(syntax and audio-path assumptions) produced no practice scores.

## Practice result

Practice score 1 on commit `4240fcff` returned **overall +0, listener +0**
(noise). Both main and this branch have 30 wrong calls: 20 missed mistakes and
10 false warnings. Attribution is 29 listener, 1 tracker and 0 checker in both.
The receipt is `.autolab/practice-result-1.json`; only one of the three allowed
practice requests was used.

This result does not measure the exported separator model. The practice helper's
`serve_practice` calls `lab run`, which replays the card's existing listener
evidence (or its current main-line successor). It does not read
`.autolab/model.json` or decode a replacement model. Existing evidence lacks the
auxiliary separator coverage, so the new generator's emissions are vetoed and
the replay is unchanged. The supervisor's separate `after_tests` path copies the
selected model and decodes it before its final score; that remains necessary to
measure precision, studio effects, clean-session warnings and the fold gates.

The premise kill did not fire: all four target spans are separator-free and
confirm with the exported head in the diagnostic free-engine replays. Keep the
implementation and model selection for the supervisor's decode. No threshold was
tuned and no second practice replay was requested, since it would use the same
head-free evidence.

## Limits

This veto can only remove the parent's emissions. It cannot recover a missing
candidate or repair a word the greedy listener does not spell. A head that
under-emits separators may preserve cross-word false spellings as well as real
words. Absence of a separator is the proposed evidence, not a proof of a real
addition; the practice fold gate must price the resulting losses.
