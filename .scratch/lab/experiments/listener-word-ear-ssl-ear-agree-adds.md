# Word-ear identity confirmed by undistilled SSL tokens

Status: premise validation in progress. The experiment card is unchanged.

## Prediction and mechanism

Predict listener −2 and overall −1: two of the five registered additions won
with about one loss. The required premise is at least three registered targets
with an eligible word-ear extra word that the independent SSL ear confirms.

Reuse the saved short-crop Whisper-small int8 word decoder and the completed
plain-CTC-v4 undistilled w2v-BERT-2.0 int8 ear, without training or calibration.
The word ear proposes unmatched whole words in its sequence diff against the
next two written words over a detector-pause/tracker-step seam span. Existing
whole-mushaf-word, neighbour, prefix, local-fragment, back-run and formula guards
exclude ineligible identities. Its autoregressive confidence is ignored.

The SSL decoder retains its native 20 ms clock, original full-session chunk
phase, actual bounded left history and lookahead. It does not average its
short consonant spikes onto the primary 40 ms clock. A closed identity check
aligns each word-ear proposal to the SSL tokens using the existing substitution,
insertion, deletion and held-sound costs. The complete proposed word must fit
with at most one hard edit and strictly beat all local word, prefix and
contiguous back-run explanations on the same observed interval. A tie abstains.
The primary string, tracker and folded teacher insertion gate are unchanged.

## Validation and limits

The preliminary probe discovers seams from the card's primary evidence using
the actual free tracker. Labels select registered sites only after discovery;
they never supply crop bounds or candidate identities. It tests every contiguous
SSL token interval within each discovered span, a generous feasibility bound
before any emission, pause-boundary or certificate-transport restriction.
Failure to reach three targets even under that bound kills the premise.

Modal run 1 confirmed that both recorded exports remain available in supervisor
scratch. Word assets have the same hashes recorded by the short-crop parent;
the SSL export is the completed eligible plain-CTC-v4 model, not its collapsed
optimization predecessors. The auxiliary word and SSL models total about
1.23 GB; this probe does not establish on-device memory or latency suitability.

Unit coverage checks hard versus soft costs, full-word alignment, held-sound
normalization, strict competitor ranking and ties, local prefix/back-run
construction, native SSL timing and context dispatch, word reconstruction,
seam discovery and lexical proposal guards. No scorer, lab tooling, test data,
supplied manifest, evidence or experiment card is changed.

Practice score calls: 0 of 3. No score change or confirmed warning is claimed
until the target prerequisite is measured.
