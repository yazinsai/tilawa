# Word-ear identity confirmed by undistilled SSL tokens

Status: three target warnings verified; final tests and practice scoring pending. The experiment card is unchanged.

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
scratch. Word asset sizes match the short-crop parent; hash verification is pending;
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

## Acoustic premise probe (Modal run 6)

The probe used the saved short-crop word decoder and native SSL decoder on
the generic spans discovered from the primary stored evidence. It found:

| Registered site | Word ear and SSL result |
| --- | --- |
| p07, 37:36:1 | Word ear proposes `سليم` or `سبيل`; SSL `تَلِم` cannot confirm either within one hard edit. |
| p48, 12:76:1 | Word ear proposes `كذب`; SSL `فَدرفَبَدَءَبِءَ` cannot confirm it. |
| p56, 32:21:1 | Planted `طين` fails the hard-edit budget. Another word-ear proposal, `عنهم`, agrees with SSL `عننَهُم` at 111.94–113.10 s. |
| p57, 51:18:1 | Proposed `قبل` agrees with SSL subinterval `َبڇل` at 95.36–95.76 s: cost 10, one hard edit, best competitor 15. |
| p57, 51:20:1 | Proposed `قبل` agrees with SSL `قَبڇل` at 112.34–112.74 s: cost 0, zero hard edits, best competitor 20. |

Thus three slots have an eligible proposal/SSL agreement, but only two
agreements identify the planted word itself. In p56 the agreeing material is
later than the planted interval (106.76–109.075 s) and may be a fragment of
the written continuation. It must be evaluated through the unchanged warning
matching and fold gate rather than credited as an acoustic identity recovery.
The subinterval search remains unchanged for emission verification.

## Built emission verification (Modal run 8)

A second diagnostic fed the actual saved word/SSL acoustic outputs through
`addWordSslEvidence`, the core checker and the unchanged session-event matching
rules. It covered the four target sessions only, not the full practice set.
Primary tokens and posterior rows were retained; labels selected only the
reported targets, never an engine identity or bound.

The built change recovered 3 of 5 registered misses: p57 at 51:18:1 (warning
95.76 s), p57 at 51:20:1 (112.76 s), and p56 at 32:21:1 (113.12 s). p07 and p48
stayed missed. The p56 warning is credited by the existing location/occurrence
and six-second confirmation window, even though it names the later `عنهم`
fragment rather than the planted `طين`. No clean-seam precision is established
by this selected-site diagnostic.

The SDK now carries dedicated word-agreement certificates with the measured
edit count and strict rank advantage; it does not fabricate a teacher marginal.
The checker rechecks the folded lexical, fragment, phrase and seam guards,
then emits through the same end-of-session seam path with duplicate
suppression. The normal Node directory loader installs both declared ears
lazily and releases their backends with the primary model. The existing
teacher file pins and its insertion lattice code are unchanged.
