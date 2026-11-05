# Session-relative F0 continuity for seam spellings

Status: implemented; all four targets confirmed. Practice score pending.

## Prediction

Retain at least three of the four named added-word targets while removing false
spellings from ownership-free excise-certify. Predicted practice listener −2,
overall −1, on `lab-v1.5u`. Base commit `f97f6234`; the experiment card is unchanged.

## Mechanism

Re-land the final engine implementation from
`exp/listener-excise-certify-seam-waqf-adds`: exact non-neighbour whole-word
spelling, surplus-adjacent nasal preservation, pause-bounded utterance enumeration,
nearby-fragment exclusions, and forward-sum neighbour read-versus-omit certification
after cutting the spelled span. Candidate generation and its cache are unchanged.

The SDK measures raw-audio pitch independently of the phone model. It carries a
box downsampler across audio pushes, computes normalised autocorrelation over
50 ms windows on the listener's frame grid, and uses the first strong periodic
peak in the human-voice range 60–500 Hz. Correlation below 0.6 is unvoiced; the
first peak within 90% of the best avoids choosing an integer multiple of the
period. These fixed signal-analysis parameters are not fitted on practice.

Each token carries the pitch frames through its end, including preceding CTC
blank frames. Evidence serialisation and chunking preserve these extra token
fields. Gap-crop replacement keeps the original session's absolute raw pitch,
so cold-crop phase cannot change continuity. Phone identities, timings,
confidences, posterior rows and tracker code remain unchanged.

For a candidate's first-through-last token interval, measure the largest absolute
log2 F0 ratio between consecutive voiced observations, including across unvoiced
frames, and the longest internal unvoiced gap. Edge silence does not count as
an internal gap. Both must be no larger than the corresponding maximum over
this session's earlier complete zero-edit visits with mean confidence ≥0.8.
The calibration records persist after the alignment window scrolls. A word ending
after the candidate starts cannot calibrate it. Missing pitch, no voiced frame,
or no eligible earlier word makes this path abstain.

The new gate is applied after the unchanged generator chooses its span. Rejection
does not search for a different candidate. It can only remove generator emissions;
ordinary checker paths retain their original decisions. The acceptance cut adds
no numeric constant.

## Validation

Modal run 4 replayed all four full free-recitation sessions twice, gated and
ungated, using the card's shipped listener evidence
`exp-listener-stack-be988c-6a17f391`, phone variant. Pitch was extracted from the
matching full raw phone WAV using the same SDK implementation. Only temporary,
remote diagnostic instrumentation logged candidate metrics; it changed no
persistent source or saved evidence. Both modes confirmed all four additions.

| Target | Spelling | Candidate jump (octaves) | Candidate gap (frames) | Earlier word count | Session jump maximum | Session gap maximum | Confirmation |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| p19, 51:33:1 | وَمِن | 1.1370 | 0 | 63 | 2.5604 | 7 | 106.36 s |
| p50, 37:99:1 | وَقِفُۥۥهُم | 1.1313 | 0 | 62 | 2.7450 | 7 | 153.20 s |
| p31, 4:157:24 | وَءِن | 1.5665 | 1 | 42 | 2.6566 | 9 | 59.40 s |
| p66, 9:5:15 | وَءِن | 0.9204 | 3 | 60 | 1.9688 | 8 | 116.44 s |

The final parent write-up had reported no confirmation at 9:5:15. The byte-identical
generator on this card's newer base does confirm it, both gated and ungated. The
four-target premise is therefore measured on this branch rather than inferred
from the earlier branch's result.

A local byte comparison verified that the generator loop and cache, lexical
module, excision helper and CTC implementation match the final parent branch.

Modal run 2 found a pre-existing TypeScript build error in `reference.ts`:
`.filter(Boolean)` did not narrow `(string | undefined)[]` to `string[]`. An
equivalent explicit type predicate fixes compilation without changing runtime
word forms. Modal run 5 then built core and SDK and ran all six test shards.
All 754 core tests passed, including the parent excision tests and new integration
cases for a pitch reset, an internal unvoiced gap, missing pitch, and a steady
contour at both seam types. SDK pitch tests verify tone recovery, push-boundary
invariance and internal silence. Existing listener tests check identical CTC
runs and posteriors across push sizes. All 18 SDK tests passed; the optional end-to-end test was skipped. Practice score pending.
