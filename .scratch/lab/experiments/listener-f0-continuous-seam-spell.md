# Session-relative F0 continuity for seam spellings

Status: implemented; all four targets confirmed in fresh SDK decodes.
The requested practice score is +0 on existing listener evidence. Fresh SDK
decoding of the full practice set and the final fold decision remain with the
supervisor.

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
runs and posteriors across push sizes. All 18 SDK tests passed; the optional
end-to-end test was skipped.

## Fresh SDK validation

Modal run 9 loaded `/lab/data/models/spliced-clean-int8` through the normal SDK
model loader and decoded each target's entire phone recording through
`createEngine`, with 20 ms audio pushes and evidence recording enabled. This
includes the normal context-free gap replacement. Its newly generated evidence
was replayed through the unchanged free-recitation engine, with no source
instrumentation, truth-position input, or saved-evidence edits.

| Target | Added-word confirmation | All token pitch arrays finite |
| --- | ---: | --- |
| p19, 51:33:1 | 106.36 s | yes |
| p50, 37:99:1 | 153.24 s | yes |
| p31, 4:157:24 | 59.40 s | yes |
| p66, 9:5:15 | 116.44 s | yes |

All four confirm, exceeding the card's three-target premise. None confirms on
the existing pitch-free evidence replayed through this branch. The fresh token
records equal the cached records for p31; the other three are not byte-identical
to the cache, with small confidence differences visible in the event records.
This comparison establishes fresh-path coverage and target recall; it does not
establish byte equivalence between separately recorded decodes. No acceptance
cut or engine code changed during this continuation.

Modal run 8 passed all 754 core tests. Its slowest shard hit the already
documented Vitest worker RPC timeout after all tests passed, which the existing
test script accepts. A new SDK regression checks absolute pitch coverage through
explicit blank runs and chunk boundaries, plus JSON recording. Its first version
incorrectly assumed that the tiny model fixture emitted blank argmaxes; the final
test supplies a controlled blank/token sequence. Modal run 10 passed all 19 SDK
tests; the optional end-to-end test remains skipped. Both packages compiled.
The first fresh-decode probe (run 7) failed before decoding because core had not
been built; run 9 added that build step and passed on all four shards.

## Practice score and evidence coverage

Practice request 1 scored engine commit `a050edaa` and returned:

| Attribution | Main | Experiment | Change |
| --- | ---: | ---: | ---: |
| Overall | 30 | 30 | 0 |
| Listener | 29 | 29 | 0 |
| Tracker | 1 | 1 | 0 |
| Checker | 0 | 0 | 0 |

Both sides have 20 missed mistakes and 10 false warnings. The receipt's verdict
is `noise`. One of the three allowed practice requests was used.

This is a replay of existing evidence, not a fresh decode of the SDK change.
Read-only inspection of the existing workflow confirmed that `serve_practice`
invokes `lab run`, which reads the card/main listener's saved evidence. The
supervisor's later `after_tests` stage separately decodes listener changes through
this worktree's SDK and then scores that new evidence. No tooling was edited.
The new path deliberately abstains when raw pitch is absent. Therefore this
cached-evidence result cannot establish the F0 gate's precision, target recall,
clean-session behaviour, studio delta or incident tradeoff. The initial four
target confirmations came from replaying the existing phone strings with the
new SDK's measured raw F0 attached. The subsequent run 9 independently confirms
all four through fresh SDK decoding as described above.

The implementation is left in place for the supervisor's normal fresh decode
and fold gate. The target premise passed, so there is no premise-kill outcome
file. No threshold or candidate rule was tuned after the practice result.

## Limits

A greedy zero-edit visit is the calibration criterion, not an independent audio
transcription. Autocorrelation on phone-band audio can select a harmonic or a
subharmonic, and extrema retain even a single such jump. The measured session
maxima are consequently broad: up to 2.75 octaves and 9 unvoiced frames (360 ms).
Keeping four targets establishes recall, but does not establish that these maxima
reject the clean seam spellings that motivated the card. No pitch thresholds were
changed after observing the targets.

This implementation is causal: later words do not widen an earlier candidate's
band. No eligible prior word means abstention. Pitch is measured only within the
candidate's existing token bounds, which can truncate a held sound at a streaming
window boundary; neither those bounds nor CTC emissions were retimed. The
candidate set inherits all limitations of the parent generator. The diagnostic
replay did not substitute truth-timed windows or change the tracker.
