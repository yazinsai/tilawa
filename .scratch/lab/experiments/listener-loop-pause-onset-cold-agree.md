# Pause-onset cold agreement for the teacher phone loop

Status: implemented; original-checkpoint acoustic premise passes on 3/6 targets.
The shipped teacher declaration is restored byte-for-byte. Runtime validation
with that pinned ONNX graph remains unavailable in the supplied data. One
official practice score returned overall +0 and listener +0; it does not prove
that the new certificates reached scoring. The card is unchanged. No fold.

## Prediction and mechanism

Predict listener −2 and overall −1. Reuse the identity-free teacher seam loop
from `exp/listener-teacher-phone-loop-seam-adds`: at least two consonant onsets,
existing `PHRASE_SWITCH` entry price, free CTC boundaries, contiguous repeat arcs,
marginal strictly above 0.5, and local-text, back-run, prefix and non-Quran guards.
The warm lattice and conditional insertion Viterbi extraction retain the
parent's behavior.

The additional condition is a second teacher pass whose audio starts at the
first detector-labelled run-end pause in the seam, after the preceding phone's
held argmax run. There is no left padding. The crop ends 300 ms after the
tracker's committed word-2 entry, rounded on the existing 40 ms clock. Its
lattice is `[optional loop | word 1, word 2]`, with the same loop construction,
repeats, price and majority cut. Emit a certificate only when both passes exceed
0.5 and their half-open Viterbi spans overlap. Missing pause, tracker step or cold
evidence causes abstention. No cut is fitted on practice.

This filter can only remove the parent's loop emissions. Relative to main,
the diff also brings in the parent's separate `phone-loop` certificate type
and checker consumption. As in the parent, a loop marginal above 0.5 replaces
the student support and `minConfidence` bars for that certificate. This is
additional behavior relative to main, not a change to ordinary added-word
confirmation or the folded exact-word teacher gate. The confidence bypass
matches the parent byte-for-byte. Exact-spell candidates retain their original
teacher lattice, asset and confirmation rules. Primary tokens, confidence,
frame bounds and posterior rows are preserved; `tracker.ts` is untouched.

The premise stop is fewer than three targets retaining cold majority mass with
overlap, or fewer than three recovered warnings after integration. The original
checkpoint acoustic probe retains three. Pinned ONNX integration is unmeasured,
rather than a measured count below three, so this retry does not declare a kill.

## Validation

Unit tests cover exhaustive CTC mass with an empty left text side, consonant
onset counting versus held frames, extra speech versus a blank pause, overlapping
versus disjoint spans, warm-only and cold-only evidence, missing pause or word-2
step, JSON certificate consumption, and primary evidence preservation. The
exact-word SDK test retains its original crop and certificate behavior.

This retry adds actual checker seam-discovery tests for a held argmax tail
followed by a pause and for a connected seam with no pause. SDK mocks are
cleared between tests to keep the checks independent.

Modal run 15 passed the supplied six-shard build and full core/SDK test harness
after restoring the registry: 767 core tests and 28 SDK tests passed; one
optional SDK help integration test was skipped. No timeout, runner or harness
was changed. Run 17 passed the final build, 20 focused core tests (including
both additional discovery cases), and all four SDK insertion-judgment tests.

Prior run 7 used the original fp32 teacher checkpoint with
`quran-muaalem==0.2.2`, `transformers==4.55.2` and torch/torchaudio 2.8.0 CPU.
It discovered seams with the free tracker/checker on the card's declared
listener evidence. Labels selected cases for reporting only, not crop edges.
The teacher received the SDK's `teacherFeatures` tensors. Full-inventory rows
were reconstructed with the SDK probability floor and scored on absolute
native frame indices. All six shards passed.

| Target | Cold marginal | Warm loop span (s) | Cold loop span (s) | Agreement |
| --- | ---: | --- | --- | --- |
| p07, 37:36:1 | 1 | 144.12–144.96 | 144.12–144.96 | yes |
| p33, 26:208:1 | 0.00000163 | 156.16–157.00 | 157.40–157.52 | no |
| p48, 12:76:1 | 0.0838024 | 40.04–40.56 | 37.44–37.64 | no |
| p56, 32:21:1 | approximately 1 | 104.80–109.08 | 106.28–109.08 | yes |
| p57, 51:18:1 | approximately 1 | 92.36–95.76 | 94.56–95.76 | yes |
| p57, 51:20:1 | 0.00154101 | 110.40–112.76 | 111.68–111.84 | no |

p33's detected pause starts at 156.96 s, after much of the planted span.
p48 loses majority mass and selects a disjoint interval. The second p57 span
overlaps but loses majority mass. No boundary or parameter was adjusted.
This is an acoustic premise probe, not a practice score or listening review.

## Correction of the previous attempt

Review correctly rejected the prior registry change: it replaced the shipped
teacher graph with an experiment re-export. This retry restores
`packages/sdk/src/registry.ts` exactly to main, including both pinned files:

| Asset | Bytes | SHA-256 |
| --- | ---: | --- |
| teacher.onnx | 11,342,825 | 5ee50c2b14d7e8e4808ee4adc33e6114f73481b607291136992a5d45444a1e18 |
| teacher.onnx.data | 2,422,885,632 | d3f24efba836219f0c556ad25e3a1046f9caf36b2ed639cd283f795c78da58b4 |

The approximately 2.43 GB figure describes graph plus external weights,
not the graph alone. The rejected graph was 10,559,069 bytes, SHA-256
`4e1d2ca950e17a952dd15f3494d0791084e588eaac119360e337eca3a4d244ed`.
Equal external-weight hashes do not establish graph equivalence. This graph
is not accepted as a substitute for the registered asset.

Runs 10–12 previously reported three recovered target warnings and 18 additional
wrong calls over five sessions through the SDK producer. Those runs used the
rejected export. Their target count does not validate this corrected candidate
with the pinned graph. The checkpoint acoustic probe above is independent of
that export and remains a separate, limited premise result.

Read-only Modal runs 14 and 16 inspected all supplied `/lab/data` and this
experiment's own scratch directory for `teacher.onnx`. The only graph found
was the rejected re-export in this experiment's scratch bundle. No pinned
graph was available for a corrected ONNX integration run. No other experiment's
scratch, external filesystem or alternative Modal access was searched.

The leftover untracked `.autolab/model.json` selected that scratch bundle.
It is removed, so the supervisor uses its normal main listener selection and
cannot silently substitute that graph through this experiment's export selector.
This experiment supplies no trained model or replacement manifest. The rejected
export is only an archived diagnostic artifact outside the selected model path.
The card's fixed listener ID is untouched. Runtime validation requires the exact
registered auxiliary asset; its absence is reported instead of changing its pin.

## Official practice result and limits

Practice score 1 of 3 ran on the previous implementation commit `056fcfa2`:

| Charged component | Main | Experiment | Change |
| --- | ---: | ---: | ---: |
| Overall | 24 (13 missed, 11 false) | 24 (13 missed, 11 false) | 0 |
| Listener | 23 (13 missed, 10 false) | 23 (13 missed, 10 false) | 0 |
| Tracker | 1 | 1 | 0 |
| Checker | 0 | 0 | 0 |

The aggregate result did not include traces or incident assignments proving that
the new loop certificates were scored. The card selects
`exp-listener-teacher-insert-marginal-gate-ad6eb8c0`, while the prior SDK probe
used the experiment scratch bundle. This +0 is not demonstrated precision for
cold agreement. No further score is requested without a valid pinned-asset
integration; one practice score has been used in total.

Leave the engine implementation for supervisor validation and the fold gate.
No scorer, lab tooling, supplied labels, test data, manifests, evidence or card
are changed. The inherited teacher path buffers audio and publishes auxiliary
judgments at session end; this change retains its model-size and latency limits.
