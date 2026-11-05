# Teacher whole-seam insertion marginal

Status: implemented; acoustic premise and SDK target validation passed. Stored-evidence practice replay was unchanged; fresh supervisor SDK decoding remains necessary. The experiment card is unchanged.

## Prediction and mechanism

Predict three of the four registered additions retained, listener −2 and overall
−1 on lab-v1.5u. The ownership-free excise-certify generator is restored from
`exp/listener-excise-certify-seam-waqf-adds`: exact soft-only whole-word spelling,
candidate-local nasal handling, pause-bounded utterances, local-fragment exclusion,
and forward-sum read-versus-omit neighbour certification after excision. Its
enumeration and certification are preserved; the proposed gate only rejects its
emissions.

The gate uses the original fp32 KD teacher on a warm crop of the whole seam.
The CTC lattice contains the previous two words, an optional arc for the exact
spelled word, and the next two words. It charges the existing `PHRASE_SWITCH`
on entry to the insertion. Blank timing and word boundaries are free. Existing
contiguous word and text-side repeat transitions explain restarts without filler.
Forward-backward sums the probability mass that enters the insertion arc over all
timings. Only a marginal strictly above one half permits emission. No numerical
threshold is calibrated on practice.

The acoustic premise is tested before packaging the teacher or deploying the
SDK gate. If fewer than three registered targets survive, the task requires
rollback, a killed outcome, and no practice score. A surviving implementation
must additionally prove normal SDK asset loading and scored-evidence transport;
the older adjacent-file teacher loader did not establish that deployment path.

## Validation and limits

The lattice has exhaustive-path unit checks for small vocabularies, including
blank placement and equal labels across a skipped insertion. Additional tests
cover real extra speech versus neighbour-only speech, contiguous repeats,
duplicate pronunciation forms, missing evidence and impossible reads.

Target diagnostics replay the free tracker and checker on the card's declared
listener evidence. Labels select and report cases only; they do not choose the
candidate or its insertion frames. Warm-crop and whole-seam bounds are obtained
from the actual generator visits in the tracker/checker replay. Teacher frame indices
remain on their native 40 ms clock. Full-vocabulary fp32 posteriors avoid a
top-k reconstruction artifact. These acoustic diagnostics are not a practice
score or an independent listening judgment.

The original teacher on the card's current listener evidence retained all four
registered target spellings (Modal run 6):

| Target | Spelling | Insertion marginal across growing seam windows |
| --- | --- | --- |
| p19, 51:33:1 | وَمِن | 0.9999956–0.99999997 |
| p50, 37:99:1 | وَقِفُۥۥهُم | approximately 1 |
| p31, 4:157:24 | وَءِن | 0.999999999 |
| p66, 9:5:15 | وَءِن | 0.5733–0.7357 |

Run 8 exported the same teacher as fp32 ONNX with external weights. The primary
model and its manifest are unmodified copies of `spliced-clean-int8`. Export used
torch/torchaudio 2.8.0 CPU, quran-muaalem 0.2.2, transformers 4.55.2, onnx 1.23.2
and onnxscript 0.7.2. Teacher graph: 11,342,825 bytes, SHA-256
`5ee50c2b14d7e8e4808ee4adc33e6114f73481b607291136992a5d45444a1e18`;
external weights: 2,422,885,632 bytes, SHA-256
`d3f24efba836219f0c556ad25e3a1046f9caf36b2ed639cd283f795c78da58b4`.

The SDK declares this auxiliary asset in its registry and lazily resolves it from
the student's installed bundle through the normal Node model loader. It runs
only on proposed seams, retains the primary decode and posteriors, and carries
span-specific marginals on the first decoded token through ordinary evidence
serialization. Missing teacher evidence disables this experiment's emissions.
Audio-backed sessions publish additional certified seam warnings at `end()`;
other live events and the primary tracker are preserved. This auxiliary path
currently buffers session audio and uses CPU fp32 inference. Its 2.43 GB model
size and end-of-session latency do not meet the device budget or an immediate
streaming warning requirement. The experiment tests the acoustic precision
mechanism, not deployment readiness.

Run 9 passed the supplied six-shard build/test harness: 756 core tests and 21 SDK
tests; one optional SDK integration test was skipped. The harness accepted only
its documented Vitest RPC timeout after every affected test had passed. A
pre-existing reference-loader TypeScript inference error required an explicit
filter type predicate, with identical runtime behavior. Run 11 then passed the updated build, 14 focused core tests, and all 25 SDK
tests (one optional integration test skipped). These include native warm-crop
timing independent of the candidate interval, shared crop inference, JSON and
frame-cell transport, and private audio retention plus end-of-session warning
publication with recording disabled.

Run 12 passed the actual Node SDK path on all four targets, using the exported
ONNX teacher and current card evidence. The same marginal ranges in the table
above survived the SDK frontend and runtime. Ordinary JSON serialization, native
frame-cell rechunking and the default checker produced a confirmed addition at
each registered location. Across each complete target session, primary phoneme
identity, confidence and frame bounds, primary posterior values, and the entire
tracker event sequence compared equal with and without teacher metadata. The
probe discovered all candidates generically; target labels only selected the
cases and reported the warnings. Run 12 took 747 seconds in the container.
The kill criterion therefore did not fire: four of four targets retained.

One practice score completed at implementation commit `0b8699a3`; two remain
available. Overall stayed at 30 wrong calls (20 missed, 10 false), listener at
29 (20 missed, 9 false), tracker at 1, and checker at 0: all changes zero, verdict
noise. The card remained unchanged and still selects
`exp-listener-stack-be988c-6a17f391`.

This was a stored-evidence compatibility replay. `serve_practice` invokes
`lab.py run`; its score command runs the core lab CLI on that existing listener
evidence, rather than decoding the selected SDK model bundle. Those primary
tokens contain no `seamInsertions` certificates, so the experiment's emissions
are disabled by construction. The replay therefore does not measure the
teacher's population precision, target gain, clean-session behavior or studio
regression. It must not be reported as a measured zero gain for the new SDK ear.

The supervisor's normal post-implementation `after_tests` path reads
`.autolab/model.json`, copies the exported bundle, and invokes the SDK decoder
on the practice phone and studio sessions before scoring its new listener
evidence. The selection is `{"path":"bundle"}` and the exported assets persist
under this experiment's scratch directory. Leave the implementation for that
fresh decode and the fold gate. Four retained targets pass the card's premise;
no conclusion about the aggregate precision gain is established yet.

Diagnostic sources have been removed from the engine/source snapshot and
retained only with the ignored run artifacts. No scorer, supplied manifest,
test data, evidence or experiment-card edits were made. No fold was performed.

A final no-certificate fast path avoids excision scans when neither teacher
metadata nor a discovery callback is available. It preserves the generator when
evidence exists and removes unnecessary work on ordinary primary-only clients.
Run 14 passed the updated build, the 14 focused core tests and all 25 SDK tests
(one optional integration test skipped).
