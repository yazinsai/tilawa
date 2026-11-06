# Pause-mask-trained streaming student

Status: training; structural and export checks passed. The card is unchanged.

## Prediction and premise

The card predicts listener −2 and overall −1, roughly two recoveries and one
loss. Its premise is that cutting pre-pause attention at inference put the
shipped student outside its training regime. Train that regime before asking
whether it can recover the planted post-pause consonants.

The registered seven incidents are checked acoustically in paired full-session
streams. Already having two planted consonants is not a new recovery. Record
ordered consonant counts, changes, and new two-consonant recoveries separately.
If fewer than three targets satisfy the acoustic premise, stop without practice
scoring. If three or more targets are won, request practice scoring before
judging losses. No practice-selected training recipe or constant is allowed.

## Built mechanism

`modal/student_pause_mask.py` shares the same masked streaming computation
between training and ONNX export. The deployed geometry remains chunk 16,
left cache 128, 45 fbank frames per window, 32-frame shift and 40 ms CTC frames.
Cutoffs are absolute on the native 20 ms embedding clock. Each attention query
masks keys earlier than its cutoff before softmax. Both depthwise convolution
branches and the ConvNeXt frontend pad mask earlier input samples. State and
positions advance normally. A pause starts at its labelled onset, so its own
frames remain available as warm-up. The downsampled stages use the corresponding
native clock with the cutoff rounded up to their frame boundary.

The training catalog uses only clean-match/train recordings from supplied s1,
with the existing gold exclusion function. A generic canonical-text matcher
requires consecutive complete ayahs inside one recording. No recordings are
joined and no audio is spliced. A frozen shipped streaming decode supplies the
existing 300 ms, blank-posterior-at-least-one-half pause labels on the original
audio before augmentation, so SpecAugment cannot create seam labels. Canonical CTC
Viterbi associates those pauses with ayah or existing mushaf-marked inner seams;
it does not label silence from text alone. Each such pause is masked with
probability 0.5, preserving its frames. Training closes the mask at the aligned
onset of the seam's second audible canonical word, matching deployment's bound.

Fixed recipe: seed 2026, at most 20 hours of eligible recordings, 800 AdamW
updates at LR 0.00005, canonical CTC plus shipped top-five KD (weight 2), and
KL(shipped || candidate), weight 4, on unmasked items. Retain the checkpoint's
RIR → noise → gain augmentation and SpecAugment. The direct streaming path has
carried caches and the deployed lookahead, with state gradients detached every
eight chunks. The ordinary checkpoint architecture and MatMul int8 quantizer
remain. Training does not read practice audio or labels.

The SDK mask input and seam selector are generic engine code. Every labelled pause
at an ayah or waqf-marked inner boundary opens a span, and the unchanged
primary tracker's next in-order word step closes it. Primary model feeds remain
unchanged when the manifest has no mask input. Primary strings and tracker rules
are unchanged. A declared auxiliary in the export bundle runs a second streaming
pass on the same audio. Its rows and tokens are carried as checker-only metadata
on the first primary token, surviving stored-evidence JSON and rechunking. The
checker replaces only declared span rows and tokens, while the primary tracker
continues reading primary phones. The folded candidate generator and teacher gate
read that scoped evidence unchanged. The existing teacher path publishes newly
certified additions at session end; this experiment retains that timing.

## Validation

Modal run 7 found 10,642 eligible original recordings from 95 speaker clusters;
the deterministic selection contains 4,428 recordings, 20.0106 hours. Eight
Python structural tests passed, including attention/cache isolation, convolution
and frontend-pad isolation, batched gradients, real-pause selection and
unmasked streaming-state parity. Five actual GPU updates completed; step 1 had
one real labelled pause and 89 masked CTC frames. The fixed 800-update recipe is
continuing in run 14; the five-update checkpoint is only an export smoke fixture.

Run 6 passed the supplied six-shard build/test harness: 756 core tests and 29 SDK
tests; one optional integration test was skipped. Run 11 passed the updated build
and nine focused SDK tests, including preservation of earlier labelled pauses
when another detector pause follows them.

The first export failed because legacy ONNX could not lower Unfold after dynamic
padding. Equivalent indexed sliding-window gathers fixed that export limitation.
Run 10 exported the five-update checkpoint and checked four carried-state calls:
unmasked, a cutoff inside a chunk, continued masking, then unmasked again. Float
ONNX probability differences were at most 1.8e-7, with cache values also checked
against PyTorch. Int8 differences were at most 0.001482; argmax phones agreed in
every checked frame. These random-window checks establish the export contract,
not target recall or population quality.

Run 8 decoded the six complete target recordings with the shipped SDK and
derived their primary tracker steps. Baseline ordered planted-consonant counts
on the seven labelled intervals are 1, 0, 1, 1, 2, 2, 3 in card order. The
span eligibility was provisional: the diagnostic initially used per-push
acquisition rather than the SDK's one-second acquisition clock. The final paired
diagnostic now uses that primary clock, and also records the already-folded cold
gap output. A recovery that this folded baseline already holds is not a training
gain. Labels report intervals and planted phones only; they do not select SDK spans.

The baseline exposed a selector defect: taking only the last detector pause
could start masking halfway through a blanked addition. The implementation now
applies every eligible seam pause, as registered, using the latest cutoff where
spans overlap. This changes no detector threshold or checker constant.

Resumed training exposed an Icefall eval-helper backward overflow at update 6.
Runs 12–13 verified that both saved checkpoints had finite weights, all three
losses had finite gradients at the posterior head, and ExpBackward in
SwooshRForward produced the invalid encoder gradient. Its selected forward value
stays finite after exp overflows, but its unused exponential branch still
poisons backward. The wrapper now uses the mathematically equivalent softplus
expression only when gradients are enabled, and retains the original helper for
inference/export. No data item was dropped and no training constant changed.
Run 15 passed the large-input gradient regression and unmasked streaming-state
parity checks. Loss/gradient guards prevent saving a corrupted optimizer update.

Run 14 passed all ten Python tests and continued training for its 48-minute
limit, reaching approximately 300 updates. The logged updates include ten masked
items out of thirty samples; these are sampled log entries, not population
counts. Losses and gradients stayed finite. Run 20 resumes the same 800-update
recipe and optimizer, without using practice results for any training choice.

Run 17 passed the supplied build/test harness: 758 core tests and 34 SDK tests,
including loading/releasing the declared auxiliary, private audio retention,
preservation of primary tracker inputs, and scoped evidence survival through JSON
and primary rechunking. One optional integration test was skipped. The harness
accepted a Vitest worker RPC timeout after all tests in that shard had passed.
Runs 21–22 passed the complete-encoder cache test, including a pause inside a
chunk with changed pre-pause input and old caches. All masked output rows were
bit-identical; unmasked output still depended on that history. The eleven Python
tests passed in run 21, and the extended focused case passed in run 22.

An additional late, unaligned-cut test in run 26 exposed pre-pause history
leaking through learned downsampling pools (maximum post-cut logit difference
0.807 when old caches/pre-pause input changed). The earlier aligned tests had
missed this. The shared wrapper now excludes earlier contributions inside a
straddling pooling group, propagates the maximum native cutoff to the pooled
stage, and zeros attention contributions where no eligible key exists. Run 27
passed all eleven Python tests, including exact complete-encoder isolation at
unaligned offsets. Float export probability differences were at most 2.4e-7;
int8 differences were at most 0.001366 with unchanged argmax on four carried-state
calls. These checks used old weights only as an export fixture.

The 574-update prototype was trained with the leaking mask and cannot establish
this card's premise. Run 28 restarts the fixed recipe from shipped weights in
`train-isolated`, reusing only the unchanged training catalog. Its preflight
extends isolation to every even native cutoff from 0 to 14, plus batched
late-cut gradients. Checkpoints now retain optimizer state at wall-time exit,
and an already-completed resume preserves its completed update count.
Run 28 passed this preflight and completed 224 fresh updates before its
48-minute training limit. Every logged loss and gradient remained finite;
10 of 23 sampled log entries were masked items, and 19 had labelled pauses.
Run 31 resumed the unchanged recipe and optimizer to update 503. Run 32
continues to the fixed 800-update endpoint and exports only at completion.
Run 33 waits for that final export, then runs the six complete recordings
covering the seven registered incidents with the primary acquisition clock,
the folded gap baseline, masked int8 stream, and unchanged teacher judgments.

The supplied student directory lacks teacher files. The unchanged original
teacher exporter recreates the public fp32 model. Its external weights match
the shipped SHA exactly (`d3f24efb...da58b4`), while its ONNX graph serialization
differs (`4e1d2ca9...4ed`, shipped `5ee50c2b...1e18`). The bundle exporter records
actual asset hashes in a local descriptor and requires original teacher weight
identity. The Node loader verifies declared files and fails on missing/corrupted
assets, leaving the global shipped registry and decision rule unchanged. Run 30
passed native fp32-versus-ONNX parity on three random feature lengths and eight
seconds of training-only learner audio: argmax agreement was 100% on every input,
with maximum probability differences 1.2e-7, 2.49e-6, 4.38e-6 and 1.91e-5.
Graph byte identity is not claimed. This finite check cannot prove parity on
every future crop; baseline and candidate diagnostics use the same teacher.

Run 30 passed the updated supplied test harness: 758 core tests and 36 SDK
tests, one optional integration skipped. Its asset-loading tests cover the
local descriptor, corruption, missing files and release on failure. The harness
accepted a Vitest RPC deadline after every test in that shard had passed.

Final training and trained int8 paired target streams remain pending. The paired
diagnostic records both folded-baseline and masked-candidate checker replay,
including unchanged teacher judgments. No practice-score calls have been made.

## Limits

Blank pause labels can include weak speech the shipped model missed. Canonical
training and the original KD teacher still carry their original supervision;
this experiment changes the context regime rather than adding insertion labels.
The finite-resolution stages quantize a pause cutoff on their native clock.
Teacher evidence is correlated model evidence, not an independent listening review.
