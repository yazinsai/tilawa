# listener-reliability-fused-subband-ear

Status: blocked on supervisor remote execution

## Prediction and decision rule

The card predicts two false warnings removed and one caught substitution lost:
listener -1, overall -1. The proposed cause is premature frequency mixing:
locally corrupted bands can contaminate a shared acoustic representation before
reliability is assessed. The card asks for two complete-pronunciation reversals,
but the task's final instruction requires at least three recovered targets to
continue. Below three, revert the engine change and record a measured kill. At
three or more, run the authorized practice score before making a gate decision.
No practice score or target comparison has been run yet.

## Implementation

`modal/student_subband.py` implements four independent temporal encoders over
overlapping log-mel slices [0,32), [16,48), [32,64), [48,80). Each encoder and its
reliability estimator sees only its own band. A learned softmax reliability
weight fuses the encoded representations before one normalized CTC phone head.
A kernel-9, stride-4 input layer produces the primary listener's native 40 ms
frame count. The six residual convolution blocks provide bounded temporal
context. Primary decoding and model weights remain unchanged.

The matched control has identical learned modules and exactly the same learned
parameter count and initialization. Its sole architectural ablation is a fixed
orthonormal DCT frequency mix across all 80 original bands before those same
encoders: every control encoder consequently receives broadband information.
It is a controlled early-mixing broadband architecture, rather than a claim
that a larger independently designed encoder would have equivalent capacity.
Both arms receive each identical input batch and corruption realization.

Training selects only authorized s1 train/clean-match recordings, with entire
speaker clusters containing non-training or gold clips excluded and unclustered
source clips excluded, plus help-train none-labelled recordings without events
or repeats. Explicit non-training help speakers are excluded entirely. Validation
holds out complete speakers from the remaining authorized help training set;
checkpoint selection minimizes the pair's mean validation loss before any
practice inspection. Source CTC uses the existing spliced-target recipe and
available teacher top-five distillation. Help-none takes use their canonical
transcript, with held runs collapsed using the existing training utility. A
quarter of inputs remain unmodified; other inputs retain the shipped room,
noise, gain and SpecAugment chain, plus independently corrupted spectral band
views. Physical corruption masks supervise relative reliability without phone
labels. Neither practice audio nor practice transcripts enter fitting.
Compatibility with the supplied files and speaker-exclusion provenance still
requires remote inspection; neither has been verified by reading unavailable
remote data.

Core exposes only the exact full written/rival strings and native frame bounds
of an otherwise supported non-graph wrong-word confirmation. The SDK runs both
side graphs on identical acoustic features and scores those same frame intervals
with ordinary complete-sequence CTC forward sums, marginalizing boundaries. It
issues a certificate only if all four scores are finite, fusion strictly prefers
the written string, and the control strictly prefers the rival. Ties, unknown
phones, impossible completions and missing intervals abstain. Core matches the
word, both full strings and exact interval before downgrading the public verdict.
The original internal confirmation and ownership ledger are retained, preserving
later merges and insertion decisions. Graph confirmations remain exempt.

Audio sessions retain PCM for an end-of-session recheck. The SDK publishes an
explicit downgrade only for the exact live occurrence removed by an otherwise
identical replay of original live evidence. Certificates are serialized with an
unchanged primary token for the scored evidence replay. The buffered completion
route is a latency and memory limitation; immediate on-device warning vetoes
have not been established.

The Node loader recognizes `subband-ear-asset.json`, checks all declared files'
byte counts and SHA-256 hashes, validates inventory, native clock and equal
capacity, and binds the config to the exact primary graph hash. A declared
missing/corrupt pair fails loading. Export copies the existing primary bundle,
exports both ONNX ears, verifies PyTorch/ONNX numerical agreement on multiple
lengths and writes a hashed side descriptor. Those procedures are implemented,
not yet executed. Existing auxiliary primary assets must also be present in the
completed bundle.

## Validation and remaining work

Python compilation and `git diff --check` pass. New Python tests exercise band
isolation, broadband control mixing, exact parameter equality, physical
reliability targets, normalization and the native output clock. New core/SDK
tests exercise complete CTC sums, strict reversal, exact-window certificates,
graph preservation, audio completion downgrades, unchanged positions/insertions,
and declared missing/corrupt assets with resource cleanup. Both TypeScript builds
pass. The complete local SDK suite passes 38 tests, with one optional help-data
test skipped. All 761 core assertions in 54 files pass, including the five new
certificate tests. The core runner exits 1 because of two `onTaskUpdate` worker
RPC timeouts, with no failed assertion or failed test file. This is the sole
Vitest error explicitly accepted by `tools/autolab_tests.sh` when every test
passes; the remote harness itself remains unexecuted. No existing test was
changed to suppress a failure. The actual pure source-selection function also
passes a local exclusion check without loading Torch. The PyTorch mechanism
suite remains unexecuted; compilation and source filtering do not validate
PyTorch behavior or acoustic quality.

Supervisor run 1 attempted a training-image inventory; run 2 attempted a minimal
CPU inventory; run 3 attempted the prescribed six-shard test harness; run 4
requested H100 mechanism tests and a short paired training smoke run. All four
returned `NotFoundError: No such file or directory` without remote command output
or returned artifacts. The three CPU requests reported zero cost; the H100
failure reported $1.0127 spent, leaving $178.99. No remote training or test
execution could be verified. The user has been asked to restore the supplied
execution environment. These failures provide no acoustic evidence for or
against the premise. Target comparisons: zero; practice scores: zero of three.
The experiment is blocked, not killed or accepted. No fold or discard occurred.

No `.autolab/model.json` or purported trained asset bundle is declared. Scoring
the ordinary main listener would not evaluate this change. Before this experiment
can be handed back as evaluable, inspect the authorized data, verify exclusions,
run all tests, train/freeze both arms, export and validate the complete primary
and side bundle, and compare the registered targets on engine-requested windows
using `packages/sdk/src/bench/subband-ear-cli.ts`. Only then apply the target
continuation rule and, if appropriate, practice-score. No scorer, lab tooling,
supplied manifest, evidence, test data or experiment card has been edited.
