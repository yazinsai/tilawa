# Fading-memory phone readout

Status: blocked by the supplied supervisor execution environment. The engine,
training utilities and tests are prepared; no acoustic model has been fitted,
no complete bundle exists, and no target or practice result is established.
The experiment card is unchanged. This is not a measured kill or fold result.

## Prediction and stopping rule

The card predicts one false warning removed with no counted losses: listener
−1 and overall −1. Its hypothesis is that a nonlinear summary of preceding
acoustic changes can distinguish histories whose final encoder features are
similar. A useful reversal must prefer the complete written pronunciation
while both matched controls still prefer the already accused pronunciation.

The task's stopping rule requires at least three target warnings actually
removed before practice scoring. Fewer than three requires reverting engine
changes and recording a measured kill. No target win count is currently known;
the infrastructure error provides no evidence for the acoustic premise.

## Prepared mechanism

`modal/student_fading_memory_export.py` exposes the frozen deployed encoder's
actual final linear-head input in a separate ONNX copy. For a dynamically
quantized head this means dequantized activation codes, with effective weight
scales and zero points included. Export requires exact equality of every
primary output and cache over eight consecutive streaming windows, identical
feature/logit clocks, and reconstruction of the final affine projection.
These acoustic-model checks are implemented but have not run.

The SDK CLI caches each complete recording using its existing fbank and
streaming windows, with separate encoder cache state. Only usable help-train
training-speaker `none` takes without edits or repeats are accepted. An explicit
excluded-speaker list and every speaker appearing in a non-training label are
excluded before caching. The trainer independently rejects excluded speakers
and non-training observations. The supplied training copy excludes release
speakers by construction; its actual contents remain uninspected.

Canonical transcripts supervise ordinary complete-transcript CTC. Text, engine
word boundaries, previous predicted labels and practice examples never drive
the reservoir. Feature centering and scaling use fitting speakers only. The
fixed state update is

    h[t] = tanh(A x[t] + 0.8 P h[t-1])

where A is a fixed-seed random input matrix with 64 rows and P is a cyclic
permutation. Because tanh is 1-Lipschitz and P is orthogonal, state differences
contract by at most 0.8 when the inputs agree. State starts at zero once per
recording and continues through pauses and encoder/chunk boundaries.

Each arm has a linear full-inventory phone readout, including blank, over the
normalized current features plus 64 nonlinear coordinates. The memory arm
uses h[t]; the instantaneous control uses tanh(A x[t]); the shuffled control
uses the same reservoir on a deterministic text-blind permutation of the
recording's feature rows, then restores each summary to its current frame.
Shuffling preserves the current acoustic observation while destroying ordered
history. Its offline permutation can contain future observations, so this
control does not establish a deployable streaming control or equal latency.

All three arms share trainable parameter count, initialization from the shipped
affine head, examples, optimizer and update order. Only the output heads train;
the encoder and reservoir remain fixed. The default budget is 40 CPU epochs,
bounded to at most 120, with two Torch threads. A deterministic speaker-grouped
partition within training speakers selects one common checkpoint by mean
validation transcript CTC across the arms, preserving equal update budgets.
Reports retain partition membership, initial and selected validation losses,
parameter counts and the cached-observation checksum. No settings have been
selected against practice outcomes.

The checker exposes only its already-supported complete string-path wrong-word
comparisons. All arms process the full recording before slicing those exact
existing windows. Complete written and accused sequences receive ordinary CTC
forward scores under identical frames and topology. Certificates require finite
strict written preference from memory and strict rival preference from both
controls. Core revalidates these comparisons and matches both full strings,
word identity and exact frame bounds before withdrawing the public verdict.

Primary phonemes, confidences, posteriors and tracker input remain unchanged.
Graph confirmations are exempt. Internal confirmation and ownership remain
intact for later issue generation and insertion merges. Audio sessions retain
certificates in recorded evidence and publish an uncertain downgrade only for
the exact live occurrence removed by identical-primary-evidence replay.
Only actual live events recorded by the string-comparison callback can be
retracted, preserving live graph confirmations even if replay cells differ.
Existing insertion evidence keeps its own route.

The experimental side review buffers audio and runs at session end. Timely
streaming warnings, device latency and memory use remain unvalidated. Retaining
positive written evidence and two rival-preferring controls can limit
intervention, but does not guarantee preservation of genuine substitutions.

The Node loader accepts an optional adjacent `fading-memory-asset.json` only
when both declared files exist, match their byte counts and SHA256 hashes, and
are tied to the exact primary inventory, checksum and 16 kHz/40 ms geometry.
Declared missing or corrupt assets fail instead of falling back to the primary.
Packaging copies the supplied primary directory, including its existing
auxiliary assets, without changing its manifest or model bytes. A training-only
parity fixture compares exported JavaScript readouts against Torch outputs.
No `.autolab/model.json` is declared; scoring the primary alone would not
evaluate this implementation.

## Validation and limits

Three supervisor requests failed before command execution with
`NotFoundError: No such file or directory`. Each result reports $0 compute
and no shard log:

| Request | Purpose | Result |
| --- | --- | --- |
| 1 | Inspect permitted training data, model availability and scratch storage | Pre-execution failure |
| 2 | Minimal working-directory/data probe | Same failure |
| 3 | Six-shard `bash tools/autolab_tests.sh` | Same failure; suites did not run |

Restoration of the supplied execution environment was requested. No alternate
Modal access, files outside this worktree, or release benchmark access was
sought.

Local lightweight checks passed: Python syntax; TypeScript syntax parsing;
`git diff --check`; reservoir contraction and history retention; deterministic
shuffle alignment; exact CTC sums and repeated-label blank requirements; strict
three-arm reversals and unknown-phone abstention. A Node assertion adapter
executed all 14 synthetic checks in the new core and SDK behavior test files,
including exact-window rejection, graph preservation, unchanged issue creation,
training eligibility and independent full-recording encoder-cache continuity.
This did not use Vitest and does not establish TypeScript type correctness.

Unit tests also cover audio-session certificate transport and missing/corrupt
declared assets; those tests and the existing core/SDK suites remain unexecuted.
The existing insertion-loader mock was adjusted to report absent optional files
as ENOENT instead of returning a manifest for every requested path.

Training, deployed-graph parity, exported-readout parity, bundle completeness,
target replay and practice scoring remain pending. Practice scores used: zero
of three. No win, loss, overall or component change is claimed. No scorer, lab
tooling, supplied manifest, evidence, test data or experiment card was edited.

## Resume after execution is restored

1. Inspect the allowed training labels and primary/auxiliary directory, then
   rerun the supplied six-shard test harness on the current files.
2. Export the diagnostic graph into this experiment's persistent scratch:
   `python3 modal/student_fading_memory_export.py --model <primary-dir> --out <extractor-dir>`.
3. Cache eligible recordings with
   `pnpm --filter @tilawa/sdk exec tsx src/bench/fading-memory-cli.ts --model <primary-dir> --extractor <extractor-dir> --labels <help-train-labels> --excluded-speakers <exclusions.json> --out <observations.jsonl>`.
4. Fit and freeze all three arms with
   `python3 modal/student_fading_memory.py --observations <observations.jsonl> --metadata <extractor-dir>/fading-memory-metadata.json --excluded-speakers <exclusions.json> --out <fit-dir>`.
5. Check training/runtime parity through the SDK CLI's
   `--verify <fit-dir>/readout-parity.json --fitted <fit-dir>/fading-memory-emissions.json`
   route. Package into a new directory through the exporter using
   `--bundle <bundle-dir> --fitted <fit-dir>/fading-memory-emissions.json`.
   Load and verify the complete bundle before declaring its relative scratch
   path in untracked `.autolab/model.json`.
6. Replay the four registered targets using the CLI's generic
   `--model <bundle-dir> --audio <wav> --evidence <evidence.json> --out <comparisons.json>`
   route. Count actual target warnings removed, not just favourable scores.
   Case selection stays outside engine code. Apply the task's three-target
   stopping rule; at least three wins requires practice-score before deciding.
