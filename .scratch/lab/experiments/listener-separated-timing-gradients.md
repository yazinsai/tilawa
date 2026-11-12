# listener-separated-timing-gradients

Status: implementation prepared; supervisor infrastructure blocks runtime tests,
training, export and target validation. No evaluated model is declared. The
experiment card is unchanged.

## Prediction and premise

Predict two recovered additions and one collateral wrong call: listener −1 and
overall −1. The speculative premise is interference between blank/emission timing
gradients and features that distinguish weak consonants. The registered phone
targets are p07 at 144.12 s, p33 at 156.16 s, p48 at 39.98 s, p56 at 106.76 s,
and p57 at 95.26 s and 112.26 s.

The card requires at least two previously missing planted consonants to recover
in correct order in full-session exported decoding relative to the matched
shared control. The user's later instruction additionally requires at least
three target incident wins before practice scoring. Acoustic consonant recovery
and confirmed warning wins must be reported separately. Neither has been
measured. Infrastructure failure does not establish the kill premise.

## Readout and control

`modal/student_timing_head.py` freezes both the encoder observations and the
shipped classifier. Two independent 64-unit tanh residual branches predict a
timing scalar `t` and conditional nonblank identity residuals `i[c]`. Given the
shipped normalized log probabilities `b`:

```text
o = logsumexp(b[nonblank]) - b[blank] + t
log p[blank] = log sigmoid(-o)
log p[c] = log sigmoid(o) + log_softmax(b[nonblank] + i)[c]
```

This is an ordinary normalized CTC distribution. Transcript CTC trains both
branches through that distribution, but blank/emission gradients cannot change
the identity branch's trainable feature projection. Identity residuals cannot
change speech mass. Encoder features, baseline probabilities and normalization
statistics are detached observations. The inference path retains the existing
alphabet, blank ID, 40 ms clock and decoder; there is no extra acceptance rule.

The matched control deep-copies every trainable tensor from the separated arm.
Both residual outputs start at zero, giving identical initial shipped
probabilities. The shared arm applies a fixed orthogonal DCT rotation to the
concatenated nonlinear features, then reads the timing and identity groups.
Every hidden feature thereby participates in both readouts' gradients. The
rotation occurs after tanh; rotating before it would merely reparameterize
independent branches. Both arms have exactly the same trainable parameters,
with no unused padding parameters: 35,691 for D=256 and V=43. The actual source
width remains to be checked on the supplied bundle.

This control uses fixed mixing to share features while retaining identical
parameter tensors and counts. It is not an unconstrained dense output layer:
the fixed readout routing is part of the comparison, and capacity/routing
effects cannot be separated completely from gradient interference. The shared
arm also incurs a small fixed matrix multiply absent from the separated export.

## Training and export

`modal/student_timing_train.py` admits complete help-train none-labelled takes
from consistently train-only speakers. It rejects scripted mistakes, repeats,
events, unsupported/excluded takes and speakers appearing in another split.
The supplied help-train excludes release-held-out speakers by the user's data
contract; an optional additional exclusion list is also applied. Independently
transcribed additional recordings require explicit authorization, verification,
train split and an audio SHA-256. No mistake script becomes a performed
transcript. No practice recording enters preparation or model selection.

Full recordings retain their ordinary speech and natural pauses. Original views
are always cached; if authorized shipped noise/RIR banks are supplied, two views
from the existing augmentation recipe are also cached once for both arms.
The requested preparation uses original views only. Source model, auxiliary
assets, audio, transcripts, augmentation banks and cached views have recorded
identities. Duplicate audio and impossible complete CTC transcripts are rejected.
Normalization uses only the training side of each speaker partition.

Three stable speaker-grouped training partitions evaluate checkpoints at epochs
6, 12 and 18. A single epoch is selected jointly for both arms by training-held-out
speaker macro phone error plus shipped-output KL, with the earlier epoch breaking
ties. Both final arms then train on all admitted speakers for that same epoch
count. Every recording/view, loss, initialization, optimizer configuration,
shuffle order and update count is matched. The fixed objective is full-recording
CTC plus KL preservation with weight 1; AdamW uses learning rate 0.001, decay
0.01 and gradient norm clipping at 5. There is no practice-fitted constant or
checkpoint choice. Preparation and refitting share a one-day wall-clock budget.
Small-head CTC uses deterministic CPU backward on the supplied Modal training
image; the Mac performs no training.

`modal/student_timing_export.py` exposes native encoder features and grafts the
readout into a copy of the shipped int8 graph. Encoder/cache computations and
the SDK interface are inherited. It requires the unchanged SDK insertion teacher
and all external weights, copies bundle files and verifies their hashes.
New output manifests describe the new graphs; supplied manifests are never
rewritten. Source/output separation and empty output directories are enforced.
Prepared extraction checks all original model/cache outputs for exact parity.
Final validation streams every cached full recording/view through each exported
primary graph, checking finite normalized probabilities and parity with the
trained readout. These runtime checks are implemented but unexecuted.

No tracker, checker, SDK verification, scorer, lab tooling, test data, supplied
manifest or evidence file has an edit.

## Validation and limits

All four new Python files pass `py_compile`. A dependency-free check executes
the actual source functions for speaker grouping, none-take admission and CLI
parsing: deterministic grouped splits pass; too few speakers, mistakes,
repeats, events and mixed-split speakers are excluded as intended. The CLI uses
`--help-labels` to avoid colliding with argparse's built-in `--help`.

A lightweight numerical check executes the actual `phone_log_probs` function
with NumPy operator shims in float64. It passes baseline initialization,
timing/conditional-identity independence, identity/blank independence and finite
normalization at timing residuals ±1000. The maximum normalization error is
4.44e-16. This is not a Torch autograd, CTC, ONNX or SDK test. The installed local
NumPy is x86-only, so that small check ran with `arch -x86_64 python3`; no training
or production acoustic inference ran locally. Ignored audit artifacts are under
`.autolab/timing-*.json` and `.autolab/timing-math-check.py`.

Added 14 Python unit tests covering exact parameter/tensor initialization,
gradient isolation versus sharing, CTC gradients into both branches, frozen
observations, extreme timing probabilities, admission, matched optimizer updates,
CLI commands, ONNX/Torch parity, unchanged state outputs, stream draining, model
identity and required auxiliary assets. None has executed on Torch/ONNX.

Supervisor requests:

| Run | Requested work | Result |
| --- | --- | --- |
| 1 | Read-only supplied inputs and runtime probe | `NotFoundError: No such file or directory`, before execution |
| 2 | Supplied six-shard core/SDK test harness | Same infrastructure error, before execution |
| 3 | Training-image Python tests and input probe | Same infrastructure error, before execution |
| 4 | Training-image preparation and fitting | Same infrastructure error, before execution |
| 5 | Current Python tests, preparation and fitting | Same infrastructure error, before execution |

No training input was read, no checkpoint was fitted, and no model bundle was
exported or validated. Targets evaluated: none. Recoveries, warning wins,
listener/overall changes, fully clean-session warnings and studio regressions
are unmeasured. Practice scores requested: 0 of 3. No `.autolab/model.json`,
killed outcome or fold is recorded. Running the main listener would not evaluate
this change.

When the authorized supervisor works, run the new Python tests and required
core/SDK harness, prepare admitted training inputs, fit both arms, and verify
their complete exports under `/scratch/listener-separated-timing-gradients/`.
Use `--help-labels "$LAB_DATA/help-train/labels.jsonl"` for preparation. Compare
both exported models through full-session SDK decoding of the registered
targets, including ordinary insertion verification. If the acoustic premise
fails or fewer than three target incidents are won, revert the engine/test
changes and record the killed result. If both requirements survive, declare
only the complete verified separated bundle in the untracked model selection
and request the authorized practice score before deciding. Preservation loss
does not establish safety, and unexecuted checks must not be reported as passing.
