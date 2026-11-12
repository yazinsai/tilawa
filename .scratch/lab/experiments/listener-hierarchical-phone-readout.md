# listener-hierarchical-phone-readout

Status: source implementation prepared; acoustic evaluation blocked by the
supplied supervisor. No trained primary model is declared.

## Prediction and premise

The unchanged card predicts one recovered addition, listener −1 and overall −1.
The six targets are p07 at 144.12 s, p33 at 156.16 s, p48 at 39.98 s,
p56 at 106.76 s, and p57 at 95.26 s and 112.26 s. The proposed defect is
weak consonants losing to unrelated phone classes in a flat classifier.

The card requires at least two spans to regain a previously missing planted
consonant in order relative to an equal-parameter flat nonlinear readout.
The user's subsequent instruction requires at least three actual target wins
before practice scoring. Consonant recovery alone does not establish a caught
addition: the unchanged listener verification and engine must confirm it.
Neither threshold has been acoustically evaluated.

## Implementation

`modal/student_hierarchical_tree.py` derives a fixed tree from generic manner,
vowel quality and inventory-mark categories. Its 15 internal nodes distinguish
blank versus nonblank, broad families, then individual inventory leaves.
Obstruents split into closures/affricates and frication; sonorants into nasals,
liquids and glides; vowels into low, front and back groups. Ghunna remains in
the nasal family, and length/release and pause marks retain separate leaves.
The tree covers all 43 classes in their original output order. Its coarse
closure/frication grouping is a modeling assumption, including contextually
variable realizations of ج and ض, rather than measured acoustic truth.

`modal/student_hierarchical_head.py` gives each node its own biased acoustic
projection, tanh, and bias-free branch projection. For frozen encoder feature
`x`, training-only normalization `z`, and shipped leaf log probabilities `q`:

```text
prior[n,j] = logsumexp(q[c] for c in child[n,j])
conditional[n] = log_softmax(prior[n] + B[n] tanh(A[n] z + b[n]))
log p[c] = sum(conditional[n,j] for (n,j) in path[c])
```

Thus leaf probabilities multiply conditional probabilities along unique paths
and form one normalized ordinary CTC distribution. The shipped encoder and
linear classifier remain frozen; the latter supplies conditional priors.
Every residual branch output starts at zero, reproducing shipped probabilities.
This initialization preserves speech and pause outputs before fitting. It is
not a preservation guarantee after fitting.

All nodes have 25 hidden units for the shipped 256-channel feature width.
The flat control adds a conventional 326-unit tanh residual to the same frozen
shipped distribution and performs one flat log-softmax. Each arm has exactly
97,800 active trainable parameters, 391,200 fp32 parameter bytes. Widths are
derived from inventory and encoder dimensions, with no unused padding weights.
Nonlinear conditional odds have nonzero acoustic mixed derivatives and are
independent of a different family's leaf discriminator when routing is fixed.
This avoids the linear-softmax cancellation of the earlier coded-head audit.

`modal/student_hierarchical_train.py` requires both authorized, verified
performed-transcript recordings and help-train training speakers' complete
none-labelled takes. It rejects help mistakes, repeats, exclusions and unscored
takes; excludes any speaker appearing in a nontraining help split and the
explicit supplied release-speaker exclusion list. Additional recordings require
train split, authorization, transcript verification and an audio SHA-256.
Duplicate audio is rejected before caching. Scripted mistake labels never
become performed transcripts.

Preparation caches original plus two deterministic shipped-augmentation views
of each full recording, including its pauses, frozen encoder features and
shipped probabilities. Noise/RIR banks and cached views are hashed. Both arms
receive identical views, recording order, updates and CTC plus KL preservation.
Three speaker-grouped training partitions select preservation from 0.1, 1 and
10 using equal-speaker phone error plus KL, with stronger preservation breaking
ties. Feature normalization is fitted only to each fold's training side.
The schedule is fixed at 18 epochs; no practice input is accepted by preparation
or selection. Final refits use all admitted training speakers.

`modal/student_hierarchical_export.py` reuses the prior interacting-head
experiment's frozen-feature extraction and bundle plumbing, replacing its
readout with explicit conditional log-softmax and root-to-leaf summation.
The ordinary SDK directory loader consumes the resulting primary graph without
SDK, tracker, checker, timing or insertion-rule edits. Export requires and
checksum-verifies the unchanged insertion teacher and external data. It copies
auxiliary bundle members, preserves cache outputs and creates a new hashed
manifest without rewriting the source. Fitting checks full-recording exported
probabilities against the cached-feature Torch implementation before producing
its completed report.

## Validation

The dependency-free tree suite passes four tests: complete inventory paths,
generic families, probability conservation for independently sampled branch
distributions, inventory-order invariance and exact parameter matching.
All six Python modules compile, whitespace checks pass, and the actual
`prepare --help` command executes successfully with the required data inputs.

All **15 local unit tests pass** with Torch 2.8.0, ONNX 1.19.0 and ONNX Runtime
1.22.1 on the Mac CPU. Torch/ONNX tests cover initial preservation, nonlinear conditional
independence, CTC gradients through every node, frozen observations, extreme
score normalization, speaker admission, primary graph parity, unchanged caches,
stream draining, source-bundle immutability and missing auxiliary rejection.
Both fp32 and dynamically quantized stateful toy encoders exercise actual
exported graph inference, with nonzero readouts and optimization enabled.
The quantized-source test caught and fixed copying an older adjacent
`model.onnx` over the new primary when the source's selected filename differed.
The exporter now excludes that conflicting primary filename from auxiliary
copying. These synthetic tests do not establish production acoustic recovery.

The supplied supervisor failed before executing each requested command:

| Request | Work | Result |
| --- | --- | --- |
| 1 | Read-only runtime, supplied-data and storage probe | `NotFoundError: No such file or directory` |
| 2 | Required core/SDK build and tests, six shards | Same error |
| 3 | Training-image data/dependency probe and new Python tests | Same error |

All three result records report zero compute spend. No alternate route to
Modal or release data was attempted. Lightweight local unit tests use an
ignored environment inside this worktree; no encoder extraction or recording
training runs on the Mac.

## Limits and remaining work

No training recording has been loaded, no head has been fitted, and no complete
trained bundle exists. Authorized transcript/augmentation/exclusion paths and
admitted speaker counts remain unestablished. The production int8 feature
tensor and actual SDK full-session behavior have not been exercised.
Targets evaluated: **0 of 6**. Recoveries, listener/overall deltas, clean-session
warnings and studio drift are **unmeasured**, not measured zero. Practice scores
used: **0 of 3**. Core/SDK test results are unavailable.

The supervisor must not score this source change using the main listener as a
substitute. No `.autolab/model.json`, killed outcome or fold is recorded.
Infrastructure failure does not establish a negative acoustic premise, so the
source is retained as unvalidated work.

After supervised execution is restored, establish authorized training inputs,
prepare identical frozen observations, select and fit both arms exclusively
within training partitions, and verify both complete bundles under the
experiment's scratch directory. Freeze selection before paired full-session
target replay. If the acoustic or actual-win criterion fails, revert the engine
implementation and record the killed result. With at least three actual target
wins, declare the verified hierarchical bundle and run the authorized practice
score before deciding. Incorrect family routing and new false warnings remain
material risks; output-preservation KL cannot establish their absence.
