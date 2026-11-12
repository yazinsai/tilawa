# listener-paired-gradient-subspace

Status: blocked before real-data training and evaluation; not a premise kill.

## Prediction and mechanism

The card predicts listener −1 and overall −1: independently degraded copies of
the same recording should produce some reproducible phone-head gradients, while
noise-specific directions should vary between copies. Restricting head updates
to reproducible directions might remove stable false consonant distinctions.
The encoder, alphabet, streaming clock, primary decoder, tracker, checker and
confirmation rules remain unchanged.

The card's diagnostic asks for two target windows to flip toward the complete
written pronunciation against both controls. The builder instructions require
three target wins before continuing to practice scoring. Neither criterion has
been evaluated. Infrastructure failures provide no evidence for or against the
premise.

## Implementation

`modal/student_paired_gradient.py` is an ordinary Python training utility for the
authorized CPU training sandbox. It has `cache`, `train` and `export` commands.
It reads the deployed streaming ONNX graph and exposes the dequantized input to
its final integer affine head. Retaining that input quantizer holds encoder
features fixed. Its effective float parent must reproduce the shipped head's
posteriors and greedy identities before caching is allowed.

Training eligibility is limited to original `s1` clean-match training records
with known speaker clusters and help-train training speakers' `none` takes with
no error or repeat annotations. Published gold exclusions apply to entire
speaker groups as well as individual records. Original targets use the existing
canonical-inner/teacher-edge splice; help targets use authorized canonical
transcripts. Scripted mistakes are never taught as canonical speech. Selection
uses a deterministic 80/20 speaker-grouped partition of eligible training data.
No practice recording is read by the training utility.

The fixed cache protocol draws two independent copies using the authorized
training noise and RIR banks, plus independent gain draws. It retains the
unaltered recording and adds 600 ms of silence at either end of every view.
The cache records parent and data identity, checks shard agreement, and refuses
reuse under a changed parent or protocol. The bounded initial corpus is at most
384 original recordings, at most 12 per original speaker, plus eligible help
takes. Cache jobs have a 23-hour elapsed guard; the proposed 20 CPU shards and
one refit request stay within the one-day compute allocation if each finishes
within the supervisor's one-hour limit.

At the parent, the utility evaluates one transcript CTC gradient per degraded
view per training recording. It centers gradients across recordings, estimates
symmetrized cross-view covariance and half the covariance of view differences,
and solves their regularized generalized eigenproblem in a common randomized
parameter range of at most 128 dimensions. It retains up to 16 directions whose
signal/noise ratio exceeds one. The shuffled control uses a one-to-one recording
derangement, the same parameter range and gradient evaluations, and exactly the
paired rank. Both bases are Euclidean-orthonormal.

The ordinary, paired and shuffled arms use identical data, three-view objectives,
batch order, learning rate, clipping and update counts. The objective averages
frame-normalized transcript CTC and preservation KL to the parent across all
three views (clean KL weight 1; degraded weights 0.5). Plain SGD projects the
entire update, avoiding adaptive optimizer steps outside the chosen subspace.
Each arm independently selects its checkpoint from 12 epochs by the
speaker-averaged objective on held-out **training** speakers. Rank selection,
checkpoint choice and protocol are frozen before any target diagnostic.

Export replaces the affine head in the original graph, preserves every encoder
and cache node, copies the parent's auxiliary bundle, checks external tensor
presence, and verifies selected-head output and exact cache parity across three
carried-state calls. It writes the new primary hash and manifest only after those
checks. This is compatible with the existing SDK model loader and needs no new
runtime warning rule.

## Validation and current limits

Five numerical unit tests passed locally: reproducible signal versus independent
noise, identical and anticorrelated views, non-amplifying/idempotent projection,
recording derangement, grouped/order-independent speaker selection, and invalid
gradient rejection. A synthetic quantized ONNX export test also passed, checking
effective-parent parity, export output names, retained alphabet and clock,
auxiliary copying, model hash, and refusal to overwrite the parent. These are
small synthetic checks, not acoustic or practice results.

A full synthetic-cache refit integration test passed, checking matched update
counts, training/validation speaker separation, unchanged parent files and
projected checkpoint updates. The final local run passed all seven tests in
17.3 seconds (NumPy 2.5.4, PyTorch 2.15.0 CPU, ONNX 1.23.2 and ONNX Runtime
1.31.1). This is additional implementation validation; the supervisor's pinned
training environment and production graph remain untested.

The authorized supervisor returned `NotFoundError: No such file or directory`
before sandbox logs were created for all of these CPU requests:

- Run 1: practice-model/help-train inventory (default image, 10 minutes).
- Run 3: training-data/model inventory (train image, 10 minutes).
- Run 4: Python training unit tests (train image, 10 minutes).
- Run 5: prescribed core/SDK suites, six shards (default image, 10 minutes).

Run 2's GPU inventory was rejected explicitly: GPUs are reserved for bet cards;
this is an explore card. Subsequent requests used CPU only. Error records are in
the ignored `.autolab/modal-result-*.json` files. The runner recovery question
was sent to the user while independent local validation continued.

Consequently the current deployed graph's head topology, authorized corpus
availability, actual gradient spectrum, real-data checkpoint selection, complete
production export, target acoustic comparisons and engine warnings remain
unverified. Core/SDK tests were requested but did not execute. No practice score
was requested (0 of 3 used). No target has been evaluated; there is no measured
target-win count or overall delta.

There is no `.autolab/model.json` and no production model bundle. Scoring the
main listener would not evaluate this implementation. Do not interpret this as
a zero-gain experiment or a killed hypothesis. The committed utility and tests
are preparatory implementation, awaiting authorized runner recovery.

## Resume after runner recovery

First repeat the CPU inventory and verify the parent bundle matches the card's
listener, including its existing auxiliary teacher files. Then run:

```sh
python3 /Users/rock/ai/projects/tilawa/tools/autolab.py modal-run --image train --shards 20 --timeout 60 -- 'python3 modal/student_paired_gradient.py cache --model /lab/data/models/spliced-clean-int8 --output /scratch/listener-paired-gradient-subspace/cache'
python3 /Users/rock/ai/projects/tilawa/tools/autolab.py modal-run --image train --timeout 60 -- 'python3 modal/student_paired_gradient.py train --cache /scratch/listener-paired-gradient-subspace/cache --output /scratch/listener-paired-gradient-subspace/heads'
python3 /Users/rock/ai/projects/tilawa/tools/autolab.py modal-run --image train --timeout 60 -- 'python3 modal/student_paired_gradient.py export --model /lab/data/models/spliced-clean-int8 --head /scratch/listener-paired-gradient-subspace/heads/paired.npz --output /scratch/listener-paired-gradient-subspace/paired'
```

Copy selection/spectrum reports to `LAB_OUT` for review. Only after verifying the
complete bundle, SDK loading and unchanged timing/caches, declare ignored
`.autolab/model.json` with `{"path":"paired"}`. Freeze training selection, then
compare complete written/rival pronunciation evidence on the three card windows
against the ordinary and shuffled selected heads. Use the engine's actual
request windows and confirmations, rather than choosing acoustic crop boundaries
to obtain wins. If fewer than three targets win, revert engine changes and
record a measured premise kill. If three win, run practice scoring before any
decision about losses, as required by the builder instructions. Rerun the
prescribed core/SDK suites when the runner can execute them.
