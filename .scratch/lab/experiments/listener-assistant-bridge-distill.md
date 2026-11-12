# Assistant bridge distillation

Status: training implementation present; compute environment blocked; unmeasured. This is not an acoustic kill.

## Prediction and mechanism

The card predicts two recovered additions and one collateral wrong call: listener −1, overall −1. The premise is that direct compression from the roughly 600M-parameter Muaalem teacher to the shipped roughly 70M `full` student loses weak consonant distinctions. An intermediate-capacity assistant could provide a distribution the compact student can learn more faithfully.

`modal/student_bridge_train.py` implements separate preparation, assistant training, frozen-target generation, bridge-student training, direct-control training and export stages. The assistant has three times the shipped feed-forward widths, while retaining encoder dimensions, layers, causal geometry and phone-head dimensions. It starts from compatible shipped weights with widened projections initialized afresh, then learns from the original teacher. The code checks that its actual parameter count lies between the student and original teacher. Both final student arms start from exactly the same shipped checkpoint and retain its complete architecture.

The original teacher's S1 top-five targets are shared by both arms; help-none targets are generated once with the unchanged exported Muaalem teacher and its SDK frontend. The frozen assistant generates corresponding top-five targets on the same unaugmented recordings. Student training keeps the shipped RIR, noise, gain and SpecAugment recipe, with recording/visit seeds that reproduce draws independently of supervisory arm. This retains the shipped denoising-distillation convention: supervisory targets come from original recordings rather than independently augmented observations.

All three training stages retain transcript CTC, supervisory cross-entropy and an ordinary-reading KL preservation term against the frozen shipped student. Transcript runs use the shipped spliced-target collapse convention. No synthetic speech, practice audio or planted mistakes enter training.

The direct control receives the bridge arm's total measured CUDA-event budget, including assistant training, checkpoint selection, frozen-target generation and bridge training/selection. It can make additional passes over the identical corpus and deterministic augmentation stream; update counts therefore need not match. Both student arms share optimizer, schedule, loss weights and data recipe. Export refuses a control that underspends or exceeds that budget by more than its last indivisible optimizer update. This matches measured device time rather than exact FLOPs; hardware changes between resumed stages would undermine the comparison and must be avoided. Original-teacher target preparation is shared work, incurred once for both arms.

## Training exclusions and selection

The loader retains S1 speaker clusters and derives exclusions before filtering recordings by bucket, duration or material limit. Any group represented in a non-training split or by a gold-excluded recording is denied entirely. The existing release manifest supplies additional speaker denials even when its particular recording is absent from S1; its labels and audio never enter training or selection. Unknown speaker identities cause an error.

Only authorized S1 `clean-match` transcripts and help-train scored `none` takes with no mistake, repeat, unsupported or exclusion annotations provide supervision. A planted script is not a performed transcript. Remaining training speaker groups are divided deterministically into fit and selection partitions. Selection uses only those training speakers, without calling the existing trainer's test-set evaluation. S1 identities are the repository's inferred speaker clusters, not verified identities across sources.

Recipe, starting checkpoint, corpus and augmentation banks are checksummed. Optimizer, schedule, sampler position and random states are saved for resumption. Frozen targets carry the selected assistant checkpoint hash and must be complete before bridge training. Nonfinite updates stop with an error rather than silently skipping a recording.

## Compact bundle

Export uses the existing streaming graph and MatMul-only dynamic int8 quantization. It checks inventory, blank, feature settings, window, shift, 40 ms clock and recurrent-state interface against the shipped bundle. Training-only selection clips validate fp32 ONNX against real recurrent PyTorch replay, then compare int8 greedy decodes with fp32. Both compact bundles copy and verify every insertion-teacher file declared by the SDK, including external tensors. Completed exports are checksummed and verified on retry; partial exports are never declared as the scored model.

There is no tracker, checker, SDK warning-rule, insertion-verification, scorer, lab-tooling, card, test-data, supplied-manifest or evidence change. `student_train.build_model` and `load_student` accept checkpoint-specific architecture configuration without changing existing presets or defaults.

## Validation and execution blocker

Local validation: 13 standard-library policy tests pass; four tensor/model tests are skipped because PyTorch is absent locally. Tests cover complete-speaker exclusion, training-only disjoint selection, release groups whose held-out recordings are absent, unknown identities, script/repeat rejection, reproducible augmentation seeds, complete compute accounting, padding gradients, frozen-supervisor gradients and unchanged model geometry. Python compilation and whitespace checks pass. CLI help runs without loading training dependencies.

Used only the authorized supervisor interface. Requests 1–6 failed before producing a shard directory or command log, with `NotFoundError: No such file or directory`:

| Request | Environment | Purpose |
| --- | --- | --- |
| 1 | train, CPU | Checkpoints, teacher assets, S1 schema and help labels |
| 2 | standard, CPU | Minimal path/model-directory probe |
| 3 | train, CPU | Minimal `/student` probe |
| 4 | standard, CPU | Directory preflight retry |
| 5 | train, CPU | List authorized checkpoints and data directories |
| 6 | standard, six CPU shards | Required core/SDK tests via `tools/autolab_tests.sh` |

The first request was submitted at 2026-11-12 06:38:16 UTC. Requests 4–6 were submitted after resuming the prior attempt. Their errors do not identify the missing resource, and no output establishes that a command started. All six records report $0 spent and the full $150 GPU allowance remaining.

Requests 7 and 8, submitted at 07:07:37 UTC, also failed with the same `NotFoundError` and returned no shard directory or command log. Request 7 attempted the implemented preparation command on H100; request 8 attempted the new Python suite in the train image. Their request files contain the exact commands. The H100 request incurred $0.2205 despite returning no usable execution output; its final record reports $149.78 remaining. The CPU test request incurred $0. Its earlier budget snapshot still included the H100 reservation and is not the final available balance.

No assistant or control has been trained. No compact bundle exists, and no `.autolab/model.json` override is declared. Running the default main listener would not evaluate this training change. Practice scores used: 0 of 3. Ordered-consonant recoveries, target warning wins, clean-session behavior, studio behavior and overall changes are unknown, not zero. Neither the card's fewer-than-two ordered-consonant criterion nor the user's fewer-than-three warning-win rule has been evaluated. There is no basis for a killed outcome or fold decision.

## Resume

Restore the supervisor's missing resource and rerun train-image tests and preparation preflight. Confirm the available shipped checkpoint path rather than inferring it from a directory name. Use the stage commands in the module's docstring on the same GPU type, resuming each until complete; then export and verify both bundles through the SDK. Only after the complete bridge bundle exists should `.autolab/model.json` point at its directory relative to experiment scratch.

Freeze both training-selected checkpoints before inspecting practice targets. Compare the six card targets against the matched direct control's full-session exported decode, checking previously missing planted consonants in correct order, then compare warning wins through the unchanged SDK pipeline. Apply the acoustic/target stop rules to those measurements. If at least three targets are won, run the authorized practice score before deciding. Infrastructure failure is not zero recoveries or an acoustic kill.
