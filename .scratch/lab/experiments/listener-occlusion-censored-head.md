# Listener occlusion-censored phone-head refit

Status: implemented; real-data training, complete model export and incident measurement blocked by supervisor execution failures. No deployment model has been selected. The experiment card is unchanged.

## Prediction

Listener −1 and overall −1, from one false wrong-word warning removed with no predicted losses. The targets are:

- `lab-v1.5-p11/phone/warning:wrong_word/26:106:1/104.32`
- `lab-v1.5-p36/phone/warning:wrong_word/55:48:1/282.64`
- `lab-v1.5-p55/phone/warning:wrong_word/26:174:8/107.08`
- `realuser-v1-sob-049/phone/warning:wrong_word/92:13:1/36.44`

The task's stopping rule requires at least three targets won against the matched ordinary-refit control before practice scoring. This is stricter than the card's zero-target kill criterion. No target result is available, so neither criterion has been measured.

## Premise audit

The shipped `spliced-clean` training configuration in `docs/benchmarks/student-v2.json` enables SpecAugment and has teacher imitation weight 2. In `modal/student_train.py`, the augmentation randomly zeros full-frequency time intervals. The CTC loss still requires the complete spliced transcript, and the imitation loss still uses the unmasked cached teacher on every valid output frame. The valid-frame mask excludes padding, but does not exclude synthetic time masks.

Thus the static premise survives: the recipe can remove speech observations while continuing to supervise their phones. This does not establish how often masks actually intersect speech. The new prepare command measures transcript-constrained ordinary CTC occupancy on the unmasked copy and records masked cells with speech and consonant occupancy above 0.5. Fitting refuses a cache with zero masked supervised speech. That audit has not run on authorized recordings.

## Implementation

`modal/student_occlusion.py` retains the original augmentation's output and random draw order, optionally returning the union of the applied time masks. Frequency masks, naturally zero features, pauses, teacher disagreement and padded batch positions never create this provenance. `ClipDataset` and collation preserve it, including false mask values in padding. The existing full-model training losses retain their original behavior; censorship is applied in the bounded refit utility.

`modal/student_occlusion_head.py` implements three stages:

1. **Prepare:** admit complete help-train training-speaker takes explicitly labelled `none`, with no events, repeats, unsupported labels or exclusion flags. Exclude an entire speaker if any supplied label has a nontraining split. Optionally admit supplied original clean-match shards under `LAB_DATA`, with cluster provenance and whole-cluster exclusion for evaluation or gold-excluded recordings. Reserve whole training speakers before feature extraction. Cache paired unmasked and SpecAugment-masked features from the frozen deployed ONNX encoder, the full unmasked student posteriors, target and explicit masks, with checksums.
2. **Fit:** start both ordinary and censored linear heads from the exact shipped head. Use identical examples, deterministic minibatches and SGD update counts. Both copies retain full unmasked transcript and preservation imitation. The ordinary masked copy retains complete CTC and imitation. The censored copy replaces fully hidden output rows with constant normalized uniform emissions in the usual CTC forward sum and excludes imitation on intersected cells. Select one common update count by unmasked validation loss averaged equally across reserved training speakers; export both heads at that step.
3. **Export:** replace only the phone-head constants in the original float or int8 graph. Verify encoder node/tensor identity, finite outputs and exact streaming-state parity. Preserve the supplied manifest's inventory, timing, frontend and state contract. Copy and checksum the inherited SDK teacher graph and external weights; reject missing or corrupt declared files. Produce separate ordinary and censored bundles and an export report.

Output frame j's central input cell is `[4j+2, 4j+6)`. CTC censoring requires the whole cell to have explicit time-mask provenance; imitation exclusion requires any intersection. The mapping never extends beyond real input frames. This is a conservative central-cell rule, not a claim that the frozen encoder's full receptive field contains no information. Cached features include the shipped activation quantization before the integer phone head, and final padded output rows are trimmed using the SDK's output-count formula.

The censored CTC emission is `−log(V)` for every inventory class, including blank. All ordinary CTC stay/advance transitions and repeated-label constraints remain. This sums transcript progress through a deliberately hidden observation with no phone-head gradient there. It does not force a hidden frame to blank, discard the complete transcript, or enable wildcard emissions at inference. Normalization avoids a vocabulary-size multiplier per masked frame.

The frozen unmasked student supplies the preservation teacher in both arms, rather than running the original large training teacher. Original recordings use the shipped spliced target if their authorized shards are supplied; clean help takes use canonical targets with held-phone runs collapsed. No practice recordings enter training or checkpoint selection. The optional original pool has not been inventoried remotely; a help-only run would have narrower speech/reader coverage and would need to be reported as such.

The fixed recipe is seed 831, two paired masks per recording, 12 epochs, batch size 8, SGD learning rate 0.0001, preservation weight 2 and whole recordings up to 30 seconds. Original sampling, when supplied, is bounded separately within each speaker partition. None of these settings has been selected against practice results.

## Validation and limits

Python compilation and `git diff --check` pass. Nineteen synthetic unit tests cover exact augmentation/RNG parity, overlapping and zero-width masks, genuine zero observations, frequency masks, cell mapping, false padding provenance, exhaustive CTC path sums including repeated phones, absent hidden gradients, retained observed/pause gradients and paired unmasked supervision. Additional tiny-graph tests cover actual float/int8 feature capture, streaming padding/cache handling, head-only export, complete auxiliary assets, cache integrity, training-speaker isolation, clean-label authorization and matched optimizer/checkpoint selection.

All **19 tests passed locally in 36.306 seconds with PyTorch 2.8.0**, matching the shipped training image's version. These use generated features, graphs and labels only. They exercise both float and int8 feature capture, changed head outputs with unchanged encoder states, trimmed streaming padding, complete auxiliary copying/checksums and missing/corrupt-asset rejection. Exhaustive path enumeration agrees with censored CTC and the ordinary occupancy audit; hidden gradients vanish while the paired clean copy and ordinary pauses remain supervised. The two fitted heads differ and select the same validation-chosen step with identical minibatches and update counts. No learner audio or heavy training runs locally.

Supervisor requests 1 (data/model audit), 2 (six-shard core/SDK tests), 3 (audit retry) and 4 (six-shard test retry) all failed before container execution with `NotFoundError: No such file or directory`, zero reported spend and no container logs. Neither required core/SDK test attempt ran. The failures are infrastructure results, not evidence against the hypothesis.

No real paired cache, trained checkpoint, verified SDK model bundle, Node replay or matched target count is available. No practice-score invocation has been used (0 of 3). Overall, reader concentration, clean-session false warnings, planted-mistake losses and studio drift are unmeasured. Changed phone identities may still lose real mistakes despite the preservation anchor.

No `.autolab/model.json` is declared: running the shipped model without the new head would not evaluate this experiment. No killed outcome is written because the premise has not been disproved and target wins have not been measured. The implementation remains for review; completing the requested experiment requires functioning supervisor execution.

## Reproduction once supervisor execution works

Run the Python unit tests with torch, numpy, onnx, onnxruntime and soundfile installed:

```sh
PYTHONPATH=modal python3 -m unittest discover -s modal -p 'test_student_occlusion*.py' -v
```

Within the authorized supervisor container, use one persistent directory for all three stages (add `--original` only for supplied authorized original shards):

```sh
python3 modal/student_occlusion_head.py prepare --model "$LAB_DATA/models/spliced-clean-int8" --out /scratch/listener-occlusion-censored-head/refit
python3 modal/student_occlusion_head.py fit --out /scratch/listener-occlusion-censored-head/refit
python3 modal/student_occlusion_head.py export --model "$LAB_DATA/models/spliced-clean-int8" --out /scratch/listener-occlusion-censored-head/refit
```

The prepare stage also requires torchaudio and scipy, plus pyarrow for optional original shards. Verify both complete exports through the Node SDK, replay the targets against both matched arms, and apply the task's stopping rule before declaring a model or requesting practice scoring.
