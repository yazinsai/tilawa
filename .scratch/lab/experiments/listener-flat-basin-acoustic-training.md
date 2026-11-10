# Flat-basin acoustic training

Status: blocked on supervisor execution. The training implementation and small-tensor
tests are complete; no model has been trained, exported or declared for evaluation.
The experiment card is unchanged.

## Prediction and mechanism

The card predicts two false warnings removed and one caught mistake lost:
listener −1 and overall −1. The proposed cause is sensitivity to brittle acoustic
weight configurations, rather than an inference-time scoring or warning-rule defect.

`modal/student_sam.py` computes the loss gradient at the original weights and
proposes an adverse perturbation independently for each parameter tensor. Its RMS
is bounded by the selected radius times that tensor's weight RMS, with a 0.001 RMS
floor for zero or tiny tensors. The direction maximizes the linearized loss within
those layer balls. If the actual training loss falls, a bounded backtrack reduces
the radius, with an unperturbed fallback. The original weights are restored exactly
before AdamW applies the gradient from the adverse loss. Torch CPU/CUDA, Python
and NumPy random draws are replayed, and forward buffers retain one pass's changes.
Nonfinite losses or gradients clear gradients and restore the perturbed weights.
The exported graph contains one primary model, with no perturbation or branch.

`modal/student_flat_basin.py` implements the matched training experiment:

- Both arms start from the exact supplied `spliced-clean` checkpoint. A shared,
  hashed plan fixes data, batches, augmentation, update count and selection inputs.
- The ordinary arm uses radius zero. SAM candidates use radii 0.001, 0.003 and
  0.01. All arms use AdamW at 0.00001, with no weight decay, for 1,000 updates.
  This is a matched fine-tuning optimizer, not restoration of the pretraining
  optimizer's accumulated state.
- Each update takes four original clean-match recordings, one none-labelled
  help-train take and one preservation-only help-train recording. Original CTC
  targets and KD weights come from the shipped spliced-target recipe. RIR, noise,
  gain and SpecAugment follow the shipped training code and configuration.
- All recordings retain a full-frame KL anchor to the frozen shipped model on
  identical augmented inputs, weighted by four. Mistake-labelled recordings receive
  only preservation loss. Their scripts never supply a performed transcript;
  an objective-level assertion rejects accidental transcript targets for them.
- Only explicit training splits are eligible. Whole non-training and gold tlog
  speaker clusters are removed. Any help speaker with a non-training split is
  removed. A deterministic fifth of each source's remaining training groups is
  reserved for checkpoint and perturbation-scale selection, with no speaker group
  crossing the internal boundary. Every required role must exist on both sides.
- The best checkpoint at updates 250, 500, 750 or 1,000 minimizes the matched
  role-weighted loss on those reserved training speakers, averaged over native
  audio and a deterministic shipped-augmentation view. The best SAM scale and
  the best ordinary checkpoint are frozen before target replay.
- Export verifies the selected checkpoints' hashes, streams float and int8 graphs
  on reserved none-labelled training takes, and rejects changed inventory, frame
  geometry, state layout or supported classes. Both candidate and control use
  shipped dynamic MatMul int8 quantization. Existing declared SDK teacher assets
  must be present and hash-verified in the supplied source bundle and are copied
  unchanged; missing inherited assets block export.

No tracker, checker, inference-time warning rule, registry manifest, scoring code,
supplied test data or evidence was changed. Generated model manifests belong only
to a completed export in persistent scratch storage.

## Validation and limits

Four supervisor requests failed before command execution. Every result reports
`NotFoundError: No such file or directory`, `$0` spent and `$160` remaining:

| Request | Intended validation | Result |
| --- | --- | --- |
| 1 | Training-image data/checkpoint inventory | Infrastructure error |
| 2 | Training-image SAM unit tests | Infrastructure error |
| 3 | Supplied six-shard core/SDK build and test harness, standard image | Infrastructure error |
| 4 | Completed training utility's unit tests and authorized-data audit | Infrastructure error |

After these failures, 14 small-tensor/unit tests passed locally with torch 2.8.0
on CPU. No training recordings were fetched or processed on the Mac. The tests
cover layer-relative radius, gradient descent against the adverse loss, actual-loss
backtracking, exact ordinary-optimizer equivalence at radius zero, stochastic
draw replay, single-pass buffer updates, rollback on a nonfinite second pass,
training-speaker reservation, explicit clean-label policy, deterministic resumed
batches, export geometry rejection, output preservation on deviant phones and
rejection of script-derived transcript targets. Python compilation also passed.

The full core/SDK suite did **not** run, despite the submitted required command.
Actual Zipformer training, dataset coverage and provenance, checkpoint availability,
GPU behavior, ONNX export, inherited teacher availability and complete session
replay remain unverified. Tlog speaker identity is an approximate acoustic cluster;
the implementation cannot independently establish person identity across sources.
Preservation losses and training-only selection do not guarantee safety on unseen
audio. Float/int8 drift is reported after selection, not used to select another
checkpoint against practice behavior.

Practice scores used: **0 of 3**. Target warnings removed relative to the matched
ordinary continuation: **unmeasured**. No zero-win result or premise failure is
claimed. Neither the card's fewer-than-two matched-control criterion nor the
task's stricter three-target continuation minimum can be evaluated yet. The
experiment is blocked, not killed, and no fold or discard was performed.

## Remaining execution

Once supervisor execution works, run the audit, all four matched continuations,
training-only selection and export through this utility. Resuming an interrupted
arm repeats the exact same command; it checks the shared plan and restores its
optimizer. The utility saves before the supervisor's one-hour job limit.

The complete exported bundles must be verified before writing an untracked
`.autolab/model.json` pointing to `flat-basin/candidate-int8`. No model selection
file exists now: evaluating the main model would not measure this change.
Then replay all four registered **complete sessions** with both frozen int8
continuations, matching warnings at the original occurrences. Neither warm-crop
spellings nor training losses are evidence of target recovery. If fewer than three
registered warnings are removed, follow the task's rollback/killed-result procedure.
If at least three are removed, run the required practice score before deciding;
the fold gate handles aggregate gains and losses. Check the fully clean-session
hard stop, with no post-replay adjustment of scales or checkpoints.
