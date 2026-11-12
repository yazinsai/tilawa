# Integrated acoustic phone emissions

Status: source implementation prepared; blocked before training and exported-stream validation. This is not a measured kill or a result ready for folding.

## Prediction

The card predicts two recovered added-word incidents and one collateral wrong call: listener −1, overall −1. The premise is that weak consonant evidence lacks a frame-local CTC peak but can contribute to an acoustic phone event when integrated over time. The registered six targets and the card are unchanged.

## Mechanism implemented

`modal/student_integrate.py` attaches a learned nonnegative mass head and a separate affine phone classifier to the streaming acoustic encoder. Frame mass is clamped to [0, 1]. Consecutive contributions fill unit event bins; weighted acoustic vectors are classified without autoregressive text history. Exact zero mass leaves silent frames out of the contribution list. A contribution crossing a unit boundary is split between its two events.

Training uses ordered phone cross-entropy after pooling, absolute quantity error, and the jointly trained original framewise CTC head. Only the training identity loss rescales mass to the transcript count. Free decoding uses raw mass and the ordinary half-unit terminal-tail rule, without a transcript or target-count input.

The matched CTC arm has the same encoder, heads, parameter count, batch tensors, augmentation draws, dropout draws, optimizer, and update count. Both arms compute both identity objectives and pooling; the inactive identity objective has zero weight. The control trains the new identity head with framewise CTC rather than pooled identity cross-entropy. A shared checkpoint step is selected by the mean of the two training-speaker development PERs, ensuring identical update counts in the exported pair.

`modal/student_integrate_train.py` accepts only clean-match Quran recordings from the training split and help-train speakers' none-labelled, unexcluded takes without mistakes or repeats. Quran clusters present in non-training metadata or the gold exclusion list are excluded in full. Development speakers are selected deterministically from the remaining training speakers. Full recordings retain their pauses, and explicit no-speech examples supervise zero count. The shipped RIR, noise, gain, fbank and SpecAugment recipe is retained. No practice audio or incident labels enter training or checkpoint selection.

`modal/student_integrate_export.py` exports raw mass, affine phone logits, the auxiliary CTC distribution, and the encoder states. It requires a fresh output directory and `--base-bundle`, copies installed auxiliary assets from that base bundle with SHA-256 verification, and rejects an insertion teacher missing its external weights. It verifies streaming numerical parity and event-token equality before writing the generated model manifest. Accumulating affine logits with unit weights is algebraically equivalent to classifying the pooled acoustic vector; their softmax is taken only after firing. Both arms retain the base bundle's installed insertion verification.

The SDK uses those events as primary tokens when the exported model declares its CIF decoder. A token is timestamped at its firing frame so the existing evidence transport can deliver it with a contiguous posterior chunk. Every original contribution keeps its half-open sample interval and fractional mass in `integration` metadata, on the model sample-rate clock. A possible terminal tail retains one posterior row and is delivered at end of audio; its original source intervals remain intact even after trailing silence. Context-free crops shift contribution sample intervals back to the original audio clock.

All CTC classes are retained for CIF models because a pooled identity need not appear among CTC's top five. These are the independent CTC probabilities, with the existing rounding, never pooled probabilities copied across frames. Tracker and checker code and decision rules are unchanged.

## Validation

The core and SDK TypeScript builds passed. The listener and integration behavior tests passed (11 tests in the final SDK suite), including pooling before identity selection, split mass conservation, repeated equal phones, zero-mass silence, terminal rounding, malformed outputs, chunk-size invariance, independent CTC evidence, original crop contribution intervals, and terminal delivery after trailing silence.

The full local suites initially encountered worker contention and timeout failures. With one worker, the full SDK suite passed (32 tests, one existing help-e2e skip), including preservation of source samples when a crop is rebased from a different sample rate. All 766 core assertions across 54 files passed with one worker and a 60-second default test timeout. Vitest still returned exit 1 for its known `Timeout calling "onTaskUpdate"` worker RPC error; `tools/autolab_tests.sh` explicitly accepts that exact condition when every test passes. The actual remote wrapper did not run successfully.

Python syntax compilation and all seven small CPU tensor/bundle tests passed, covering pooling gradients, affine equivalence, raw-count inference, zero-length silence, both training objectives, silence supervision, and preservation of a complete installed auxiliary bundle. These small tests do not execute a Zipformer, train on recordings, or validate ONNX export.

The authorized supervisor failed five requests with exactly `NotFoundError: No such file or directory`, without any container log:

- Run 1: read-only execution/data probe.
- Run 2: train-image dependency and training-data probe.
- Run 3: the required six-shard `bash tools/autolab_tests.sh` run.
- Run 4: the integration tensor unit tests in the train image.
- Run 5: a fresh train-image/checkpoint/schema probe after local preparation and validation.

Returned request results report $0 spent and the full $240 GPU budget remaining. No training run, selected checkpoint, ONNX parity result, full-session target comparison, or practice score exists. Practice-score calls used: 0 of 3. Target recoveries and wrong-call changes are unknown, not zero.

## Limits and required continuation

Restore supervisor execution first. Inspect available authorized training checkpoints and training-data metadata through that supervisor, then train and export the matched pair. Do not declare `.autolab/model.json` until the complete selected CIF bundle has passed export and SDK checks. Running the main listener without the CIF bundle would not evaluate this experiment.

Compare full-session exported CIF events against the matched CTC control on the registered additions. A recovery must be a previously missing planted consonant whose entire recorded contribution interval lies within the added speech. Cropped or teacher-only recoveries do not meet that condition. Apply the user's premise kill rule only to actual measured recoveries; infrastructure failure cannot establish it. If three or more targets win, take the required practice score before deciding. Global losses belong to the fold gate, subject to the fully clean-session hard stop.

The source change is retained for continuation, with an untracked `.autolab/outcome.json` recording `blocked` and an empty measured object. No trained model is declared, no recovery is claimed, and the hypothesis remains unresolved. Firing-frame token timing and independent CTC support are material collateral risks that require the full exported-stream and practice checks.
