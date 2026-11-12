# Sequence-gradient pronunciation evidence

Status: needs-info. The supplied supervisor still cannot execute the model/data audit after the implementation retry. Implementation and acoustic validation are pending; this is not a premise-based kill. The experiment card is unchanged.

## Prediction

Remove two false wrong-word warnings and lose one caught substitution: listener −1 and overall −1. The proposed signal is the acoustic-model correction required by a complete pronunciation, rather than its likelihood or a pooled encoder representation.

The card's early criterion requires two target windows to favour the written pronunciation under the gradient scorer while both controls favour the accusation. The user's final execution rule requires at least three actual target warnings removed to continue. If fewer than three are recovered by the built change, revert it and record the measured kill; if at least three are recovered, run a practice score before deciding. Unavailable compute does not establish either acoustic result.

## Intended mechanism and implementation prerequisite

Extract a fixed low-dimensional projection of each complete candidate's CTC log-likelihood gradient with respect to the frozen uppermost encoder block's parameters. Fit a small pronunciation-compatibility scorer on authorized transcribed recordings and eligible help-train training-speaker `none` recordings. Include recorded-versus-competing and neither-correct pairs, randomize candidate order, exclude release-held-out speakers, and freeze selection within speaker-grouped training partitions. Use equal-capacity likelihood-only and encoder-feature controls on identical examples and partitions.

The SDK comparison must use the complete written and already-accused pronunciations over the same existing string-path warning window. A veto requires gradient preference for the written pronunciation and opposing preferences from both controls. Bind the certificate to the occurrence, both complete strings and exact frame bounds. Preserve primary decoding, tracker events, graph confirmations and insertion decisions, including the current insertion teacher. The recheck updates no model parameters.

`modal/student_train.py` defines the streaming Zipformer encoder and its CTC head. `modal/student_export.py` exports the streaming graph and recurrent states. The normal supplied deployment is an int8 ONNX model. Before implementing extraction, inspect the actual supplied assets and determine whether a matching frozen differentiable encoder or a verifiable reconstruction of its uppermost block is available. Head/logit gradients, Gaussian-mixture Fisher scores and a straight-through quantization surrogate do not establish the specified encoder-parameter gradient mechanism. Source availability and numerical parity are unresolved, rather than evidence that the acoustic premise is false.

Existing experimental branches contain reusable exact-window certificate transport and separate streaming-cache patterns. Their scorers do not implement this experiment's CTC encoder-parameter gradients. No prior engine diff has been copied into this branch.

## Validation and limits

Read `AGENTS.md`, the unchanged card, the student training/export definitions, the SDK model/Node/ONNX loading path, and prior listener-evidence integrations. The initial worktree was clean. No model, training recording or target acoustic window has been inspected, and no scorer has been trained or bundled.

Supervisor request 1 was a read-only dependency and asset inventory under the authorized `LAB_DATA` and this experiment's persistent scratch directory. It returned `NotFoundError: No such file or directory`, zero compute cost, and no shard output. Thus even the dependency and asset inventory did not run.

The prescribed six-shard test harness (request 2) and a minimal `pwd` execution retry (request 3) also returned the same `NotFoundError`, zero compute cost and no shard output. No build or test assertions ran. Even the minimal retry failed before execution, so changing the training command cannot resolve the current blocker. The user has been asked to restore the supervisor.

On the renewed implementation attempt on 2026-11-13 (Bahrain time), request 4 repeated the read-only inventory through the prescribed supervisor, with a ten-minute timeout. It checked dependency availability and listed model/checkpoint and help-train metadata files only under `LAB_DATA` and `/scratch/listener-sequence-gradient-evidence`. Request 5 independently ran the exact prescribed six-shard harness, `bash tools/autolab_tests.sh`, also with a ten-minute timeout. Both returned `NotFoundError: No such file or directory`. Their result JSON files record `usd: 0.0`; neither produced a shard directory or command log. No inventory or test output was returned.

The renewed source review established that existing branches can capture actual checker comparison requests and carry occurrence-bound veto certificates without modifying primary tokens. It also confirmed that the logit-nullspace/depth extractors expose hidden activations and the Fisher-score scorer differentiates a fitted background Gaussian mixture. None supplies gradients with respect to the acoustic encoder's parameters. Reusing their scorer would implement a different mechanism. The only model file present locally is the tiny SDK test fixture; there is no local trained checkpoint, deployment model or labelled training corpus with which to establish the required gradient extraction or fit the controls.

### Why this retry cannot implement the engine change

The previous feedback asks for engine code or an explanation of why implementation cannot be done. The concrete blocker is execution of the sole authorized compute/data route, rather than an observed failure of the proposed acoustic signal. Five supervisor requests, including a minimal command and independent inventory/test commands, now have the same error before returning any command output. The supplied model and training recordings exist only behind that route. Whether a matching differentiable model is available remains unknown; the inventory never ran, so this report does not infer its absence.

Implementing and handing back an operational recheck requires accessing those assets, verifying the projected uppermost-block derivatives, fitting and freezing all three scorers on eligible speaker partitions, and verifying the complete primary/teacher/side-asset bundle. The user explicitly requires that complete bundle before handoff and prohibits alternate Modal/data access. Local certificate plumbing or a mock gradient extractor could not satisfy those prerequisites or evaluate this experiment. Therefore this retry leaves the engine unchanged and records the execution blocker instead of declaring an untrained or incomplete side asset.

Target recoveries, listener/overall deltas, gradient parity and training-control performance are unmeasured. Practice scores used: zero of three. There is no trained side asset or `.autolab/model.json`; decoding with the main model would not evaluate this mechanism. There is no killed outcome because infrastructure failure supplies no evidence for the acoustic kill condition.

No engine code, scorer, lab tooling, supplied manifest, test data, evidence or experiment card has been edited. The only tracked change is this required experiment report. `git diff --check` passes. The requested behavioral tests were attempted twice through the prescribed harness and could not execute.

## Resume

Repeat the dependency/model inventory and required tests after the supplied supervisor is restored. Establish faithful uppermost-block gradient extraction and test projected derivatives against an independent numerical check, including full CTC completion and repeated phones. Then implement and test the exact-window SDK/core route, perform training-only speaker-grouped selection, and verify the complete primary/teacher/side-asset bundle before declaring it.

Use actual checker-generated windows for target diagnostics; target labels select report cases only and never training examples, candidates or thresholds. Count actual removed warnings after otherwise identical replay and verify tracker, graph and insertion preservation. Apply the user's three-target rule and practice-score requirement using those measured results.
