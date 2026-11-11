# Conditional phone identity distillation

Status: implementation present; real training and target evaluation blocked by the supervised Modal runner before execution. No trained model is selected, and this branch is not ready for model scoring.

## Prediction and decision rule

The card predicts one fewer listener wrong call and one fewer overall wrong call on phone practice. Its premise is that ordinary distillation attenuates supervision among non-blank identities by the teacher's speech probability. The registered complete written-versus-accused comparisons are:

- `lab-v1.5-p11/phone/warning:wrong_word/26:106:1/104.32`
- `lab-v1.5-p36/phone/warning:wrong_word/55:48:1/282.64`
- `lab-v1.5-p55/phone/warning:wrong_word/26:174:8/107.08`
- `realuser-v1-sob-049/phone/warning:wrong_word/92:13:1/36.44`

Checkpoint selection must precede these comparisons. No preference reversals relative to the matched ordinary arm would falsify the card. The implementation instructions additionally require at least three target wins before a practice score; fewer wins require a kill. Neither criterion has been measured here. Zero practice scores have been used.

## Mechanism implemented

`modal/student_identity_distill.py` provides `prepare`, `fit`, and `export`. Preparation captures the frozen streaming encoder activations actually consumed by the shipped integer classifier, including activation quantization. Both arms start from the same dequantized non-blank classifier weights and bias. The encoder, original classifier, blank probability, and streaming states remain in the exported graph.

The fp32 Muaalem teacher hears the same waveform as the student. Full-transcript CTC forward/backward supplies one soft occupancy per spoken phone occurrence; blank states participate in alignment but receive no identity supervision. Repeated articulations retain separate CTC states. The identity target is the teacher distribution normalized over every non-blank class, rather than its stored top-k approximation.

For frame occupancy `gamma`, teacher conditional identity `q`, and identity-head prediction `h`, the conditional loss is `-gamma * sum(q * log(h))`. The ordinary control additionally multiplies this term by the teacher's total speech probability. Both use the same occupancy denominator, so normalization does not cancel the attenuation being tested. An identical all-frame KL anchor to the original conditional identity accompanies both losses. Both arms have identical architecture, initialization, batches, update count, and regularization.

Authorized original clean-match recordings require training-speaker cluster provenance. Entire clusters with non-training rows or gold exclusions are banned. Only help-train training speakers' audited none-labelled takes without mistakes, repeats, unsupported labels, or exclusions enter training; non-training or excluded speakers are banned in full. The plan samples at most 256 eligible original recordings and reserves a deterministic fifth of training speakers per source for validation. No practice audio, labels, or evidence enter preparation or checkpoint selection.

The recipe is fixed before acoustic evaluation: seed 831, three views per recording, twelve epochs, batch eight, learning rate 0.0001, and identity anchor 0.1. Views reuse the shipped reverb, noise, gain, and spectrogram masking functions. Checkpoint selection minimizes conditional validation loss averaged across reserved speakers. The ordinary control is taken at that same update count. Plan, cache, and selected checkpoint checksums prevent replacing training inputs after selection.

The exporter appends a small float identity projection to the original int8 graph. It verifies unchanged original nodes and tensors, byte-identical original outputs and caches on eight consecutive optimized-runtime windows, unchanged blank rows, and conserved speech mass. It requires and verifies every inherited SDK teacher asset before preparation and copies those assets into each complete bundle. Generated bundle manifests receive the new primary checksum; supplied manifests are untouched.

In `packages/sdk/src/listener.ts`, the optional `phone_identity_logits` output activates reconstruction: the original blank log probability is copied, and `1 - p(blank)` is distributed according to the conditional head. `expm1` preserves very small speech mass. Exact silence remains silence. The existing decoder and clock then consume these rows; tracker and checker code are unchanged. Models without that output retain the existing path.

## Validation

- Ten Python unit tests passed locally. They compare transcript occupancies with exhaustive complete CTC paths, check repeated articulations and impossible transcripts, verify analytic gradients numerically, establish ordinary-distillation attenuation, reject overlapping speaker partitions, and exercise float/integer graph parity and a complete synthetic bundle export with inherited assets.
- The full SDK suite passed locally with one worker: 30 tests passed and one optional help-e2e test was skipped. Its five identity tests include a streaming integration check for changed identities, conserved blank/speech probabilities, contiguous posterior frames, and state continuity.
- Core and SDK TypeScript builds passed. The full core rerun with one worker and a 60-second default test timeout completed with 754 tests passed, two existing startup-persistence tests exceeding their explicit 15-second deadlines, and two Vitest worker RPC deadline errors. The startup file alone again exceeded those same deadlines (one test passed, two timed out). No assertion failures were reported. These unchanged core tests do not exercise the new SDK identity path; the full core suite cannot be reported as passing here. The earlier concurrent run had 709 passed tests, 47 timeout failures, and worker RPC deadline errors.
- No real training, real-audio export parity, target replay, clean-session comparison, studio comparison, or practice score has completed.

## Execution blocker and limits

Supervised runs 1 (inventory), 2 and 5 (six-shard tests), and 4 (CPU training preparation, fitting, and export) returned `NotFoundError: No such file or directory` before command execution, with zero reported compute spend and no command logs or model outputs. Run 3's GPU request was rejected because this is an explore card; the subsequent training request used the authorized CPU training image. These failures establish an execution blocker, not absence of suitable data or absence of target gains.

The numerical and synthetic tests establish the loss factorization and integration contract. They do not establish that the teacher has useful conditional evidence in ambiguous frames, that this small head learns it, or that the chosen head repairs three targets. Fixed speech mass prevents increasing total speech probability, but concentrating that mass can make a phone beat a formerly winning blank. Changing identity rankings can still create warnings or erase real substitutions.

No `.autolab/model.json` is declared because no complete real trained bundle exists. No killed outcome is declared because the premise and target-win criterion remain unmeasured. Resume by preparing, fitting, and exporting both arms through the supervised CPU runner into `/scratch/listener-conditional-identity-distill/identity-v1`, then compare the training-selected heads on complete existing warning windows and verify actual SDK target verdicts. If at least three targets are won, declare the verified conditional bundle and run the authorized practice score before any narrowing decision.
