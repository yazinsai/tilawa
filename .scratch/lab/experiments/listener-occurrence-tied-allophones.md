# Occurrence-tied acoustic mixtures

Status: needs-human. The implementation draft is locally checked, but the supplied supervisor cannot execute Modal requests. Training, extractor verification and the four target comparisons are unmeasured. This branch is not ready for scoring. The experiment card is unchanged.

## Prediction and premise

Predict one false warning removed with no losses: listener −1 and overall −1. The hypothesis is that a rival pronunciation can combine individually plausible frames from incompatible realizations of the same sound. Holding one mixture identity throughout that phone occurrence should reverse the comparison toward the written pronunciation.

The card's premise fails if no target changes from rival-preferred to written-preferred under occurrence tying. The task additionally requires at least three actual target warnings removed before proceeding to practice scoring. Neither condition has been evaluated. An execution-service failure is not evidence that the acoustic premise is false, so no measured kill is claimed and the engine draft is retained.

## Implemented mechanism

`modal/student_occurrence_mixture_export.py` exposes the deployed encoder's final linear-head input in a separate ONNX copy. For a dynamically quantized head it exposes dequantized activation codes, including the actual activation zero point and scale. It leaves the primary graph's nodes and outputs intact. Export requires exact equality of all primary outputs and recurrent caches over eight successive windows, and verifies effective affine-head reconstruction. These checks have not run on the supplied model.

The SDK preparation CLI replays the complete recording with the deployed streaming window, shift, padding and recurrent-state semantics. It preserves the native 40 ms acoustic clock and records full parent log probabilities and frozen features in a reusable JSONL cache. It accepts only usable help-train training-speaker `none` takes without annotated mistakes, repeats or exclusions. An explicit speaker exclusion list is required; any speaker appearing in a nontraining label is also excluded. The trainer independently checks source, split, label kind, speaker exclusions and observation geometry. The supplied training-only copy's release-speaker isolation still requires an actual metadata audit, which could not run.

The trainer fits a 12-dimensional whitened PCA projection and three diagonal Gaussian components per phone. Parent transcript-conditioned soft occupancies initialize the emissions. Three frame-mixture initialization updates are followed by three joint boundary/component EM updates. Gaussian statistics and mixture priors shrink toward a shared training distribution, with a fixed variance floor. Regularization is selected from 4, 16 and 64 using three disjoint speaker groups and complete-pronunciation likelihood per frame, giving each validation speaker equal weight. Projection and emissions are refitted within each training fold; final parameters are fitted on all eligible training speakers after selection. No practice observations enter fitting or selection.

The sequence model uses the complete CTC phone chain with optional intervening blank states. A phone occurrence selects one latent component on entry, retains it on every self-loop and marginalizes it on exit. Different occurrences, including equal successive transcript labels, select independently; equal successive labels still require blank separation. Blank emissions use the frame mixture in both arms. The matched control uses exactly the same fitted projection, Gaussian components, mixture weights, timing arcs and observations, but marginalizes component identity separately at every phone frame. Both scores subtract the same emission-free completion partition. No phone is optional, and every complete boundary path is summed. This isolates component persistence rather than increasing acoustic capacity or changing pronunciation timing.

The SDK requests the checker's complete written and already accused string-path comparisons on their existing frame windows. It issues a certificate only for finite strict opposing preferences: tied prefers written and switching prefers rival. Unobserved phones, unsupported pronunciations and unavailable frame windows abstain. The checker validates scores, word identity, complete strings and exact endpoints. Graph confirmations are exempt. Its internal issue and ownership ledgers retain their original verdicts, preserving insertion merges and later candidate generation.

The Node loader requires both declared side files to exist, match their byte counts and SHA-256 hashes, and match the primary model identity and geometry. A declared broken bundle throws rather than silently scoring with the primary listener. Audio sessions perform an auxiliary review at completion, retain certificates through ordinary evidence serialization and publish explicit downgrades only for the exact live occurrences removed by otherwise identical replay. Event identity includes error class, so an insertion cannot collide with a wrong-word retraction. Primary tokens, posteriors and tracker inputs are unchanged by certificate transport.

The exact-window evidence and loader scaffolding were adapted from the earlier logit-nullspace branch. No fitted assets or acoustic conclusions were reused from that draft.

## Validation and execution block

All three supplied-supervisor requests failed with `NotFoundError: No such file or directory`, zero compute cost and no container command log:

1. Read-only discovery of the authorized model, help-train labels and persistent scratch directory.
2. The prescribed six-shard `tools/autolab_tests.sh` harness, followed by the small Python numerical tests on shard zero.
3. A minimal read-only directory-listing retry.

The user was asked to restore the supervisor's execution/data setup. No alternative Modal access was attempted. No training or practice audio was processed on the Mac.

Local checks used isolated ignored dependencies under `.autolab` and ran only small unit cases:

- TypeScript build of both core and SDK passes.
- Nineteen relevant Vitest cases pass: nine scorer/transport tests, five checker certificate tests, one session completion test, one missing/corrupt bundle test, and the three existing insertion-loader tests. SDK configuration supplies its existing 60-second test timeout; the new core cases allow 30 seconds for reference initialization.
- Three Python numerical tests pass. Complete boundary and latent-component sums match exhaustive enumeration; state occupancies match finite-difference derivatives; entry responsibilities count the correct number of phone occurrences; a single-component model agrees between arms; and a tiny synthetic EM fit has finite, positive parameters with zero coverage for an absent phone.
- Python compilation and `git diff --check` pass.

The first local Vitest invocation omitted the SDK configuration and therefore encountered its absent package alias and 5-second timeout. Rerunning with that existing configuration exposed a same-ID insertion collision in the new retraction test, which was corrected by including error class in event identity. The final 19 cases pass. These local cases are not the required complete Modal suites or acoustic validation.

No target recording was accessed. Actual target wins, losses, listener/overall changes, clean-session behavior and studio behavior are unknown. Practice scores used: zero of three. There is no fitted scorer or verified complete bundle, and no `.autolab/model.json` declaration. A main-listener score would not evaluate this draft.

## Continuation and limits

Restore the supplied supervisor and audit the actual model and help-train eligibility metadata. Run the required complete core/SDK suites there. Export and verify the frozen extractor, prepare the training-only feature cache, fit and freeze the scorer, and verify a complete bundle retaining the unchanged primary model, manifest and installed insertion-teacher files. The additional declared assets are `occurrence-mixture.onnx`, `occurrence-mixture-emissions.json` and `occurrence-mixture-asset.json`; export metadata and the training report record provenance and selection.

Entry points:

```sh
python3 modal/student_occurrence_mixture_export.py --model <primary-dir> --out <bundle-dir>
pnpm --filter @tilawa/sdk exec tsx src/bench/occurrence-mixture-cli.ts --model <primary-dir> --extractor <bundle-dir> --labels <training-labels> --excluded-speakers <exclusions.json> --out <observations.jsonl>
python3 modal/student_occurrence_mixture.py --metadata <bundle-dir>/occurrence-mixture-metadata.json --input <observations.jsonl> --excluded-speakers <exclusions.json> --out <bundle-dir>
pnpm --filter @tilawa/sdk exec tsx src/bench/occurrence-mixture-cli.ts --model <complete-bundle> --audio <wav> --evidence <evidence.json> --out <comparisons.json>
```

The comparison CLI discovers engine-requested windows without a target list and asserts tracker and insertion equality. Count actual target warnings removed after replay, not merely favorable scores. Fewer than three wins requires reverting engine changes and recording a measured kill. Three or more requires practice-score before deciding; aggregate losses are handled by the fold gate.

A diagonal GMM on a small learner corpus may not separate phonetic realizations from speaker or channel variation. PCA may discard useful distinctions, sparse coverage may cause abstention, and EM can converge to poor boundaries. CTC duration-expanded labels remain separate transcript occurrences rather than one explicitly annotated physiological articulation. The current audio route buffers complete sessions and reviews at completion; it does not establish timely streaming vetoes, bounded memory or mobile latency. Full-stream primary-output parity, graph/insertion preservation on real audio, and genuine-substitution retention remain unverified. Positive written-word preference cannot itself guarantee that an unusual real substitution survives.
