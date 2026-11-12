# Latent spoken-phone channel for local tracking

Status: implemented as an opt-in prototype; real-data fitting and required Modal validation are blocked by the supplied runner failing before execution. No fitted parameters, target wins or practice delta are available. The experiment card is unchanged.

## Prediction

Tracker −1, overall 0: separating pronunciation variation from recognition error may reverse the near continuation's recorded 0.3915-edit disadvantage, recovering the false skipped-passage warning at `lab-v1.5-p46/phone/warning:skipped_passage/54:36:1/121.84`. Correct localization may expose the previously documented nearby listener wrong-word warning, so the prediction includes that loss.

## Mechanism

`packages/core/src/spoken-phone-channel.ts` represents reference phone → latent spoken phone → observed token. The first stage is a normalized, phonetic-class-shared pronunciation matrix. The second stage emits a token or epsilon for each spoken phone and has a normalized geometric insertion process. Matrix composition computes

`P(h | r) = sum_s P(s | r) P(h | s)`

including recognition deletion as an output. This exactly marginalizes the intermediate sequence for this one-latent-phone-per-reference-phone model; it does not select a single best pronunciation. The complete-string training forward/backward includes both insertion placements and a terminal insertion/stop state, so extra alignments divide probability rather than receiving free counting weight.

Recognition is fitted only on eligible `none` takes, then frozen. Vowel and tajweed takes estimate the pronunciation stage with latent alignments. Only their labelled reference words may vary during training; vowel scripts restrict variation to vowel phones. Other words retain identity pronunciation. Source/destination phonetic classes and identity relations pool sparse recognition counts; pronunciation changes share within a class or the existing confusable-phone pairs. Reference-word ownership never changes.

The direct single-channel control uses exactly the same combined training recordings, inventory, smoothing family, number of EM iterations, selected pooling mass and speaker folds. It fits reference → observation directly. It does not receive the pronunciation script mask. The control isolates the proposed channel separation from a change of training corpus.

`tracker.ts` accepts a training-frozen model through `TrackerOptions.spokenPhoneModel`. `local.ts` uses its composed likelihood for every local candidate, including continuation and restart, on the existing common consumed observation history. The observed-token likelihood baseline is common to all candidates, using the existing conversion of three nats per edit. Duplicate reference symbols and existing held-sound insertion discounts retain their old duration charges. Unweighted phone-edit metadata is retained even when a pronunciation discrepancy has low model cost.

Word-start jump edges, whole-word bypasses, jump/distance/surah-entry penalties, local window size, commitment rules, event handling and checker code are unchanged. Candidate diagnostics return copied relative scores for comparing the frozen two-stage and control replays. With no model supplied, local tracking uses the original costs. No untrained default model is installed.

## Training isolation and planned freeze

`packages/sdk/src/spoken-phone-fit.ts` opens only help-train labels/audio, the repository reference and the supplied deployed listener. It caches training decodes under the acoustic model hash in its output directory or the experiment's permitted scratch cache. It writes the fitted model and provenance report to `LAB_OUT`.

Eligibility requires `split: train`, a scored, non-excluded take, no repetitions, and kind `none`, `vowel` or `tajweed`. Speakers with any non-training row are excluded entirely. Lexical omission, addition, substitution and passage-skip scripts are excluded. Supported and unsupported tajweed site locations can weakly supervise their original words. No release-held-out speaker identifiers are hard-coded or opened.

Up to three speaker-grouped training partitions compare pooling masses 8 and 64 with 20 EM iterations. Selection uses the combined held-partition full-string likelihood of the two-stage model and direct control; validation allows pronunciation variation everywhere, as deployment does, rather than supplying the held script mask at inference. The selected mass is shared by both models and the final fit uses all eligible training speakers. The report records the selected-label hash, acoustic model hash, kind/speaker counts and every fold likelihood. Any partition lacking clean anchors or pronunciation examples fails instead of silently using an ungrouped validation split.

None of that real-data selection has executed or returned a model. The optional tracker plumbing is ready to receive the frozen result, but this branch's default engine has no learned contribution yet.

## Validation

Seven core and two SDK test bodies passed under Node 22's TypeScript transform with a temporary assertion adapter for the Vitest API. The adapter and temporary runner were removed; the actual test files remain. These checks cover:

- Analytic latent-phone composition, including recognition deletion, and identity-stage equivalence.
- Full-string probability normalization and the sum over alternative insertion placements.
- Script-mask isolation and unchanged recognition anchors when pronunciation supervision changes.
- A matched direct control that responds to the different training labels.
- Retained duration costs, exact-match costs and unknown-phone fallback.
- Common local model application, copied candidate diagnostics and exact save/restore replay.
- Lower pronunciation likelihood cost without erasing the observed phone edit.
- Exclusion of held-out/mixed-split speakers, lexical scripts, repeats and extra mistakes.
- Supervision confined to the original labelled word and rejection of an out-of-reference site.

Changed runtime sources also passed JavaScript syntax checks after Node's TypeScript transform. `git diff --check` passed. These are focused numerical/behavioral and syntax checks, **not** a TypeScript build or the required full Vitest suites.

Supervisor requests:

1. Inspect help-train availability/schema: failed before execution with `NotFoundError: No such file or directory`.
2. Six-shard core/SDK builds and tests: the same pre-execution failure.
3. Core/SDK build followed by training-only fitting: the same pre-execution failure.
4. Final six-shard core/SDK build and tests: the same pre-execution failure.

No real training decodes, selected parameters, target candidate comparison or product replay were returned. Practice scores used: **0 of 3**. Target wins and overall delta remain unknown. Infrastructure failure does not establish the premise kill or a measured fewer-than-three-target result.

## Limits and resumption

This compact pronunciation stage changes phone identity but does not insert/delete spoken articulations or model syllable/context-dependent pronunciation. Duration expansions remain the existing local cost's responsibility. None-labelled takes are treated as canonical-pronunciation anchors; unscripted learner variation in those recordings could contaminate that assumption. The two-stage model has extra structure and parameters, so normalization and the same-data control are necessary but do not themselves prove generalization or identifiable pronunciation errors.

Resume only through the supplied supervisor runner when its missing resource is restored:

```sh
python3 /Users/rock/ai/projects/tilawa/tools/autolab.py modal-run --timeout 60 -- \
  'pnpm --filter @tilawa/core... --filter @tilawa/sdk... build && SPOKEN_PHONE_CACHE=/scratch/tracker-latent-spoken-phone-channel/decodes pnpm --filter @tilawa/sdk exec tsx src/spoken-phone-fit.ts'
python3 /Users/rock/ai/projects/tilawa/tools/autolab.py modal-run --shards 6 -- \
  'bash tools/autolab_tests.sh'
```

Freeze the returned model without practice-driven changes, install it as the tracker default, and compare continuation/restart on the same target observation interval against `model.control`. Check the product warning as well as the local ranking. Apply the user's premise/fewer-than-three-target kill rule only to measured results: revert the engine changes and retain a killed write-up/outcome if it fires; if at least three targets are won, obtain a practice score before a gate decision. No localization recovery or warning reduction is claimed from the synthetic checks.
