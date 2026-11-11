# Acoustic posterior / pronunciation cross-encoder for local runs

Status: needs human infrastructure recovery. The implementation and training pipeline are on this branch, but the supplied supervisor runner has failed before executing any command. No trained weights, frozen calibration, target replay, or practice score exist. This is not a premise kill.

## Prediction

Tracker −1, overall 0 on lab-v1.5u phone. The target is the false skipped-passage warning in `lab-v1.5-p46` at `54:36:1`, 121.84 s. Correcting the local binding may expose the documented listener false warning at `54:37:2`; checker handling is unchanged.

The card's premise test is whether a training-frozen comparison ranks the near continuation ahead of the rhyme restart on their identical acoustic interval. The builder must also stop if the built change wins fewer than three targets. Neither condition has been measured here. Zero targets measured is not a measured result of zero wins. No practice score was requested (0 of the allowed 3).

## Mechanism implemented

`trackerInput` retains the listener's top-k posterior trajectories in a separate `localPosteriors` field while keeping `posteriors` absent, so startup and global acquisition retain their decoded-only product behavior. A trained local scorer, when installed, accumulates these rows in a separate bounded buffer.

The local DP keeps immutable pronunciation provenance for its existing states. Each comparison uses the same suffix of at most 16 consumed observations, the existing recent-cost horizon. Reference-only deletions remain in the candidate pronunciation; jump edges contribute no unspoken pronunciation. The old acoustic charges for that suffix can therefore be replaced while movement, distance and surah-entry charges stay in the original DP cost. Prefix charges before that common suffix also remain. Provenance is periodically truncated without changing the scored suffix, and phrase cancellation rewinds its observation interval with the DP.

Only an original run-origin change invokes inference. Candidates inside the unchanged tracker commitment margin compete on the same posterior interval. Empty-pronunciation insertion states cannot supply a lexical comparison. If required rows or symbols are unavailable, the original choice is retained. Local landing confirmation, acquisition, relocation, checker windows, accusations and lock-drop rules are unchanged.

The scorer is a small acoustic/text cross-encoder with three-position acoustic and pronunciation convolutions, learned cross-attention, a learned relative-time bias, and a joint interaction layer. All class probabilities, including blank-dominant frames and sub-blank classes, reach the model. Long intervals are averaged into at most 64 ordered temporal bins; no argmax or forced CTC pronunciation likelihood is used for comparison. JS CPU inference matches the Python architecture. Acoustic features are shared across candidates on an interval.

## Training and freeze protocol

The SDK decoder reads only `LAB_DATA/help-train/labels.jsonl` and the corresponding audio. It accepts explicitly training-split, scored, none-labelled recordings with no events, non-errors or exclusions. A speaker with any non-training row is excluded entirely. Every take from a speaker belongs to one deterministic training-only fold. Audio paths are constrained to the help-train root. Decoder, label and reference hashes are recorded.

The estimator samples four transcript-constrained latent edit paths for word boundaries and streaming prefixes. These paths provide segmentation supervision; the network still receives complete posterior trajectories. Contrasting pronunciations are drawn from the unchanged local candidate search at those observations, rather than selected verse identities or a replacement vocabulary search.

Dimensions 8/16 and 12/24 epochs are selected by speaker-held-out ranking errors, then calibrated validation NLL, then smaller size. Temperature is fit to out-of-speaker training predictions, and the selected model is refit on the authorized training speakers. Cost conversion uses the tracker's existing 3 nats per edit. A model, provenance/selection report, and Python/JS parity fixture are exported together. No target replay may precede that freeze.

`acoustic-run-model.ts` currently exports `null`. This intentionally records the absence of trained weights; the default product cannot apply an untrained network. There is also an explicit model/audit option for post-training inspection. The read-only audit helper requires a training-only fit and parity check before opening a practice case, and records the frozen model hash, original candidate costs and learned costs. It does not score the practice set.

## Validation performed

- `git diff --check`: passed.
- Python syntax compilation of estimator and estimator tests: passed. Torch-dependent tests were not executed locally.
- Lightweight Node smoke checks of the actual source modules: passed for sub-blank distinctions with identical greedy observations, temporal-order sensitivity, complete posterior binning, local-only posterior preservation, unchanged unambiguous local progress, common candidate observation counts, and exclusion of a speaker with a non-training row. These are smoke checks, not the required full Vitest suites.
- Added core unit tests for scorer behavior, suffix accounting, bounded provenance, invalid model handling, unchanged progress, candidate intervals and rewind. Added SDK unit tests for training eligibility and speaker partitions, and Python estimator invariant tests.

Supervisor requests:

| Request | Purpose | Result |
| --- | --- | --- |
| 1 | Inspect authorized training layout / storage | `NotFoundError: No such file or directory`, before execution, $0 compute |
| 2 | Required core/SDK build and six-shard tests | Same pre-execution error, $0 compute |
| 3 | Decode, latent boundaries, unchanged-search contrasts, train and freeze | Same pre-execution error, $0 compute |
| 4 | Retry required core/SDK build and tests | Same pre-execution error, $0 compute |

No data, manifests, scorer, lab tooling or experiment-card edits were made. No fold or discard was requested.

## Resume after supervisor recovery

Run the required build/tests through the supplied supervisor and run the three training stages, on Modal only:

```sh
python3 /Users/rock/ai/projects/tilawa/tools/autolab.py modal-run --shards 6 -- 'bash tools/autolab_tests.sh'

python3 /Users/rock/ai/projects/tilawa/tools/autolab.py modal-run --timeout 60 -- 'pnpm --filter @tilawa/core... --filter @tilawa/sdk... build && pnpm --filter @tilawa/sdk exec tsx src/acoustic-run-decode.ts && python3 packages/core/src/acoustic_run_train.py boundaries "$LAB_OUT/acoustic-run-training.json" "$LAB_OUT" && pnpm --filter @tilawa/core exec tsx src/acoustic-run-candidates.ts "$LAB_OUT/acoustic-run-boundaries.json" "$LAB_OUT" && python3 packages/core/src/acoustic_run_train.py fit "$LAB_OUT/acoustic-run-examples.json" "$LAB_OUT" && python3 -m unittest discover -s packages/core/test -p test_acoustic_run_train.py'
```

This requires NumPy and CPU Torch in the supervisor environment; that dependency could not be inspected. Preserve the exported model, fit and parity files under this experiment's `/scratch/tracker-acoustic-run-crossencoder` directory for subsequent runs. Verify the Python/JS parity before target inspection with `.scratch/lab/acoustic-run-target-audit.ts` in audit mode, then replay with apply mode. Only install the frozen model into `acoustic-run-model.ts` if the premise survives. Follow the supplied fewer-than-three-targets rule before requesting practice scoring. If the premise fails, revert engine changes, retain this write-up with the measured finding, and mark the outcome killed.

## Limits

The acoustic distinction that motivated the card has not been established by training or target replay. Calibration, cross-language parity, CPU latency, full build/tests, target wins and downstream warnings are all unvalidated. The synthetic smoke checks demonstrate representational capacity and plumbing, not real-audio discrimination. Finite latent boundary samples and 64-bin resampling are approximations. Candidate provenance follows the existing minimum-edit traceback, so latent runtime boundaries are not independently inferred by the network. The default model slot remains empty until the prescribed training completes.
