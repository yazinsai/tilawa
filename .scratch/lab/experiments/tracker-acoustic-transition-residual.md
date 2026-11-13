# Original acoustic transition evidence for local tracking

Status: needs-human. Prototype prepared; training and validation blocked before execution by the supervisor's Modal runner.

## Prediction before measurement

Tracker −1, overall 0 on practice. Original log-mel trajectories may distinguish the near continuation from the rhyme restart where decoded phones lose that distinction. Correcting localization may expose the documented nearby listener false warning.

Target: `lab-v1.5-p46/phone/warning:skipped_passage/54:36:1/121.84`. The previous audits establish a retained near continuation only 0.3915 edits behind the restart; this experiment has not independently remeasured that margin.

## Prepared mechanism

The SDK listener optionally retains original log-mel features before phone classification. Each output frame carries 40 adjacent-mel-bin means and 40 within-frame endpoint differences, padded to the same width as the control. Feature rows follow the listener's absolute output-frame coordinates. SDK evidence rechunking retains them, and final evidence serialization packs successive intervals onto primary token metadata, ending at each token's own last frame and including intervening blank frames. `trackerInput` restores these segments while leaving startup posterior reranking disabled.

The estimator accepts only help-train `split=train`, `kind=none`, scored recordings without scripted errors, repeats or exclusions. Every speaker with any non-training row is excluded. Deterministic speaker partitions keep all takes of each speaker together. Four sampled transcript-constrained edit alignments supply uncertain word/prefix boundaries. The unchanged local candidate search supplies negative pronunciation strings.

Two scorers receive exactly the same examples, latent boundaries, candidate pronunciations and temporal bins. Both use the same temporal convolution, adjacent-phone pronunciation embeddings, attention, and output architecture. The original arm consumes the spectral means and transition differences; the control consumes every expanded posterior class probability, including blank, and its temporal difference. Both inputs have 86 dimensions for the shipped inventory. Normalization is estimated independently from the fitting speakers in each partition. A joint selection rule minimizes summed ranking errors across both arms on omitted training speakers, then calibrated negative log likelihood, then model size. Temperature comes from those out-of-speaker predictions, using the existing three-nats-per-edit convention. Practice is absent from all training stages.

Local DP provenance retains the pronunciation and acoustic charge over the same recent 16 observations for each existing candidate. When a competing run-origin change wins the original search, the original-feature scorer compares retained alternatives inside the existing commitment margin. It replaces only that interval's acoustic charge; accumulated older charges and movement costs remain in the DP score. Both alternatives consume the same acoustic frame interval. Unavailable feature evidence causes abstention. Phrase cancellation restores observation provenance with the search. Movement edges, commitment constants, published event handling and checker rules are retained.

The default model is `null`. Scoring and default SDK feature collection remain disabled until real training weights have been frozen and validated. No fitted artifact exists in this branch.

## Validation and observed infrastructure result

The supplied supervisor interface was used for three requests:

- Run 1: inventory help-train labels and locate target audio/evidence.
- Run 2: six-shard `bash tools/autolab_tests.sh`, including both package builds and core/SDK tests.
- Run 3: minimal `pwd` and `LAB_DATA` diagnostic.

All three returned `NotFoundError: No such file or directory`, with no shard output and zero reported compute spend. None reached the requested command. No data inventory, training decode, model fitting, Python/JS parity check, target audit or unit-test result was obtained. This repeats the infrastructure condition recorded by the earlier acoustic-run experiment; it does not measure the acoustic premise.

Locally, `git diff --check` passed and both Python sources parsed without importing training dependencies. Added tests cover preservation of transition direction, token-only serialization with blank intervals, frame-buffer retirement/continuity, shared comparison intervals, missing-evidence abstention, phrase rewind, and training speaker exclusion. Their execution remains pending. These static checks do not establish that package builds or tests pass.

Practice scores used: **0 of 3**. Target wins: **unmeasured**. The card's original-versus-posterior preference criterion and the instruction to kill a built change with fewer than three target wins have not been evaluated. An infrastructure failure supplies no premise-based kill result.

## Resume after the executor is restored

Run the six-shard test command on the final source snapshot. The training pipeline, run on Modal through the supplied supervisor, is:

```bash
pnpm --filter @tilawa/core... --filter @tilawa/sdk... build
ACOUSTIC_TRANSITION_CACHE=/scratch/tracker-acoustic-transition-residual/decodes pnpm --filter @tilawa/sdk exec tsx src/acoustic-transition-decode.ts
python3 packages/core/test/test_acoustic_transition_train.py
python3 packages/core/src/acoustic_transition_train.py boundaries "$LAB_OUT/acoustic-transition-training.json" "$LAB_OUT"
pnpm --filter @tilawa/core exec tsx src/acoustic-transition-candidates.ts "$LAB_OUT/acoustic-transition-boundaries.json" "$LAB_OUT"
python3 packages/core/src/acoustic_transition_train.py fit "$LAB_OUT/acoustic-transition-examples.json" "$LAB_OUT"
```

The Python stages need the supervisor's training image, with PyTorch and NumPy. Keep intermediate training outputs in this experiment's scratch directory for later runs. The generic SDK `acoustic-transition-audit.ts` accepts `--evidence`, `--audio`, `--fit` (the output directory), and `--at`, with optional `--apply`. It checks both exported models against Python parity samples before replay, records actual comparison boundaries and the matched control score, and writes an audit to `LAB_OUT`. First audit the training-frozen comparison; then apply the prototype and inspect target warning outcomes. Follow the requested premise/target-win rule before spending a practice score. Install only the frozen original arm as engine code if the experiment survives that check.

## Limits

Finite latent boundary samples approximate marginalization. Temporal pooling caps each comparison at 64 bins. The two-bin spectral grouping does not retain every original mel value. Only the existing recent-cost suffix receives learned acoustic scoring; older evidence keeps its original charges. The control has full posterior class probabilities but cannot recover distinctions already removed by phone classification. Without executing training and the actual interval audit, neither useful discrimination nor transfer to the target is established. No release-held-out audio, test labels, manifests, scorer or lab tooling was changed.
