# Local mistake-recovery motion state

Status: needs-human. The prototype is committed, but training, target replay and validation require a working supervisor Modal runner. The model is opt-in; no fitted model has been enabled.

## Prediction

Tracker −1, overall 0, as recorded on the unchanged experiment card. The target is the phone false `skipped_passage` warning at `lab-v1.5-p46 / 54:36:1 / 121.84 s`. Better localization may expose the previously observed listener false wrong-word warning nearby, so a binding change is not a claimed net gain.

The premise test is whether a model frozen using training speakers retains the near continuation ahead of the rhyme restart over the identical retained observation interval. That comparison has not been measured here. No practice score was requested.

## Mechanism

The existing local search in `local.ts` retains one minimum-cost history per reference-phone cell. Its edit and jump prices do not read a mistake-recovery history. This confirms that the proposed state is absent; it does not establish that adding one will recover the target.

`episode-local.ts` retains two candidate states per reference cell, ordinary reading and mistake recovery, with separate costs, run origins and edit counts. Every consuming step and reference-only deletion propagates both states. A normalized joint `P(event, next state | previous state)` supplies a likelihood ratio against an independent event distribution. The event alphabet is ordinary matching, phone editing, a one-word omission and an in-ayah repeat. The contribution uses the tracker's existing three-nats-per-edit conversion.

Acoustic insertion, substitution and deletion prices remain the existing fuzzy costs. Local jump distance, jump penalty, ayah horizon and surah-entry cost come from `local.ts`, rather than duplicated replacement constants. Jumps outside the single-ayah-supervised omission/repeat cases retain their existing passage-relocation cost and propagate the current episode state without an extra motion charge. Global acquisition and relocation, commit margins, event delivery and checker code are unchanged.

Candidate ranking and acoustic bookkeeping are separate. `recentCost`, `stepCost` and edit counts describe acoustic and existing movement charges on the selected path; the episode likelihood contribution does not become a new misfit or checker threshold. Save/restore includes both states and uses immutable arrays between steps.

`TrackerOptions.episodeMotion` accepts a frozen model. With no model, or a training-selected independent control, the existing search is used. No hand-initialized model is enabled by default.

## Training implementation

`episode-motion.ts` supplies an edit DAG and forward/backward estimator. Labelled omissions bypass a complete word without consuming heard tokens. Labelled single-word repeats are unfolded into two reference copies, with one repeat entry. Substitutions retain the written reference and use the unchanged phonetic edit channel. Both event boundaries and latent states are summed during estimation; there is no best-alignment boundary labelling. Every transition-model row is normalized jointly across events and next states, with positive additive smoothing.

`packages/sdk/src/episode-motion-fit.ts` reads only `help-train/labels.jsonl` and its recordings, using the shipped `models/spliced-clean-int8` listener. It restricts fitting to scored, unexcluded, single-ayah recordings labelled none, skipped word, substitution or repeat. It excludes all speakers having any non-training row, not just rows marked non-training, and rejects unsupported/multiword scripted events.

Model selection compares independent and two-state controls at smoothing masses 1 and 16 using up to five speaker-grouped folds wholly within training speakers. The selected model is refit on those training speakers. The command writes a model and provenance report under `LAB_OUT`, including listener and label hashes, eligible class counts and fold likelihoods. No practice information enters fitting or selection. Passage relocation is not estimated from these clips.

The initial probabilities in the estimator are EM initialization, not measured parameters. They must not be deployed as a substitute for the training fit.

## Validation and limits

Added core unit tests cover:

- Forward/backward agreement with exhaustive enumeration of tiny boundary/state paths.
- Mandatory omission and repeat identity over every complete alignment.
- Joint normalization and invalid-model rejection.
- Positive fitted distributions and the independent training control.
- Existing-path equivalence when no dependent model is selected.
- Unambiguous clean forward progress and acoustic-cost reporting.
- Rewinding both states and retaining both finite explanations after a mismatch.

`git diff --check` passes. None of the unit or integration tests is claimed to have passed on Modal.

Supervisor requests:

1. Run 1: inspect authorized training labels and directory layout. Returned `NotFoundError: No such file or directory`, with no data output and zero reported spend.
2. Run 2: six shards of `bash tools/autolab_tests.sh`. Returned the same `NotFoundError`, with no build/test logs and zero reported spend. This request preceded the final estimator and unit-test additions.
3. Run 3: build core/SDK and run `packages/sdk/src/episode-motion-fit.ts`, with a 60-minute timeout. Returned the same `NotFoundError`, without a model, provenance report or build log.
4. Run 4: six shards of `bash tools/autolab_tests.sh` over the final implementation and tests. Returned the same `NotFoundError`, with no build/test logs and zero reported spend.

All four requests failed in the supervisor runner. `.autolab/outcome.json` records `needs-human`, with an empty measured object rather than fabricated zero wins or score changes.

No training recordings, frozen parameters, identical-interval candidate costs, product target replay or practice result were obtained. Target wins and overall/component deltas are unknown, not zero. This is an infrastructure block, not a killed hypothesis. There is no evidence to apply the premise kill criterion or the fewer-than-three-wins rule.

## Remaining work

Recover the supervisor runner and repeat the final training/build/test requests. Inspect the returned eligible labels and provenance before accepting the frozen model. If the two-state model wins training-only selection, enable that frozen model in the tracker and compare near continuation and rhyme restart over the identical retained target interval, then replay the product warning. Kill and revert engine changes if the stated premise or target-win criterion fires. Only request a practice score if the target-win rule requires it; all three practice-score allowances remain unused.
