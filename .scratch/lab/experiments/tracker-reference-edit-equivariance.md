# Paired-reference edit equivariance

Status: needs-human. The implementation is a disabled prototype: training, full tests and target replay are blocked by the supervisor's Modal environment. No weights were fitted or frozen, and no practice score was requested.

## Prediction

Tracker −1, overall 0 on lab-v1.5u, using the card's listener `exp-listener-teacher-insert-marginal-gate-ad6eb8c0`. The proposed recovery is `lab-v1.5-p46/phone/warning:skipped_passage/54:36:1/121.84`; a corrected binding may expose the documented nearby listener warning. The prediction remains unmeasured.

## Mechanism implemented

`packages/core/train/reference-edit-build.ts` admits only explicitly training-speaker, none-labelled help recordings with no planted events, repeats, unsupported labels or exclusions. It decodes each recording once with the available primary student, caches by audio and listener hash, and reuses exactly that token string for all reference variants. A deterministic hash selects a nonempty internal word. Three training-only copies delete it, insert a different word pronunciation, or substitute that pronunciation. Unchanged words retain an exact mapping to original word ordinals; the edited reference never replaces a product reference or data file.

Each original and edited run exposes the local search's word-start competitors. The training graphs retain insertion, substitution, reference-only deletion and existing word-start jump edges, including their ordinary distance and surah-entry charges. Their jump sources are frozen from the baseline search during graph construction. The transcript supplies training supervision, not an inference destination.

`reference-edit-fit.py` differentiates conditional path partitions through those graphs. Its location objective compares the complete competing graph with an in-order transcript region that allows lexical edits but excludes broader forward movement and repeats. A symmetric Jensen–Shannon loss compares the two runs' filtering distributions on corresponding unchanged word positions; changed and unmapped positions occupy a residual bin. Reference-only deletion closure uses `logcumsumexp`, so alternate edit paths remain in the partition.

The correction has six log multipliers: substitution, insertion, deletion, a jump over one word's remainder, broader forward movement, and repeat. The existing jump graph supplies all of these transitions. Three deterministic speaker folds select L2 regularization from 0.1, 1 and 10; a final fit uses the training speakers, and the estimator emits a source module only after selection succeeds. No practice observation enters estimation or selection.

`local.ts` carries original path costs separately from corrected ranking costs. The selected path's original charges still supply recent-cost and step-cost evidence. `tracker.ts` supplies the model to local continuation/restart search. Global acquisition, movement eligibility, commitment thresholds and checker handling have no edits. `REFERENCE_EDIT_MODEL` is currently `null`: this is explicit absence of a trained model, not fitted zero coefficients.

## Validation completed

Lightweight synthetic Node checks passed against the original `local.ts` from the branch base:

- Exact correspondence of every unchanged word under all three reference edits; the original index remains intact.
- Every reported position and raw cost agrees for identity weights and for the disabled model over five short clean, edited and repeated streams.
- Save/restore retains the selected position and costs.
- A deliberately weighted toy lexical jump changes its ranking while its reported original step cost remains 2.0003 edits.

Type-stripping syntax checks passed for the new TypeScript helpers and training/replay scripts; Python compilation passed for the estimator and its tests. `git diff --check` passed. These checks do not substitute for TypeScript build or the core/SDK suites.

Unit tests were added in `packages/core/test/reference-edit.test.ts`. The Python graph tests compare a tiny exact path partition with exhaustive enumeration and check gradients against finite differences; they have not executed because PyTorch validation belongs on Modal.

## Modal obstruction

All three supervisor requests failed before a command log or shard status was returned:

| Request | Requested work | Result |
| --- | --- | --- |
| 1 | Read authorized help-train layout and model metadata | `NotFoundError: No such file or directory` |
| 2 | Six-shard `bash tools/autolab_tests.sh` | Same error |
| 3 | Build, focused unit test, training export and estimator | Same error |

The supervisor result for each reports USD 0. No data inventory, decode, graph export, fitted coefficient, target snapshot or full test result exists from these requests. The training scripts were subsequently placed under `packages/core/train/` so source packaging includes them, and the Python fit should run in the supervisor's `train` image.

## Remaining work and limits

After restoring the supervisor environment, run the following through the authorized supervisor executable, with dependent steps kept sequential:

1. In the node image, build core/SDK and run `pnpm exec tsx packages/core/train/reference-edit-build.ts`. It preserves the paired graphs at `/scratch/tracker-reference-edit-equivariance/pairs.json`.
2. In the train image, run `python3 packages/core/train/reference-edit-test-fit.py`, then `reference-edit-fit.py --pairs /scratch/tracker-reference-edit-equivariance/pairs.json --out "$LAB_OUT/model.json" --ts-out "$LAB_OUT/reference-edit-model.ts"`. Preserve the fitted report in the experiment scratch directory and copy only the successfully generated runtime module into `packages/core/src/reference-edit-model.ts`.
3. Run the core/SDK suites on six shards. Replay the target with `pnpm exec tsx packages/core/train/reference-edit-trace.ts "$LAB_DATA/lab-v1.5u/sessions/lab-v1.5-p46.exp-listener-teacher-insert-marginal-gate-ad6eb8c0.phone.evidence.json" 54:36:5 54:37:9 121.84 /scratch/tracker-reference-edit-equivariance/model.json`. The inspector accepts a generic evidence path, changes only a disposable tracker source copy, and reports both candidates' costs on each common observation row, plus confirmed warnings.
4. Evaluate the trained target result under the user's kill rule. Fewer than three target wins requires reverting the engine changes and recording the measured kill; three or more requires a practice score before deciding.

The training graph freezes baseline jump sources and the initial local reference grid; inference continues to derive its sources and recenter from the selected state. Filtering distributions are used for the paired objective, rather than full future-conditioned smoothing or an exact product of the two reference graphs. These approximations need validation on training speakers and the target before an efficacy claim.

The card, scorer, lab tooling, manifests and data were not changed. Practice-score calls: **0 of 3**. Target wins: **unmeasured**. Overall/component deltas: **unmeasured**. The kill premise has not been evaluated, so no killed outcome is claimed. The branch must not be treated as a trained correction or a completed experiment while the model remains disabled.
