# Marginalize uncertainty in training edit parameters

Status: needs-human. The prototype and focused checks are implemented, but the supplied
supervisor runner fails before executing commands. No fitted posterior, target replay,
TypeScript build or full test-suite result is available. The default engine remains on
its original channel until training calibration supplies a posterior.

## Prediction

Tracker −1 wrong call, overall 0 on lab-v1.5u practice. The target is
`lab-v1.5-p46/phone/warning:skipped_passage/54:36:1/121.84`.
Correcting the binding may expose the documented nearby listener false wrong-word
warning. Checker handling is unchanged; this experiment does not claim a standalone fold.

The supplied audits report the retained near continuation 0.3915 edits behind the rhyme
restart. That number is prior evidence from the brief, not a measurement reproduced here.
`lab-experiment.json` is unchanged.

## Mechanism and implementation

`packages/core/src/edit-parameter.ts` fits a compact empirical-Bayes token-edit channel.
Each written phone has a Dirichlet posterior over exact recognition, substitution and
deletion. A shared Beta posterior governs insertion versus advancing the reference.
Conditional substituted and inserted phone identities are normalized, frozen nuisance
distributions. The pooling mean comes from the supplied training partition, and the
prior strength controls how strongly sparse reference-phone rows pool toward it.

An ordered trajectory carries the sufficient counts for those posteriors. Its predictive
probability uses `B(alpha + counts) / B(alpha)` for each row, calculated with exact integer
rising factorials. There is no multinomial counting coefficient: this scores one ordered
edit path. This integrates one shared vector of channel parameters over the complete
trajectory. The control arm instead multiplies probabilities at that same posterior's
mean. The two arms share every observation, emission-identity distribution, movement
charge and duration charge; their only difference is parameter integration.

`packages/core/src/local.ts` carries these statistics beside each existing weighted
alignment representative. Insertions, substitutions, deletions and jump transitions
copy their predecessor's whole statistic. A jump changes run origin but does not erase
earlier evidence. Duration indels retain the existing 0.2-edit cost and do not teach
ordinary acoustic deletion or insertion rates. Jump, jump-distance and surah-entry costs
remain fixed and convert at the existing three nats per edit. Terminal endpoints and
available movement edges remain as before.

When the original weighted winner would change run origin, the optional channel compares
all finite retained complete trajectories over the same heard interval. `audit` records
both likelihood arms and preserves the original winner; `mean` and `marginal` select an
existing trajectory using the corresponding arm. Cost normalization, recent weighted
misfit accounting, save/restore and the event layer continue through that selected cell.
The tracker exposes this through `TrackerOptions.localEditParameters`, passing the same
training posterior to each local search. Each search owns a copy of the posterior, and
audit snapshots own their counts. No adaptation to the current reader occurs.

No calibrated default has been installed. Supplying a placeholder channel would not test
the card's hypothesis. Without the option, original candidate scores and decisions remain
unchanged. No checker, scorer, lab tool, manifest, training label or evidence file was edited.

## Training-only estimation

`packages/sdk/src/edit-parameter-fit.ts` opens only help-train and the supplied deployed
`models/spliced-clean-int8` listener. Eligibility requires `split: train`, `kind: none`,
`scored: true`, no exclusion, no labelled mistakes and no labelled repeats. Any speaker
appearing in a non-train label is excluded altogether. Decodes are cached under the
experiment's scratch directory, keyed by model hash and recording ID.

Prior strengths 1, 4, 16, 64 and 256 are compared in up to five deterministic speaker
folds. Each candidate is fitted entirely on the other training speakers and evaluated
by complete-recording posterior-predictive likelihood on that fold. Selection and the
final refit occur before any practice case is opened. The estimator writes a posterior
and a report containing model/label hashes, eligibility, clip/speaker counts and fold
likelihoods to `LAB_OUT`. No such output was returned in this attempt.

## Validation and limits

Eleven focused test bodies passed locally with Node 22's TypeScript stripping and a small
assertion adapter for Vitest imports. This is a numerical/synthetic check, not a Vitest
suite pass or a TypeScript type check. These checks cover:

- One shared draw versus multiplying per-token means, including normalized probabilities
  for an exhaustive small set of ordered paths.
- Convergence toward point estimates for concentrated posteriors, and a synthetic reversal
  relative to the identical posterior-mean channel.
- Fixed duration/movement charges, directed edit classification and normalized identity rows.
- Complete prefix accounting through a restart, byte-identical original/audit positions,
  rewind/replay, snapshot ownership and frozen input parameters.
- Selection of an existing trajectory in either likelihood mode.
- Exclusion of mistakes, repeats and held-out speakers, and reproducible speaker grouping.

The supplied supervisor requests were:

| Run | Requested work | Result |
| --- | --- | --- |
| 1 | Inspect authorized data/schema and deployed model files | Failed before execution: `NotFoundError: No such file or directory` |
| 2 | Core/SDK build followed by training-only calibration | Same pre-execution failure; no fitted model or training measurements |
| 3 | Required core/SDK builds and tests on six shards | Same pre-execution failure; no test logs |
| 4 | Required builds/tests again after completing trajectory hooks and tests | Same pre-execution failure; no build or test logs |

The local matcher still retains the original weighted representative at each cell; this
prototype reranks those complete trajectories rather than doing an exact Bayesian search
over every possible alignment/history. Its predictive integrals are exact conditional on
those paths. Training likewise conditions on existing directed weighted alignments rather
than marginalizing uncertain training alignments. Pooling means and nuisance identity
distributions are point estimates. These restrictions isolate parameter uncertainty, but
they limit what a negative result would establish about more general Bayesian decoders.
Runtime and memory with real session lengths have not been measured.

No practice-score requests were made (0 of 3). There is no measured target-win count,
target preference reversal or overall delta. A synthetic reversal is not a target win.
The runner failure supplies no evidence for a premise kill, so this is needs-human rather
than killed. The single supplied target cannot independently meet the user's three-win
threshold; any eventual measured result must still follow that rule.

## Recovery and remaining work

Restore the supplied supervisor runner, then obtain an actual calibration and full test
result:

```sh
python3 /Users/rock/ai/projects/tilawa/tools/autolab.py modal-run --timeout 30 -- \
  'pnpm --filter @tilawa/core... --filter @tilawa/sdk... build && EDIT_PARAMETER_CACHE=/scratch/tracker-edit-parameter-marginal/clean-decodes pnpm --filter @tilawa/sdk exec tsx src/edit-parameter-fit.ts'
python3 /Users/rock/ai/projects/tilawa/tools/autolab.py modal-run --shards 6 -- \
  'bash tools/autolab_tests.sh'
```

Copy the returned posterior into this worktree as the experiment's calibration output.
The read-only replay aid `.scratch/lab/edit-parameter-target-audit.ts` takes the posterior,
case and comparison mode as arguments. It uses the existing evidence chunking and original
checker, and writes comparisons and events without calling the scorer. Run `audit` first:

```sh
python3 /Users/rock/ai/projects/tilawa/tools/autolab.py modal-run -- \
  'pnpm --filter @tilawa/core exec tsx /app/.scratch/lab/edit-parameter-target-audit.ts --set lab-v1.5u --session lab-v1.5-p46 --at 121.84 --model /app/.scratch/lab/edit-parameter-model.json --mode audit'
```

At the target comparison, inspect the complete near-continuation and restart trajectories,
including their common observation count, original raw cost, fixed charges and both channel
likelihoods. The premise requires integration to reverse their preference relative to the
identical posterior-mean channel. If it does, replay `mean` and `marginal` to examine the
actual target warning and the accepted nearby listener risk. A failed reversal or fewer
than three measured target wins requires reverting engine changes and recording a killed
result. If enough targets are won, install the training-frozen posterior as the default and
request a practice score before any gate-dependent narrowing. No calibration or decision
may be tuned against practice results.
