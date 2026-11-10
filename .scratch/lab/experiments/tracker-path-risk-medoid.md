# Posterior position-sequence risk at local run decisions

Status: needs-human; Modal infrastructure prevents training and validation. The prototype is committed, but its channel is inactive. This is neither a measured kill nor a recommendation to fold.

Experiment: tracker-path-risk-medoid, tracker, explore. Branch: exp/tracker-path-risk-medoid. The experiment card was not edited.

## Prediction

The card predicts tracker -1 and overall 0: recover the rhyme restart's tracker false warning, with one nearby listener false warning potentially exposed. The supplied target is `lab-v1.5-p46/phone/warning:skipped_passage/54:36:1/121.84`.

The premise to validate is strict: over the target's complete retained comparison interval, the near continuation must have lower posterior expected position-sequence edit risk than the committed restart. The supplied earlier audits report a 0.3915-edit acoustic disadvantage for the retained near continuation; that number was not independently reproduced here.

## Prototype

`packages/core/src/path-risk.ts` implements a normalized token edit channel and exact weighted Levenshtein risk over ordered reference-word position sequences. Identical sequences merge by summing their probability before normalization. Adjacent repeated occurrences remain separate entries when an explicit jump starts another reading of the same word. The selected sequence is an existing trajectory, rather than a synthesized sequence of independent word decisions.

`packages/core/src/local.ts` retains a map of complete position histories at each cell alongside the original acoustic DP. It extends every existing insertion, match, deletion and eligible jump edge, including paths dominated by another history's acoustic cost at the same endpoint. Equivalent histories at a cell sum their channel mass and keep the cheapest existing acoustic representative for event metadata. Retained sequences cover all observations since the local search began; recentering moves the endpoint window without truncating surviving histories. Save/restore shares immutable prior maps.

The existing acoustic recurrence still defines jump sources, the nearby window, recent cost and step cost. When that recurrence proposes a different run origin, the event-facing position may instead come from the posterior sequence medoid. Acoustic prices, confirmation thresholds, event dispatch and checker rules are unchanged. There are no session, reader or verse tests in engine code.

`packages/sdk/src/path-channel-fit.ts` decodes only scored, none-labelled help-train rows in the train split, excluding speakers having any non-train row, exclusions, scripted events and repeat annotations. It uses the deployed listener model and records its hash and the eligible-label hash. Existing directed weighted edits provide training alignments, retaining expanded duration symbols. Jeffreys-smoothed operation probabilities and conditional identity distributions are normalized separately. No hyperparameters are selected on practice. Movement potentials use the existing jump prices and the existing three-nats-per-edit conversion; posterior mass is normalized over retained trajectories on the same interval.

The fitter produces `path-channel.ts` and `fit-report.json` in LAB_OUT. No fit was obtained: `packages/core/src/path-channel.ts` deliberately contains `PATH_CHANNEL = null`. The tracker uses the original acoustic position decisions in this state. This placeholder must be replaced with a successful training-only fit before evaluating the experiment's premise or product effect.

## Validation obtained

Dependency-free Node 22 smoke checks passed on the actual TypeScript source using Node's type stripping and an in-memory relative-module resolver. They checked a medoid different from the MAP outlier, merged posterior mass, repeat edit distance, normalized training probabilities, retention of dominated histories at a shared endpoint, explicit same-position repeat occurrences, and immutable save/restore replay. `git diff --check` passed.

`packages/core/test/path-risk.test.ts` adds nine unit cases for those behaviors, concentrated genuine-jump evidence, log-scale stability, invalid training input and inactive-channel equivalence. These Vitest tests and the existing core/SDK suites have not passed a Modal run here; local smoke checks are not a substitute for the required build and suites.

## Infrastructure results and limits

Three supervisor Modal requests returned `NotFoundError: No such file or directory`, each reporting zero compute spend and no job output:

- Request 1: training metadata and the provided target trace command.
- Request 2: the six-shard `bash tools/autolab_tests.sh` validation command.
- Request 3: six-shard validation plus the training fitter on shard zero.

The error does not establish that storage is full, credentials are missing, or any particular data path is absent. Its underlying cause is unresolved. No Modal log or fit report came back to inspect. Final history-retention refinements therefore have only the local smoke validation described above.

No practice-score call was made (0 of 3 used). Target wins, component change, overall change, studio behavior, clean-session false warnings and reader robustness are all unmeasured. The kill criterion cannot be evaluated, and a failed infrastructure request is not evidence of zero target wins.

Exact history enumeration can grow rapidly with the number of observations and distinct repeat/jump explanations; merging identical sequences alone does not provide a bound on runtime or memory. Long-stream feasibility remains unvalidated. No pruning, shorter evidence interval or sampled risk was introduced to conceal that limitation. These scaling checks, training and target risk measurements are prerequisites to activating the prototype.

Once Modal is repaired, build and run the final core/SDK suites, run `pnpm --filter @tilawa/sdk exec tsx src/path-channel-fit.ts` on the authorized training data, import the generated channel, and measure the target's complete-interval risks and product warnings. If the premise fails or the built change wins fewer than three targets, follow the requested kill procedure; otherwise the supervisor's practice score and gate decide. No fold or discard was performed here.
