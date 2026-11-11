# Acoustic word-boundary evidence for local tracking

Status: blocked on supervisor infrastructure. Implementation and training pipeline drafted; no trained model or measured experiment result.

## Prediction

Tracker −1, overall 0. The proposed recovery is the phone false skipped-passage warning at `lab-v1.5-p46/phone/warning:skipped_passage/54:36:1/121.84`. Correct binding may expose the previously documented listener false warning nearby. The experiment card is unchanged.

## Mechanism

`trackerInput` currently discards posterior rows. The draft preserves a separate tracker-only posterior channel, leaving acquisition's decoded scoring intact. A compact logistic scorer observes the previous CTC spike's tail, the gap, and the next spike's head. Features summarize blank probability, posterior entropy, nonblank peaks, and overlap/change between the two posterior distributions, together with gap duration. There are no reader, session, reference-location or word-identity features.

The estimator uses only help-train training speakers' none-labelled recordings without scripted events or repeats. If any label places a speaker outside training, that speaker is excluded entirely. The unchanged primary streaming listener supplies trajectories; no tracker visits supply training labels. CTC forward–backward marginalizes transcript-consistent boundaries, including connected/terminal isolated pronunciation and pronunciation-length variants, and projects the latent next-word entry distribution onto observed spike seams. The resulting fractional labels train the scorer. Speaker partitions select regularization and calibrate probabilities before a final fit on the authorized training recordings. Practice data cannot enter this estimator.

The local-search draft carries acoustically consumed word ownership through reference deletions. Proposed word entries and restart entries receive the boundary likelihood; within-word continuations receive the within likelihood. Both divide out the training boundary prior and use the existing three-nats-per-edit likelihood scale. Each consumed observation contributes once; reference-only deletions carry the existing observation contribution. Movement prices, available jumps, commit margins and checker rules are unchanged. Posterior retention is bounded; unavailable old rows provide no seam evidence.

No synthetic or neutral model is shipped. The scorer is configurable through `TrackerOptions.boundaryModel`, and remains unconfigured by default. The estimator's generated model must be installed, measured against the competing target patterns, and wired as the product default before this becomes the card's completed experiment. No practice result from this unconfigured draft would measure the hypothesis.

## Validation so far

- Read `AGENTS.md`, the card, tracker/local search, posterior handling and the existing component-replay contract. No card, scorer, lab tooling, data, manifest or recorded evidence was edited.
- Added unit cases for pooled-likelihood normalization, posterior features, an adjacent-spike seam with no invented blank, boundary-supported versus unsupported restarts, and restoration after a rewind. Added a tiny exhaustive CTC-path comparison for the training marginal.
- Python syntax compilation and `git diff --check` passed. The local mathematical unit run could not import NumPy because its installed extension has the wrong architecture; no local environment changes were made.
- Supervisor runs 1 (help-train inventory and target evidence discovery), 2 (minimal environment/data-access retry), and 3 (focused TypeScript unit tests) all returned `NotFoundError: No such file or directory`, without command logs or outputs. Each response reports zero compute spend. Run 4 requests the required full core/SDK build and tests on six shards; its result is pending.
- No help-train recording or target audio/evidence has been obtained. No model has been fitted, no target comparison has been measured, and no practice score has been run (0 of 3).

## Required next measurements

Obtain the authorized training labels and primary-listener dumps through the supervisor, run the estimator and install its frozen model. Verify the CTC marginal tests and both package suites on Modal. Over the target's identical heard interval, compare the fitted boundary-pattern likelihoods of the near continuation and the original winning restart. If the near pattern is not more likely, revert the engine draft and record a premise kill. Otherwise replay the target and count actual target wins; fewer than three wins also triggers the supplied stop rule. Three or more wins require a practice score before a gate decision.

Remote infrastructure failure supplies no evidence for or against the acoustic premise and must not be reported as zero measured wins or as a killed model.
