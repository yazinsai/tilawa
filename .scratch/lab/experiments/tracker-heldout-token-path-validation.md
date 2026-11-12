# Validate a local restart on withheld token identities

Status: needs-human. The prototype, training-only fitter and focused checks are implemented.
The supplied supervisor runner fails before executing commands with
`NotFoundError: No such file or directory`. No training calibration or target replay is
available, so no premise kill or practice result is claimed. The optional validator is
not enabled by default without that calibration. `lab-experiment.json` is unchanged.

## Prediction

Tracker −1, overall 0 on lab-v1.5u practice, using the card's deployed listener.
The target is `lab-v1.5-p46/phone/warning:skipped_passage/54:36:1/121.84`.
The near continuation's reported 0.3915-edit deficit is prior evidence from the brief,
not a measurement reproduced here. A better binding may expose the documented nearby
listener wrong-word warning; checker rules are unchanged.

## Mechanism

`packages/core/src/heldout-token.ts` implements a normalized token-edit channel with
per-written-phone distributions over emitted identities and deletion, a shared insertion
probability and an insertion-identity distribution. Existing duration discounts retain
their relative three-nats-per-edit weights. Identity distributions are summed when a
token is masked, including identity-dependent held-sound insertion weights: a masked
token still consumes its original observation position.

Local search retains immutable pre-step rows over the existing 16-token recent-cost
horizon. At an original weighted winner that changes run origin, the prototype divides
that interval into alternating contiguous blocks. It seeds both complementary searches
from the same row preceding the interval, preserving accumulated earlier costs. The
reference windows and word-start movement eligibility are those of the existing search.

Each retained-only Viterbi search selects a continuation and a restart location and
infers its own edit boundaries and best-state jump sources. The continuation family is
kept separately so a cheaper restart representative cannot erase it during selection.
The proposed restart is not inserted as a forced winner: it must be selected from the
retained-only restart search in each direction.

For each retained-selected location, a forward sum marginalizes edit boundaries and
run-origin histories over that search's frozen movement graph. Predictive evidence is
`log Z(full identities, selected location) − log Z(retained identities, selected location)`.
Both partitions use identical token positions, reference endpoints, movement sources
and transition weights. Movement odds retain the existing jump, distance and surah-entry
prices; outgoing alternatives are normalized rather than rewarded for path count.

The validator retains the original proposed restart only if its selected origin agrees
with that proposal and its held-out predictive advantage over the continuation is
strictly positive in both complementary directions. Otherwise local search chooses its
best retained original-cost continuation cell. If no such original cell exists, it cannot
construct a new one and leaves the original winner alone. Empty partitions cannot
certify a restart. The original costs, movement edges, commitment thresholds, comparison
timing, event interface and checker confirmation rules are retained.

`TrackerOptions.localTokenValidation` passes the same frozen model to each local search.
The search owns its model arrays. `audit: true` reports both predictive directions while
preserving the original weighted winner. Rewinding restores the comparison history as
well as the original DP state. With no supplied calibration, the default tracking path
is unchanged; no synthetic or practice-fitted model has been installed.

## Training-only calibration

`packages/sdk/src/heldout-token-training.ts` admits only scored, none-labelled takes
from training speakers, with no events, repeats or exclusions. Any speaker appearing
in a non-train label is excluded altogether. Speaker folds are deterministic and
independent of recording order.

`packages/sdk/src/heldout-token-fit.ts` decodes only those help-train recordings with the
deployed `models/spliced-clean-int8` listener and fits smoothed directed edit counts.
Smoothing strengths 1, 4, 16 and 64 and block lengths 1, 2, 4 and 8 are selected together
using up to five speaker-grouped training folds. The selection objective is the mean
fold predictive log likelihood per withheld identity, exchanging both partitions.
Predictive evaluation sums latent edit alignments; training count estimation itself
conditions on a directed minimum-edit alignment. A final refit, model/label hashes and
fold report are written to `LAB_OUT` before any practice case is opened.

## Validation

Fourteen focused test bodies passed locally through Node 22's TypeScript stripping and
an assertion adapter. This is a numerical and synthetic result, not a Vitest suite pass
or a TypeScript type check. The committed adapter can reproduce it:

```sh
node --disable-warning=ExperimentalWarning .scratch/lab/token-test-local/run.mjs
```

The checks establish:

- Complementary contiguous blocks preserve all observation positions.
- Summing every possible withheld identity equals the retained-evidence partition,
  including duration-discounted insertions; conditional probabilities sum to one.
- Changing withheld identities does not change retained-only selected locations.
- A synthetic destination explaining both partitions is retained; a destination
  fitting only one complementary partition is rejected.
- Ordinary synthetic reading and audit mode retain original local positions and costs.
- A rejected restart continues the existing local path; rewind reproduces its decisions.
- Calibration arrays are owned, history is bounded at 16 tokens, and directed training
  counts distinguish insertion and deletion.
- Mistakes, repeats and held-out speakers are excluded, with deterministic speaker folds.

The edited TypeScript sources and replay helper also parsed successfully with Node's
TypeScript stripper. No scorer, lab tool, manifest, reference asset, labels or stored
evidence were edited.

The supervisor requests were data/schema inspection (run 1), six-shard required builds
and core/SDK tests (run 2), builds plus training-only calibration (run 3), and the required
builds/tests again after implementation (run 4). All four failed before execution with
the same `NotFoundError`; no data, build, training or test logs were returned.

Practice-score requests: **0 of 3**. There is no measured target-win count, tracker delta
or overall delta. Synthetic predictive agreement is not a target result.

## Limits and recovery

The comparison uses the existing recent-cost horizon, not the complete session prefix;
the initial row preserves the earlier accumulated costs. Its earlier histories remain
original weighted representatives. Reference-window geometry is conditioned on the
original search's retained windows, while jump sources and edit boundaries inside those
windows are inferred on retained identities. Forward sums are exact within each frozen
retained-selected movement graph, not across every graph induced by alternative search
sources. Real-session runtime and memory have not been measured.

Training geometry selection uses whole single-ayah recordings, whereas runtime uses
the existing recent-cost interval. Training alignment counts are point estimates, and
the phone rows pool through the existing phonetic edit prior rather than a learned
phonetic-feature hierarchy. These restrictions limit any eventual negative result.

Restore the supplied supervisor runner, then obtain calibration and actual suite results:

```sh
python3 /Users/rock/ai/projects/tilawa/tools/autolab.py modal-run --timeout 30 -- \
  'pnpm --filter @tilawa/core... --filter @tilawa/sdk... build && TOKEN_VALIDATION_CACHE=/scratch/tracker-heldout-token-path-validation/clean-decodes pnpm --filter @tilawa/sdk exec tsx src/heldout-token-fit.ts'
python3 /Users/rock/ai/projects/tilawa/tools/autolab.py modal-run --shards 6 -- \
  'bash tools/autolab_tests.sh'
```

Copy the returned training-frozen model into the worktree as calibration output.
The read-only replay helper takes case, model and mode as arguments, uses existing
evidence chunking and the unchanged checker, and never calls the scorer:

```sh
python3 /Users/rock/ai/projects/tilawa/tools/autolab.py modal-run -- \
  'pnpm --filter @tilawa/core exec tsx /app/.scratch/lab/heldout-token-target-audit.ts --set lab-v1.5u --session lab-v1.5-p46 --model /app/.scratch/lab/heldout-token-model.json --mode audit'
```

Replay `original` and `validate` with the same model to inspect actual warnings.
Inspect both retained-selected locations and predictive gains at the target restart.
If the restart wins predictively in both directions, the stated premise is falsified.
If the built change removes fewer than three target warnings, follow the user's kill
rule: revert engine changes, retain the negative write-up and write a killed outcome.
If enough target warnings are removed, install the training-frozen default and request
a practice score before any gate-dependent narrowing. The single listed target cannot
independently supply the required three wins.

Runner failure does not establish a premise kill. The prototype is left for validation,
with `.autolab/outcome.json` recording `needs-human` and no invented measurements.
