# Minimax regret for existing local trajectories

Status: needs-human. The prototype and training estimator are implemented; the supplied
Modal runner fails before executing commands. No uncertainty set has been fitted and
the deployed default remains inactive. This is an infrastructure result, not a premise kill.

## Prediction

Tracker −1 wrong call, overall 0 on lab-v1.5u practice: remove
`lab-v1.5-p46/phone/warning:skipped_passage/54:36:1/121.84`, potentially exposing the
documented nearby listener wrong-word warning. The supplied audits report the near
continuation 0.3915 edits behind the rhyme restart. That figure is prior evidence from
the brief, not a measurement reproduced here. `lab-experiment.json` is unchanged.

## Mechanism

For each existing local DP representative, `packages/core/src/local.ts` carries a complete
cost witness: counts of nine directed acoustic edit categories, fixed movement charges,
and the number of consumed observations. A jump resets the current run origin but does
not reset this witness. Insertions consume observations, reference deletions do not, and
recentring preserves the witnesses of retained cells. The witness describes the interval
since this local search was initialized; initial global-lock evidence is a common prefix
outside that interval. The witnesses reconstruct the original accumulated costs exactly
under the original category prices, within floating-point roundoff.

When the weighted row winner would change run origin, `path-regret.ts` evaluates all
finite retained representatives under each fitted model, including their unchanged jump,
distance and surah-entry charges. For path p and model m, regret is
`C_m(p) - min_q C_m(q)`. Selection minimizes the maximum regret across models and keeps
the original acoustic winner on an objective tie. The output is an existing trajectory's
offset, origin and edit metadata. The original recurrence, movement edges, endpoint
semantics, recent-cost accounting, confirmation rules, global commitment margin and
checker remain in use. This adds neither a skip veto nor posterior averaging.

`TrackerOptions.localPathRegret` accepts a frozen set and an optional comparison callback.
Its audit mode preserves original decisions while returning raw costs, counts, fixed
charges, per-model costs, regrets and the robust selection. The tracker uses
`PATH_UNCERTAINTY` by default after successful calibration; that constant is currently
null because no training output was obtained. No fabricated or practice-selected costs
were installed.

## Training-only uncertainty construction

`packages/sdk/src/path-regret-fit.ts` opens only help-train labels and the supplied
deployed listener. Eligibility requires a scored, none-labelled train row with no
exclusion, planted event or repeat annotation. Any speaker appearing in a non-train row
is excluded entirely. Decodes are cached by listener hash and recording-ID hash.

Full-transcript alignments use the existing directed weighted edit prices, including
terminal deletions. The acoustic categories are hard, soft and vowel substitutions;
hard, vowel and duration insertions; and hard, vowel and duration deletions. Exact matches
cost zero. Each category's fitted price is the negative log of its Jeffreys-smoothed
error-versus-exact-match probability. These are compact cost models, not a complete
normalized generative model of all possible phone strings.

Eligible speakers form up to five deterministic disjoint partitions, with every recording
of a speaker in one partition. Each partition produces a whole correlated cost vector;
the pooled fit produces one additional vector. All are retained, giving 100% empirical
coverage of these partition estimates. The uncertainty set is their convex hull: worst
regret occurs at one of the retained vertices, so no independent coordinate extrema are
invented. This empirical coverage does not claim a statistical confidence level for new
speakers. Partition geometry and coverage are fixed before practice inspection.

One shared nats-to-UNIT conversion minimizes squared error between pooled-model training
path totals and those paths' original acoustic totals. That conversion is fitted entirely
on training data and applied unchanged to every partition vector; movement prices and
commitment margins are not refitted. The estimator writes `path-uncertainty.json`, the
installable `path-uncertainty.ts`, and a report with eligibility counts, listener and label
hashes, partition counts, coverage and unit conversion. No such output was obtained here.

## Validation obtained

Seventeen new unit-test bodies passed using Node 22 TypeScript stripping, a relative-module
resolver and an in-memory assertion adapter for the Vitest imports. This executed the
actual test bodies, but is not a Vitest suite pass or a TypeScript build:

- Robust selection differing from averaging; per-model best-path subtraction; correlated
  vertices; a uniformly preferred path; movement charges; common-prefix invariance;
  original-choice ties; rejection of unequal observation horizons and invalid models.
- Directed edit categories, explicit duration indels, full-transcript terminal deletions,
  positive smoothed prices and the shared training unit conversion.
- Original/audit position equivalence, complete raw-cost reconstruction through a jump,
  selection of the chosen trajectory's metadata, rewind/replay, owned input vectors and
  callback witnesses, and exact in-order progress under the original singleton model.
- Exclusion of mistakes, repeats, held-out speakers and unscored takes; deterministic
  speaker partitions independent of input order.

`git diff --check` passed. No scorer, lab tooling, manifest, training labels or evidence
were modified.

## Infrastructure and remaining validation

The first two supervisor requests failed before execution with
`NotFoundError: No such file or directory`: a read-only authorized training-schema
inspection and the required six-shard `bash tools/autolab_tests.sh`. The returned error
record reports zero compute spend; no data or job logs were returned. A third request
retried the required tests and requested the training fitter on shard zero after tests;
it failed before execution with the same error. All three requests failed, and no actual
build, suite or calibration result was returned.

The failure does not identify its underlying cause. No alternative Modal access, external
data location or release-held-out data was sought. No practice-score call was made
(0 of 3). Target wins, the target's per-model preferences, component/overall changes,
clean-session warnings and studio behavior remain unmeasured.

After restoring the supplied runner, obtain the actual build, test and training results:

```sh
python3 /Users/rock/ai/projects/tilawa/tools/autolab.py modal-run --shards 6 --timeout 30 -- \
  'bash tools/autolab_tests.sh && if [ "$LAB_SHARD" = 0 ]; then PATH_REGRET_CACHE=/scratch/tracker-minimax-path-regret/training-decodes pnpm --filter @tilawa/sdk exec tsx src/path-regret-fit.ts; fi'
```

Inspect the returned report before installing its generated constant. Replay the supplied
target with `localPathRegret` in audit mode first, using the returned JSON set and an
`onComparison` callback near the target time. Confirm equal observation counts and compare
the restart and retained near continuation's per-model excess costs. If the restart remains
minimum-regret, the card's premise fails. Otherwise replay minimax mode with the unchanged
checker and measure actual target warnings, including the accepted risk of the nearby
listener warning. The provided standard trace CLI can inspect events after installing the
training-frozen default.

The implementation reranks existing weighted representatives at each cell. It does not
retain every edit alignment dominated by the original weighted recurrence, nor refit
alignments separately for every model. Its regret calculation is exact for that available
set, not for an exhaustive trajectory space. Training also conditions on minimum-cost
alignments. Real-stream runtime, memory, training-model support and target behavior remain
unvalidated. A negative result would therefore concern this compact set and these existing
representatives, not every possible robust acoustic model.

If the premise fails or the built change wins fewer than three targets, revert the engine
prototype and record a killed result as instructed. A synthetic reversal supplies no target
win. If three or more targets are won, request a practice score before deciding on any
narrowing. No fold or discard has been performed here.
