# Coupled shared-phone assignments at a local restart

Status: engine implementation retained; target premise, target wins and full validation are blocked by the supervisor's Modal launcher. Not killed and not scored.

## Prediction

Tracker -1 wrong call, overall +0 on lab-v1.5u practice. Correcting the binding may expose the documented nearby listener false warning. Target: `lab-v1.5-p46/phone/warning:skipped_passage/54:36:1/121.84`. `lab-experiment.json` is unchanged.

## Implemented mechanism

`tracker.ts` now supplies a paired comparison to `createLocalSearch`. `local.ts` retains immutable predecessors carrying the original reference cursor, observed frontier, directed edit, movement entry price and cumulative cost. Save/restore and reference recentering retain those predecessors. When the ordinary optimum proposes a new run origin, the best two distinct run origins compete through the paired comparison. Adjacent endpoints of the same origin do not compete with each other.

The tracker reconstructs both suffixes at their common predecessor, verifies that they consume the same complete observed string, and preserves the shared prefix cost. `coupled.ts` builds the common pronunciation correspondence from every distinct maximum common subsequence of reference-phone indices. It enumerates index-pair sequences rather than deletion routes, so equivalent correspondence routes do not receive duplicate weight.

For each correspondence, the paired edit graph requires corresponding shared phones to consume the same observation, or be deleted at the same observation frontier. Each intervening differing span keeps its own exact edit path on that observed interval. Edit prices use the original reference context, retaining duration-run prices. Entry prices remain attached to their original phone, and a jump's first phone still has to consume an observation. The joint minimum uses the sum of the two costs, symmetrically. Equal joint-cost bindings get equal normalized weight; distinct reference correspondences get equal normalized weight. Candidate scores are the negative log of their mean edit likelihood, converted using the existing `UNIT`.

Already identical shared assignments and pronunciations with no shared phones return their original costs exactly. The paired scores change cursor selection at a restart, while the original edit recurrence and recent-cost accounting remain intact. Movement constants, the two-ayah reference window, the retained observed comparison interval and the tracker's commitment rules remain unchanged. Checker and listener code are untouched.

## Validation

Eight small synthetic fixture checks passed using Node's type stripping and an assertion harness over temporary source copies, without installing dependencies or running heavy work on the Mac:

- Identical shared assignments return exact original costs, including an entry price.
- Differing spans retain their separate edit costs and swapping operands swaps scores.
- An expanded four-noon run has all four correspondences, with no free match credit from their multiplicity.
- Unrelated pronunciations remain unchanged.
- Incompatible shared-token assignments charge both candidates symmetrically.
- Binding alone can reverse a synthetic preference under existing short-vowel versus consonant edit prices, while preserving a private entry charge.
- A neutral comparison with provenance preserves the original recurrence over a synthetic repeated word.
- Rewinding paired histories reproduces subsequent positions and costs.

These are fixture checks, not a claim that Vitest, the builds or the complete core/SDK suites passed. Temporary source copies and diagnostics were removed from the final diff.

All external execution used the authorized supervisor command:

- Run 4, minimal `pwd`: supervisor `NotFoundError: No such file or directory`, no command log.
- Run 5, six-shard `bash tools/autolab_tests.sh`: same launcher failure, no build or test output.
- Run 6, target paired traceback and product replay: same launcher failure, no evidence or result. The prepared audit compared a neutral instrumented tracker against the original tracker event by event and was intended to record the shared reference/observation assignments around the target plus the changed product's checker events. It never executed.
- Run 7, six-shard validation of the later implementation and seven initial unit tests: the same supervisor launcher failure, with no build or test output. The eighth synthetic preference test and final arithmetic/selection cleanup were added afterward; they passed the same local fixture harness.

No practice scoring request was made. All three allowed practice scores remain unused.

## Limits and next decision

The target's shared-phone assignments remain unknown. The reference's expanded noon creates correspondence ambiguity, but that alone does not prove that independently repositioned shared sounds cause the target restart. Neither a target win nor zero target wins is established, so an infrastructure error cannot justify a killed outcome.

The graph re-aligns the two retained trajectories' reference and movement sequences; it does not introduce new movement routes. Full-suite compatibility, real-session latency and memory are unverified. Exact correspondence enumeration can be expensive for long, repetitive divergent histories, and predecessor retention adds memory; those costs require Modal validation before any claim of practical streaming performance.

Once the supervisor can launch jobs, reconstruct the baseline target pair, check whether its shared assignments differ, run the built paired product and inspect the target warning. Kill if the premise is absent or the built change wins fewer than three targets, reverting engine/test changes and recording the measured reason. Otherwise run practice scoring under the user's three-score limit and let the supervisor's fold gate decide. No scorer, lab tooling, manifests, test data or evidence was edited, and no fold or discard was performed.
