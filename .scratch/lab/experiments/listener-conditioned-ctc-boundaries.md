# Conditioned CTC boundaries

Status: implemented; target and required Modal validation blocked by supervisor infrastructure. No practice score was run. The experiment card is unchanged.

## Prediction and mechanism

The card predicts one fewer listener false warning and one fewer overall wrong
call, with no predicted losses. The task separately requires at least three of
the four registered targets to be won before practice scoring; that continuation
requirement has not been measured.

`PosteriorFrames.nll()` initializes each extracted comparison with a fresh CTC
blank/first-label state. The SDK recheck computes a normalized forward boundary
distribution from the unchanged primary pronunciation preceding the comparison,
and normalized backward potentials from the unchanged primary pronunciation
following it. The exterior pronunciation is the retained primary token path,
identical under both hypotheses, rather than an assumption that all surrounding
words were canonical. Exterior rows and existing comparison coordinates remain
fixed. An already-active label can continue, but entering a new articulation of
the same label requires an intervening blank. Every phone of both complete
written and accused pronunciations must still be emitted inside the original
comparison interval.

Both hypotheses use the same exterior potentials. The listener vetoes only when
the original heard-over-written preference is positive and the conditioned
comparison strictly reverses toward the written pronunciation. Missing exterior
state, missing rows, unknown labels and impossible comparisons abstain. No model,
pronunciation inventory, timing constant or confirmation bar changes.

Core exposes the existing non-graph comparison after its usual support bar
clears. The SDK supplies the recheck during live judgments. Its read-only replay
also transports reversal certificates on a primary token for stored-evidence
evaluation. Certificate consumption requires the exact word occurrence, frame
interval and both complete pronunciations. Parallel-phrase confirmations and
other error classes bypass the recheck. Primary phones, confidence, posterior
rows and tracker events retain their existing values. No new asset bundle or
`.autolab/model.json` is needed.

## Validation

Added SDK unit tests compare the scorer with independent exhaustive CTC paths,
exercise an actual fresh-entry preference reversal, verify blank-reset behavior,
and require complete repeated articulations. Core tests exercise online vetoes
and exact certificate matching; SDK transport tests cover JSON serialization,
native frame cells and preservation of primary values.

Small local checks used Node 22's TypeScript transformation, without installing
dependencies or running acoustic inference. With the real `PosteriorFrames`, 24
exhaustive path comparisons across eight small posterior tables agreed within
1e-11, including repeated target labels and labels shared with either exterior.
The blank-reset comparison agreed within 1e-4 (the existing posterior floor
prevents exact zeros). A constructed fresh-entry preference of +0.9808 nats
reversed to approximately -2.5055 nats under conditioning. These are mathematical
checks, not practice incidents or evidence of target recovery.

A synthetic core substitution produced one confirmed warning. Artificial exact
reversal certificates removed it; shifting the certificate's start frame left
the warning intact. An online veto removed it, and the ordinary path retained
it. This validates certificate consumption, not acoustic discrimination of real
substitutions. JSON/cell transport preserved every primary phone and posterior
value. `git diff --check` passed.

## Blocked measurements and limits

All four supervisor Modal requests returned exactly
`NotFoundError: No such file or directory`, with no command output and reported
compute spend of zero:

1. Listing the registered target evidence paths.
2. Core/SDK builds, focused scorer tests, then the target probe.
3. Retrying those checks with a non-ignored diagnostic source path.
4. The prescribed six-shard `bash tools/autolab_tests.sh` harness.

The target probe was prepared to replay the card's declared listener evidence
through the actual free tracker/checker, capture complete engine-requested
comparisons, and compare original and vetoed target warnings. It did not run.
The infrastructure failure is before a useful diagnostic, so it neither proves
missing target evidence nor falsifies the boundary-state premise. The user was
asked to restore the configured supervisor runner. No alternate Modal access or
files outside the authorized worktree/data paths were sought.

Target recoveries, the full TypeScript build, Vitest suites, population losses,
clean-session behavior, studio behavior and performance remain unvalidated.
Zero targets tested must not be reported as zero targets won. The implementation
is left in place for validation after infrastructure recovery; it is not killed
or approved for folding. Re-run the target probe and required tests first. If
fewer than three actual target warnings are removed, revert the engine changes
and record the killed result. If at least three are removed, run the prescribed
practice score before deciding or narrowing. Fresh SDK decoding is necessary to
populate certificates in previously stored evidence.

The complete-pronunciation constraint prevents erasing target phones, but shared
boundary conditioning can still veto a weak genuine substitution. No claim of
zero losses, target wins or release readiness is established. No scorer, lab
tooling, supplied manifest, evidence, test data or experiment-card edit was made;
no fold was performed.
