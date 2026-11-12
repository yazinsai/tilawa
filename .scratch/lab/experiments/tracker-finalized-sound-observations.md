# Finalize acoustic sound observations before local restart commitment

Status: killed on the unchanged-cost premise. No engine change retained.

## Prediction

Tracker −1, overall 0 on lab-v1.5u practice with listener
`exp-listener-teacher-insert-marginal-gate-ad6eb8c0`. Correcting the binding could expose
the documented nearby listener false warning. Target:
`lab-v1.5-p46/phone/warning:skipped_passage/54:36:1/121.84`.

## Mechanism

Distinguish an acoustically open observation from a completed sound run. Keep local
continuation and restart decisions provisional until a blank or different sound
closes the distinguishing observation. Consume each observation once, retain the
existing edit costs and commitment rules, and preserve published positions until
that decision completes. This is about an acoustic run, not an unfinished written
word or a new movement penalty.

## Source inspection

`packages/sdk/src/listener.ts` emits an open run at an inference-window boundary
and suppresses subsequent emissions for that uninterrupted sound (`carried`).
Thus a delivered token can in principle have an unfinished acoustic duration.
`trackerInput` strips posteriors, and `follow` advances local matching with only
`local.step(ph.p)`. `packages/core/src/local.ts` does not read observed duration,
confidence, or posterior trajectories when updating either candidate's costs.
Additional frames of the same sound therefore provide no additional decoded
observation and cannot by themselves alter the existing cost comparison.

The existing SDK listener test with a constant-token backend explicitly requires
one emitted token across every inference window. The insertion-teacher listener
preserves those primary tokens and adds auxiliary seam evidence; it does not
export updates to an open primary run. Rechunking likewise preserves each token
whole. These are implementation facts, not proof that the target sound was
already acoustically closed at delivery.

Completing the proposed observation can extend its endpoint or update its
confidence, but it retains the same phone identity. Local continuation and restart
both consume that identity exactly once in the same recurrence, so their cost
rows, origins, edit counts and recent-cost contributions are unchanged. A blank
closes the run without supplying another phone. A different sound closes it and
starts a new observation, whose identity is already available to both paths under
ordinary tracking. The supplied previous audits place the near continuation
0.3915 edits behind the restart; this experiment has not reproduced that number.
Finalizing the existing observation alone cannot reverse that ordering.

Changing the acoustic scoring model or deciding on additional subsequent phones
would test a different mechanism. Neither was substituted for the card's
unchanged-cost observation-finalization probe.

## Validation

A temporary read-only replay was prepared to compare ordinary delivery with
delivery deferred until a posterior top-1 blank or different sound closes each
local observation. First acquisition remains unchanged, and the checker receives
ordinary evidence and the resulting tracker positions. It was designed to record
the target's delivered run, actual acoustic endpoint, closing frame, local events
and confirmed warnings.

Both supervisor requests failed with `NotFoundError: No such file or directory`,
with no command log or returned artifacts and zero reported compute spend. Request
2 retried the audit with the explicit `node` image. The temporary audit script was
removed after both requests finished.

The previous `exp/tracker-substitution-run-binding` write-up independently
documents the once-per-uninterrupted-run contract and three supervisor requests
failing with the same infrastructure error. Its measurements are not represented
as a new replay here.

No engine behavior changed, so no unit-test changes are retained. The build and
core/SDK unit suites were not run, and no passing-test claim is made.

## Limits

No practice score was requested (0 of 3 used). There is no measured overall or
tracker delta, and the target's actual closing frame has not been established.
This kill relies on the local scoring contract, not a claim that every target
observation was closed or a claim about downstream warning behavior.

## Decision

Kill because completing an uninterrupted sound observation leaves the one
consumed identity and the existing local candidate costs unchanged, so the
specified finalization cannot reverse the target restart under unchanged costs.
Zero target wins means none demonstrated on this unchanged-engine branch, not a
measured practice improvement or regression. The card, checker, scorer, lab
tooling, labels and evidence are unchanged.
