# Shared-transcript acoustic view score

Status: implemented; validation blocked by the supervisor before execution. The experiment card is unchanged.

## Prediction and mechanism

Predict listener −1 and overall −1 on practice, with substantial zero-gain risk.
The hypothesis is that streaming and cold-crop views disagree on emission timing,
and combining frames or voting on independent verdicts loses partial sequence
support. The experiment compares the exact complete written and already-accused
transcripts used by each existing supported string-path wrong-word comparison.
Each view sums its own complete CTC alignments before the two log likelihoods
are combined with equal fixed weights. Normalizing each view over the same two
transcripts cancels in the pooled log odds.

The temporary implementation uses the unchanged shipped model and the existing
fresh-listener cold-crop decoder. Crops retain the existing 300 ms collar rounded
outward to the native frame grid (320 ms at 40 ms), and total crop duration is
bounded to eight seconds. The primary view retains its streaming context; the
crop has its own frontend and listener state. Only the original accusation
interval is scored, with identical complete transcript candidates in both views.
No framewise correspondence is imposed.

A certificate requires a finite primary comparison favoring the rival and a
strict pooled reversal toward the written transcript. It is tied to the exact
word, complete strings, frame interval and primary likelihood margin. Missing
or impossible complete sequences abstain. Certificates travel as auxiliary token
metadata; primary phones, confidence, timestamps, posteriors and tracker input
are preserved. Graph confirmations and existing confirmation bars are excluded
from this additional string-path veto. No training or new model assets are used.

## Validation and limits

The implementation includes unit checks for normalized geometric pooling,
strict reversal, an emission-timing disagreement where framewise products prefer
the wrong sequence, impossible complete paths, exact certificate identity and
JSON/frame-cell transport. The six-shard supplied test harness was requested.
Target diagnostics discover actual checker comparisons without using target
labels to choose pronunciation strings or acoustic windows. The labels select
only which generic candidates to inspect and which existing warnings to report.
The intended probes are acoustic comparisons, not independent listening
judgments or a practice score. The ignored diagnostic source is retained under
`.autolab/target-probe.ts` for retry after the supervisor is repaired.

The current SDK integration enriches end-of-session saved evidence in sessions
that already retain PCM (recording enabled or an insertion teacher installed).
It does not revise warnings already published by the live checker. Immediate
warning vetoes and normal non-recording sessions remain an integration limit;
the pending acoustic target check must resolve the premise before further
streaming integration or any claim of production behavior.

Supervisor Modal requests 1 (data-layout check), 2 (the six-shard test harness)
and 3 (four-shard target probe) all failed before their commands started with
`NotFoundError: No such file or directory`. Each result reports zero spend and
contains no command output. Run 4, a minimal `pwd` startup check, failed with the same error.
Consequently, the tests have not run, no target audio has been processed, and no
practice score has been requested. No measured target or aggregate gain is
established. This infrastructure error is not evidence for the acoustic kill
criterion, so no killed outcome is declared.

The user's stopping rule requires rollback if the built change wins fewer than
three of the four registered targets, even though the card forecasts only one
net gain. If three or more targets are won, a practice score is required before
making a gate decision. No manual fold is authorized.
