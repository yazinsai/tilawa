# Diphone readout resume

Status: infrastructure-blocked, unmeasured.

The authorized execution service has not been restored. Both minimal health requests
returned `NotFoundError: No such file or directory` before producing any container
output. The card requires service restoration before resuming the archived draft;
that prerequisite remains unmet. This explains why this branch contains no engine
change. It is not evidence against the acoustic mechanism.

## Prediction and mechanism

The prediction is two false warnings removed and one caught substitution lost:
listener −1 and overall −1. Independent evidence for individual sounds may favor
a rival without supporting its successive sound joins. The proposed recheck trains
a bounded overlapping-diphone readout and an equal-capacity single-phone control
on identical authorized recordings, with the shipped encoder frozen. Help-train
transcript supervision uses only training speakers' none-labelled takes; release
held-out speakers are excluded and model selection stays within training-speaker
partitions.

At existing string-path warning windows, the SDK compares the complete written
and already-accused pronunciations, marginalizing placements and requiring
successive diphones to agree on their shared articulation. A veto requires the
diphone readout to prefer written while the matched control prefers the rival.
Primary decoding, tracker events, graph confirmations and insertion decisions
must be preserved. These restrictions do not establish preservation of genuine
substitutions.

The archived source and tests are recoverable from commits `e2006685` and
`29714bfe` on `exp/listener-overlapping-diphone-readout`. Read-only inspection
confirmed the frozen-feature export, matched training utility, SDK readout,
certificate transport, checker integration and unit tests. The archive's final
write-up records seven earlier executor failures and rollback of the draft.
No archived engine source was restored on this resume branch, including during
the follow-up prompted by the previous attempt's no-engine-code feedback.

## Validation and limits

Supervisor request 1, submitted at 2026-11-12 16:30:45 UTC, asked for a minimal
Python health marker in `LAB_OUT` and available scratch-storage information,
using the authorized `autolab.py modal-run` route, one shard and a two-minute
container timeout. It returned `NotFoundError: No such file or directory`; the
result was observed at 16:34:22 UTC. The result contains no successful container
execution, health marker or storage measurement. Request and result are retained
in untracked `.autolab/modal-request-1.json` and `modal-result-1.json`.

Supervisor request 2, submitted at 2026-11-12 16:38:24 UTC in the follow-up,
asked only for the execution environment, writable persistent scratch storage,
available disk space and the authorized data directory's top-level listing.
It used the same authorized route, one shard and a five-minute container timeout.
The command returned `NotFoundError: No such file or directory` at approximately
16:41:29 UTC, exit code 1. The supervisor result again reports cost zero and
contains no container output, storage measurement or data listing. The requested
scratch marker therefore has not been verified. The request and returned failure
are retained in untracked `.autolab/modal-request-2.json` and
`modal-result-2.json`; the command completed before this report was written.

After these failures, no training, test-harness request or practice score was
attempted. Repeating the archive's training or tests against the same
failed service would not test the premise. No alternative execution route was
used. Core, SDK and Modal engine files remain identical to this worktree's
starting state; the experiment card is unchanged. No new model bundle or
`.autolab/model.json` is declared, because none could be trained and verified.
The missing execution service prevents access to the authorized acoustic training
inputs and the shipped graph needed for export parity checks. Restoring source
alone would leave the actual runtime without trained diphone and control assets,
and scoring the main listener would not evaluate the proposed experiment. That
is the concrete reason this follow-up still cannot ship an engine change.

Zero target windows were tested. Target wins, preference reversals and practice
change are unknown, not measured zeroes. Neither the card's fewer-than-two
reversals criterion nor the task's fewer-than-three target wins stopping rule
can be assessed. This is an infrastructure-blocked outcome, not a premise kill
or a completed implementation. Resume the recoverable draft only after the
authorized execution service succeeds, then complete training, bundle loading
and parity checks, unit tests and actual engine-window comparisons before
applying the target continuation rule.
