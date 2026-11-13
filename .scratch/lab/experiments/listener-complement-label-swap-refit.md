# Complementary-label residual phone-head refit

Status: killed before acoustic evaluation. The card is unchanged.

## Prediction and mechanism

The card predicts one recovered substitution, zero losses, listener −1 and
overall −1. Its two registered targets are:

- `lab-v1.5-p30/phone/mistake:wrong_word/2:284:28/247.30`
- `lab-v1.5-p48/phone/mistake:wrong_word/12:78:8/135.25`

Freeze the shipped encoder and fit a small residual phone head on help-train
training speakers. A substitution label supplies evidence against the written
word without claiming that its scripted replacement is the performed transcript.
The canonical-slot probability is conditional on the transcript outside the
marked slot: canonical transcript CTC mass divided by CTC mass of the same
prefix and suffix with an arbitrary nonempty phone sequence in the slot. All
frame boundaries remain latent. A deterministic denominator automaton is needed
so ambiguous flank boundaries do not count the same frame path more than once.
The loss is minus log of one minus this event probability.

None-labelled recordings retain ordinary transcript loss. Shipped-posterior KL
outside the marked slot preserves intact speech; the mask is derived from
canonical transcript-constrained forward-backward on frozen features so the
residual head cannot move its own preservation mask. Identical-data ordinary
continuation and deterministic-script supervision are necessary controls.
Regularization and checkpoint selection must be fixed within training-speaker
partitions before inspecting practice outcomes. Release-held-out speakers must
never enter fitting, selection or analysis. Cached-feature fitting is bounded to
one day. Inventory, streaming timing, tracker, checker and warning requirements
are unchanged.

The card's acoustic kill check compares the selected complementary head with
matched ordinary continuation: a reduced canonical-slot event probability alone
is insufficient; a target must gain a correctly ordered distinguishing consonant
absent from that control. The task additionally requires at least three target
wins before practice scoring, although this card registers only two targets.
No extra practice cases may be invented to resolve this mismatch.

## Validation and limits

Read `AGENTS.md`, the card, the audio-review instructions, the existing student
training/export code and SDK model interface. The earlier real-swap fine-tune
branch named in the brief is not present in this worktree's available refs; no
outside checkout was searched.

A temporary objective prototype and unit checks were prepared for exact
open-slot mass, repeated CTC labels, ambiguous boundaries, complement gradients,
latent slot occupancy, zero-initialized residual output and deterministic
speaker partitions. Python syntax compilation passed. The prototype has not
been trained, exported or tested against real audio. The local interpreter has
no torch, ONNX or ONNX Runtime, and heavy work belongs on Modal.

Supervisor Modal request 1 requested the supplied six-shard core/SDK build and
test harness. Request 2 requested read-only help-train/model/dependency
inspection. Both failed with `NotFoundError: No such file or directory` and
reported zero spend, with no command log or dataset output. Consequently neither
the test suites nor training-data eligibility nor model feature extraction was
validated. Request 3 retried execution with a working-directory probe, the objective unit
checks and graph inspection. It also failed with the same error, zero spend and
no command output. All three requests therefore failed before an acoustic or
runtime test result. These are infrastructure failures, not evidence that
complementary supervision fails to repair substitutions.

## Decision

The task's three-target-win minimum remains in force. Even perfect recovery of
both registered targets has an upper bound of two wins, below that minimum.
The experiment is stopped on this scope limit, without claiming that the
acoustic premise was falsified. No targets have been examined or verified won.
No practice score was requested, so all three scoring opportunities remain.

Temporary objective/test sources are removed from engine code on stopping; the
preparation snapshot is retained only in ignored local run artifacts. No trained
weights, exported model bundle or `.autolab/model.json` is declared. Tracker,
checker, SDK, scorer, supplied data, evidence and manifests are unchanged.
The only committed change is this result write-up. Core and SDK test execution
is unverified because the supplied remote harness did not start successfully.
