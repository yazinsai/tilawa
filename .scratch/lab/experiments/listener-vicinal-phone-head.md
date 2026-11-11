# Vicinal frozen-encoder phone head

Status: ready-for-human. Implementation draft builds; acoustic training and target
validation are blocked because the authorized supervisor failed before executing
any command. The experiment card is unchanged. This is not a measured kill or a
bundle ready for scoring.

## Prediction and mechanism

The card predicts two false warnings removed and one caught substitution lost:
listener −1, overall −1. The proposed defect is an incorrect rival decision pocket
between legitimate acoustic realizations. It remains a hypothesis.

`modal/student_vicinal_export.py` adds a pre-CTC feature output to a separate copy
of the shipped frozen ONNX encoder. It verifies unchanged primary outputs and
caches on eight consecutive windows before preparing training data. The primary
model and supplied manifest are copied unchanged. A seeded 32-dimensional
projection precedes matched 48-unit tanh phone heads.

Only help-train training speakers' none-labelled recordings supply supervision.
Exact transcript-constrained CTC forward/backward produces soft phone and blank
occupancies; it does not force hard frame labels. Interpolation donors come from
different recordings with the same immediate left/central/right phone context.
The treatment learns convex feature mixtures and corresponding mixtures of soft
targets. Its control uses the same original samples, donor endpoints, mixing
coefficients, initialization, capacity and loss weight, without interpolating
features. Sampling balances recordings.

Three speaker folds select interpolation strength from {0.25, 0.5, 1} and AdamW
regularization from {0.0001, 0.01}. Selection first minimizes the fraction of real
substitution comparisons vetoed, then speaker-averaged soft-target validation
loss. The fixed recipe uses 18 epochs. None-labelled speakers are spread across
folds before assigning speakers used only for preservation checks. Every split
is confined to training speakers; any speaker named by a non-training label is
excluded. Weak substitution labels select baseline engine accusation windows
for preservation evaluation and never supply assumed performed transcripts.

The SDK scores complete written and already-accused pronunciations over the
checker's unchanged string-path confirmation window. Both heads marginalize
CTC boundaries under the same completion-conditioned timing prior. A certificate
requires finite strict disagreement: control prefers the rival, treatment
prefers the written pronunciation. It names the exact word occurrence,
pronunciations and frame bounds. Equal scores, unknown phones and impossible
complete pronunciations abstain.

The checker downgrades only the public confirmed string-path wrong-word verdict.
Its internal ownership and confirmation ledger remains unchanged, preserving
subsequent candidate creation and insertion merges. Parallel-graph accusations
are exempt. Primary phonemes, posterior rows and tracker inputs are retained.
Audio sessions transport the certificates into stored evidence and publish
matching live downgrades at `end()`; evidence-only sessions consume certificates
without running another acoustic model.

## Validation and limits

Five local Python mechanism tests pass: exhaustive soft-occupancy marginalization,
genuine repeated-phone blank requirements, donor context/recording restrictions,
complete normalized timing scores and speaker-fold anchor coverage. Both core
and SDK TypeScript builds pass. The six new core tests pass, including exact
certificate transport, actual insertion preservation and graph exemptions. The
SDK tests verify complete comparisons, matching native feature windows/caches,
end-of-session publication and rejection of incomplete declared bundles.

The local SDK suite passed 31 cases and skipped one optional help integration;
two existing loader tests initially failed because their filesystem mock returned
the primary manifest for every file, including the new optional descriptor.
After making that mock represent an absent descriptor, all three tests in that
file passed on a focused rerun: all 33 SDK cases have passed across the runs.

The full local core run passed 750 of 762 cases. Twelve existing cases timed out,
and Vitest reported two of the worker RPC deadline errors documented in the
supplied harness. A one-worker rerun with a 60-second command-line timeout
passed eleven of those twelve cases. The remaining startup-persistence case
still exceeded its explicit 15-second limit (about 18 seconds); that limit and
its assertions were left unchanged. Thus 761 core cases have passed across the
runs, but the complete core suite and required Modal harness have not passed.

The Node loader verifies both declared scorer files, their lengths and SHA-256
hashes, the primary graph hash, inventory, head dimensions and native sample/frame
geometry. Missing or corrupt declared assets throw instead of silently evaluating
the primary listener. The export utility is intended to finish the entire bundle
and verify ordinary SDK loading before any target diagnostic.

Supervisor requests 1, 2 and 3 all returned
`NotFoundError: No such file or directory` before command execution, with zero
compute reported. They attempted a read-only dataset census, the required
six-shard test harness, and the complete test/extract/train/export command.
Consequently the actual help-train coverage, frozen feature output, real
substitution preservation coverage, selected recipe, export parity and fitted
bundle have not been measured. No practice scores have been used. Target
removals are unmeasured, rather than zero. No `.autolab/model.json` is declared
and no killed outcome is written.

This draft buffers full-session audio and runs a separate feature pass only when
there are eligible accusations. Memory and recheck latency therefore grow with
session duration. Live primary warnings can appear before their end-of-session
downgrade; this experiment does not establish immediate streaming warning
precision or device readiness. The small training set and soft alignments could
still teach an incorrect written-word preference and suppress real substitutions.
Unit tests establish plumbing and mathematical behavior, not acoustic separation
or population safety.

The requested next step remains restoring the authorized supervisor and running:

```sh
python3 modal/student_vicinal_export.py \
  --model /lab/data/models/spliced-clean-int8 \
  --out /scratch/listener-vicinal-phone-head/bundle \
  --labels /lab/data/help-train/labels.jsonl
```

After successful ordinary SDK loading, the reusable SDK CLI can compare existing
evidence with and without certificates, checking unchanged tracker events and
insertion decisions. Labels identify reported incidents only; they never select
inference windows or tune a score. The user requires at least three actual target
warnings removed to continue, superseding the card's two-window reversal
criterion. Fewer than three actual removals calls for rollback and a measured
kill. Three or more requires a practice score before deciding. Infrastructure
failure does not establish either result.

No supplied data, evidence, scorer, lab tooling or experiment-card edits were
made. No fold was performed.
