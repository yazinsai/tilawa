# Channel-transition phone readout

Status: blocked on supervisor command execution. Implementation is complete,
but training, bundle verification and remote tests have not run.

## Prediction and frozen mechanism

Predict one phone wrong-word false warning removed with no losses: listener −1,
overall −1. The card is unchanged. The user's continuation rule requires at
least three actual target wins before practice scoring, overriding the card's
two-window premise threshold.

Freeze these choices before target inspection: 32 projected frozen encoder
features; a 48-unit tanh temporal head over previous/current/next native 40 ms
frames; seed 1729; three speaker-grouped training folds; 36 epochs; AdamW at
0.002 with 0.0001 weight decay; paired pronunciation-consistency weight 1;
checkpoints at epochs 12, 24 and 36, selected by mean speaker canonical CTC NLL
on unaugmented and augmented training-validation recordings. No practice audio
or labels select geometry, losses or checkpoints. A common checkpoint minimizes the mean validation loss across all three arms,
so the exported heads have identical update counts as well as transcript data.
This refinement was made during static review, before any target inspection.

The augmentation family is ordinary zero-phase microphone-band filtering,
already used by the listener's channel augmentation experiments. Draw lower
cutoffs uniformly from 80–300 Hz and upper cutoffs from 3000–7600 Hz. Apply both
filters to the complete recording before mixing, so neither resets at a seam.
One view holds filter A fixed; a second crosses smoothly from A to B; a third
holds B fixed. The transition center is drawn from the middle half of the
recording and its width from 0.4–1.2 seconds (capped at half the recording).
A raised cosine preserves sample count and timing, and every view inherits the
same authorized transcript. Two seeded triplets per training recording provide
fixed cached features; validation uses a separate seed.

All three heads receive identical clean/A/transition/B CTC examples and a common
three-epoch soft-occupancy warm-up. The transition arm also matches pronunciation
probability distributions between A and transition; the unpaired control has no
consistency term; the fixed-channel control matches A and B. Distributions score
the complete canonical phone sequence and three deterministically sampled
single-phone alternatives through exact CTC. Thus the controls isolate paired
consistency and within-recording transition exposure, without changing capacity
or transcript supervision. Only eligible help-train none-labelled training
speakers supply canonical labels. Speaker exclusions and invalid recordings are
respected independently; mistake and repeat labels never become transcripts.

The SDK requests complete written and already-accused pronunciations from the
checker's existing string-path support comparison. It scores both with identical
CTC boundary marginalization and a timing prior conditioned on completion.
A veto requires a strict written preference from the transition head and strict
rival preferences from both controls; ties, unknown phones, impossible complete
pronunciations and nonfinite scores abstain. The certificate binds both full
strings, the word and exact frame interval. Graph confirmations remain exempt.
Primary phones/posteriors, tracker events and the internal checker ownership
ledger remain unchanged, preserving insertion decisions. The implementation
reviews at audio completion, so it does not establish timely streaming warnings.

## Validation and limits

The implementation includes unit tests for timing-preserving crossfades, grouped
speaker eligibility, soft canonical CTC occupancies including repeated phones,
temporal-context construction, symmetric pronunciation consistency and gradients,
complete-sequence timing marginalization, all three strict preference signs,
nonfinite/impossible comparisons, exact certificate frames and strings, metadata
transport, missing/corrupt asset rejection, audio-session downgrades, graph
exemptions and insertion preservation. Python syntax checks and
`git diff --check` pass. These are static checks, not behavioural test passes.

A complete bundle must retain the primary model and its existing insertion
teacher, expose frozen features in a separate graph, prove bitwise parity for all
original graph/cache outputs over eight successive windows, and declare sizes
and SHA-256 hashes for the feature graph and three fitted heads. The SDK loader
checks those files and ties the heads to the exact primary graph hash. The export
utility verifies normal SDK loading and requires the existing insertion teacher.

Supervisor request 1 (data/model inventory), request 2 (six-shard tests), request
3 (dependency installation, Python tests, training/export and bundle loading),
and request 4 (six-shard tests of the implementation) all returned
`NotFoundError: No such file or directory` before executing their commands. Each
returned result reports zero compute spend; there are no returned shard logs,
training reports or model files. Installing dependencies inside request 3 could
not help because its command never launched. No acoustic target comparison and
no practice score have been used. This does not show that the mechanism is
absent or that fewer than three targets are won; no killed outcome or measured
zero has been fabricated.

There is no verified bundle and no `.autolab/model.json`. Running the primary
model alone would not evaluate this experiment. Do not score or fold this branch
as a completed acoustic trial until the supervisor can execute the pending
workflow. Reissue request 3's command from `.autolab/modal-request-3.json`, then
rerun the six-shard tests. The generic SDK diagnostic accepts `--model`, `--audio`,
`--evidence` and `--out` and compares actual engine-requested windows without
label-based cutting. Inspect the four card targets only after training selection
and bundle verification. If fewer than three target warnings are actually
removed, revert the implementation and record a killed outcome. If at least
three are removed, declare the verified scratch-relative `bundle` through ignored
`.autolab/model.json` and run the authorized practice score before deciding.

Small none-labelled training coverage, incorrect transcript occupancies and weak
genuine substitutions remain risks. Reviewing only at completion does not prove
timely streaming vetoes: a live confirmation can precede its final downgrade.
No scorer, lab tooling, supplied manifest, test data, evidence or experiment card
was changed. No fold was performed.
