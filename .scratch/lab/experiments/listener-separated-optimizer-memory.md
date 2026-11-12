# Listener: separated optimizer memory

Status: needs-info — implementation prepared; training and measurement blocked by the supplied supervisor.

## Prediction and premise

Prediction: listener −1, overall −1, one recovered addition and no losses.
This is below the standalone fold bar and the proposed loss-free result is
uncertain. The premise is that blank and sustained-sound updates contaminate
optimizer history used for intermittent consonant supervision. Teacher/student
disagreement alone does not establish that premise.

The six target incidents and the accepted card remain unchanged. No practice
score has been requested. Target consonant recoveries, target warning wins,
collateral losses and overall/component deltas are **unmeasured**, not zero.
There is no premise-based kill finding.

## Implementation

`modal/student_optimizer_memory.py` refits only the original affine phone head
on detached native streaming encoder features. It computes transcript-
conditioned CTC label occupancy, including repeated-phone and blank paths.
The shipped acoustic objective stays CTC + 2 × renormalized top-five teacher
cross-entropy, reduced by valid frames, with the original clamped teacher
frame mapping. No extra preservation loss or consonant weight is introduced.

For each frame, the CTC occupancy partitions the gradient into ordinary
non-nasal consonants, vowels/nasals/duration marks, and blank. Each partition
receives its occupancy share of the unchanged KD gradient. The sum is checked
against the complete objective's independently differentiated gradient on
every update. Nasal inventory symbols retain their transcript identities.

The three matched arms are:

- **Shared:** one Adam first/second-moment pair for the complete gradient.
- **Separated:** one pair per acoustic regime; the conditional gradient is
  normalized by that regime's occupancy mass, and the proposed updates are
  recombined with the same masses. Zero-occupancy regimes do not advance their
  moment clocks. This avoids implicitly tripling update magnitude or giving
  every rare regime the weight of a complete recording.
- **Memoryless:** the separated algebra with both moment coefficients zero;
  no gradient moment state is retained. The unchanged clipping rule still
  retains its ordinary norm history.

All arms start with the extracted shipped head, use the same ordered
recordings and fixed views, and receive identical update counts. The refit
schedule is fixed in code: seed 1729, 12 epochs, learning rate 0.0001,
moment coefficients 0.9/0.98. No practice input chooses these settings. This
bounded head refit uses Adam; it does not reproduce the full encoder training
trajectory or ScaledAdam's learned parameter-scale updates.

Clipping retains the pinned training optimizer's scale-2 median norm rule,
including parameter-RMS normalization, four-update RMS refresh, 100-update
history and initial 10/20/40-update estimates. A single factor derived from the
complete gradient scales every contribution before its optimizer sees it.
Regimes are never clipped independently.

`modal/student_optimizer_memory_train.py` enrolls only authorized, verified,
hash-matched performed transcripts and scored training-speaker help-train
none takes with no events, repeats or unsupported labels. A held-out speaker
in either corpus excludes that speaker globally. Duplicate audio is rejected.
Explicit speaker exclusions and both corpora are required. Speaker hashing
groups every take into fit or selection; both groups must contain both sources.

Each recording has its unchanged view plus two deterministic views using the
shipped noise/RIR/gain/SpecAugment chain. These are common to all three arms;
regime occupancy never changes the sampler or recording weights. Checkpoint
selection uses only the shared control's equal-speaker selection objective,
and selects one epoch/update count for every arm. The one-day preparation/refit
deadline is persisted with the cache. Cached evidence and model/auxiliary
hashes are rechecked before fitting.

`modal/student_optimizer_memory_export.py` exposes native encoder features and
replaces the existing head, with no added capacity. All arms remove the head's
dynamic activation rounding equally; the frozen encoder stays quantized.
Source files are immutable. The exporter copies and verifies every existing
insertion-teacher member, including external ONNX weights, and retains the
alphabet, blank index, 40 ms clock, frontend, native state inputs/outputs and
streaming geometry. The driver requires native ORT-versus-cached-head parity
on every selection view and three complete bundles before writing its final
training report. Tracker, checker, SDK decoding and insertion rules are unchanged.

The extraction/export scaffolding is adapted from the retained
`exp/listener-curvature-aware-head-refit` implementation; its curvature
objective, curvature solver and additional KL loss are not used here.

## Validation and infrastructure result

Local Python compilation and AST parsing passed for all four files. The
staged diff passes whitespace checks. No target-specific sessions, readers or
verse references occur in the engine or test code. The card, scorer, lab
tooling, test data, existing manifests and existing evidence were not edited.

The 23 synthetic tests cover exhaustive CTC path enumeration (including
repeated labels and an empty transcript), gradient-sum parity, frozen
encoder/teacher inputs, original loss reduction and top-five frame mapping,
nasal regime assignment, a PyTorch Adam oracle, separated-history isolation,
memoryless equivalence, occupancy-weighted update scale, clipping, global
speaker exclusion, corpus containment, matched order/update/checkpoint
selection, the one-day deadline, head replacement and complete auxiliary
bundle preservation. Their execution is pending; local Torch/ONNX dependencies
are absent, and heavy computation was not moved onto the Mac.

Supervisor requests 1–6 all returned `NotFoundError: No such file or directory`,
with zero reported compute spend and no shard logs. They comprised the supplied
data probe, baseline six-shard core/SDK tests, synthetic tests, and final test
retries. Request 6 also failed on the supplied runner's CPU `train` image,
which supplies Torch/ONNX. Changing to the appropriate dependency image did
not restore execution. No tests executed and no data listing was returned.

No supplied data listing, eligible training corpus, training run, checkpoint,
exported model, target full-session decode or practice score has been obtained.
No `.autolab/model.json` is declared: there is no verified bundle to declare.
Running the main listener would not evaluate this experiment. Infrastructure
failure is not evidence that separated memory fails, so no killed outcome or
fabricated target-win count is recorded.

## Remaining work

Restore the supplied supervisor and identify the authorized transcript index,
speaker-exclusion list and shipped augmentation banks. Run the 23 tests and
core/SDK suites, then use the driver's `prepare` and `fit` commands under the
CPU training image. The model input is the supplied `models/spliced-clean-int8`
bundle and the help input is `help-train/labels.jsonl`; all generated files must
remain under `/scratch/listener-separated-optimizer-memory/`.

After three bundles pass export parity, compare the training-selected
separated and shared heads on the target **full sessions**, distinguishing
missing planted-consonant recovery from confirmed warning wins. Follow the
user's final kill rule: fewer than three target wins is a kill; three or more
requires practice scoring before any decision. The card's separate two-target
consonant criterion is also untested. Do not use target decodes to select a
checkpoint or adjust the frozen schedule. Only a complete verified selected
bundle may be declared for the supervisor's practice decode.
