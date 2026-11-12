# Recording fixed effects in a frozen phone readout

Status: implementation written; blocked on supervisor execution before training
or target measurement. This is not a measured kill or a scoring-ready bundle.

## Prediction

The card predicts two false warnings removed and one caught substitution lost,
for listener −1 and overall −1 on practice. Persistent recording-specific class
preferences may otherwise be absorbed into the shared acoustic classifier.

The card's early threshold is two pooled-rival to shared-written reversals.
The user's later instruction requires **three actual target warnings removed**
before practice scoring. That later instruction governs the experiment. No
practice score has been requested.

## Mechanism

`modal/student_fixed_effect_export.py` makes a separate feature-output copy of
the shipped encoder, checks eight consecutive seeded windows for bit-identical
primary outputs and caches with graph optimization disabled, and preserves the
original primary bundle. A seeded 32-dimensional projection feeds a small
32→48→phone tanh readout.

`modal/student_fixed_effect_phone.py` trains matched pooled and fixed-effect
arms. Both start from the same readout, initialized for four epochs to the
shipped posteriors on the eligible training recordings, then optimize exact
complete-transcript CTC on identical recordings in identical update order.
CTC sums over latent frame assignments; the training objective does not freeze
the shipped alignments or use scripted mistake transcripts as truth.

The treatment adds a learned recording-specific nonblank phone-logit vector.
Double centering makes each recording's phone sum zero and each phone's
recording mean zero; the blank offset is fixed at zero. A squared-offset penalty
limits nuisance absorption. Held-out scoring and deployment use only the shared
head. Recording identities and offsets are never exported to the SDK.

Only help-train training speakers' none-labelled takes supply canonical CTC
transcripts. The extractor rejects none labels with events or repeats and
excludes every speaker found in a nontraining split. Actual engine-requested
wrong-word windows overlapping scripted substitutions supply weak preservation
controls; the script never supplies a performed pronunciation or a crop.

Three speaker-grouped training folds select the offset penalty from 0.01, 0.1
and 1, and the checkpoint from epochs 8, 16 and 24. Selection first minimizes
the fraction of preservation comparisons reversed away from the rival, then
held-out-speaker shared-head CTC loss. Normalization and common initialization
are fitted within each training fold. The selected setting is refitted on all
eligible training takes with an identical-data pooled control.

The SDK compares both complete pronunciations on the exact existing support
window, using ordinary CTC forward sums with identical blank and repeat rules.
A certificate requires strictly finite shared-written preference and strictly
finite pooled-rival preference. Ties and unscorable candidates abstain.

Core consumes certificates only on the exact word, written pronunciation, rival
pronunciation and frame bounds of an otherwise supported string-path wrong-word
confirmation. Graph confirmations remain exempt. The internal confirmation and
ownership ledger stays unchanged; the public verdict becomes uncertain, so
subsequent insertion merges and candidate generation retain baseline state.

Audio sessions collect features through an independent side-graph cache on the
listener's exact input windows before delivering primary evidence. At live
confirmation, the cached features score the requested complete window; a
certificate prevents the public confirmation. The primary tensors are returned
unchanged. At `end()`, stored-evidence review reuses the collected features,
stores certificates in recorded evidence, and downgrades only matching baseline
events that were not already vetoed live. It does not change primary phones,
posterior rows or tracker events. A declared side
bundle is checked for file length, hash, inventory, geometry and exact primary
model hash; missing or corrupt declared assets fail loading.

## Validation

Python syntax compilation and `git diff --check` pass locally. Unit tests cover
double centering and its gradients, exact latent CTC against enumerated paths,
identical pooled controls, release-speaker exclusion, discarded recording
parameters, complete pronunciations and repeat blanks, exact-window certificate
matching, graph/insertion preservation, independent streaming caches, unchanged
primary tensors, live suppression and session downgrade/serialization.
Their execution is requested through the authorized supervisor.

Remote requests 1–5 (data census, minimal `pwd`, six-shard core/SDK validation,
persistent bundle export, and a test retry after the streaming integration)
all failed with `NotFoundError: No such file or directory`, without command
output. The result files report zero spend. No remote request remains pending.
These errors establish no result about the acoustic premise.

The persistent build request was:

```sh
python3 /Users/rock/ai/projects/tilawa/tools/autolab.py modal-run --timeout 60 -- 'python3 modal/student_fixed_effect_export.py --model /lab/data/models/spliced-clean-int8 --labels /lab/data/help-train/labels.jsonl --out /scratch/listener-recording-fixed-effect-head/bundle'
```

Once execution is restored, the build must finish and normal SDK loading must
pass before declaring `.autolab/model.json` with path `bundle`. Then inspect
training-only selection/provenance and run the SDK diagnostic on the supplied
target audio/evidence. Its comparisons use actual checker requests, never the
label's time as a crop. Fewer than three actual target removals requires the
user's kill procedure and engine rollback; three or more requires a practice
score before any gate decision. Nothing has been folded.

## Limits

No trained model, training provenance report, export parity result, actual-window
target result, full practice score or studio result is available yet. The
mechanism remains untested; absent observations are not zero targets won. No
model path is declared until the complete bundle is built and verified.

Evidence-only sessions require carried certificates; they do not infer features
from text or posterior rows. The side encoder adds computation and keeps the
projected feature history for complete-window comparisons. Recording offsets
may absorb genuine articulation differences instead
of nuisance variation; the weak training substitution controls cannot guarantee
preservation. Three folds and a small projected head also do not establish
generalization beyond the authorized training speakers.
