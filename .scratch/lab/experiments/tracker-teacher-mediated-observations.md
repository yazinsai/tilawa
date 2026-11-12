# Teacher-mediated student observations

Status: needs-human. The normalized transducer, identical-capacity direct control,
training estimator, optional local comparison and unit tests are implemented.
Supervisor requests for inventory, training and full validation failed before
execution with `NotFoundError: No such file or directory`. No trained model or
practice result was obtained. The default model is deliberately unset; this branch
is an unfinished experiment, not a measured unchanged-engine result. The card is
unchanged.

## Prediction

Tracker −1, overall 0 on lab-v1.5u practice. The target is
`lab-v1.5-p46/phone/warning:skipped_passage/54:36:1/121.84`.
The prediction includes one possible exposed listener warning at 54:37:2 after
correcting localization. Checker handling is unchanged.

The supplied earlier audits report that the near continuation remains available
and loses by 0.3915 edits over the complete observation interval. That is supplied
evidence, not a measurement reproduced here. Neither a teacher-mediated preference
at this incident nor its warning outcome has been established.

## Observation model

`packages/core/src/teacher-observation.ts` implements two normalized run
transducers. Each input phone emits epsilon, one phone or a two-phone fragment.
The first stage maps reference pronunciation to teacher run fragments; the second
maps complete teacher phone runs to student token fragments. Paired greedy teacher
and student decodes are observations, not verified spoken transcriptions. Only
authorized none-labelled training pronunciations anchor the first stage.

Composition multiplies and sums the stage probabilities over every intermediate
teacher fragment and every split of its student output. A reference phone can
therefore emit up to four student tokens. Epsilon teacher output and deletion of
a nonempty teacher fragment are distinct explanations sharing normalized mass.
Full-string forward sums then marginalize every segmentation of the observed
student string into these emissions. There is no best-teacher-string substitution
and no free counting credit for extra routes.

Runtime composition computes coefficients lazily rather than enumerating all
four-phone output strings. A fragment cache and shared reference-prefix likelihood
rows reuse work as observations extend. An observation fork or rewind invalidates
the row cache. Eviction changes computation only. Rows and model parameters contain
phone strings and probabilities, without passage or reader identities.

## Direct control and fitting

`packages/sdk/src/teacher-observation-fit.ts` reads only help-train labels/audio,
the shipped reference and the declared student/teacher bundle under
`models/spliced-clean-int8`. It requires the declared fp32 teacher in that bundle;
it neither searches for another teacher asset nor silently substitutes one.

Eligibility requires training-split, scored, none-labelled takes with no exclusion,
scripted event, unsupported extra mistake or non-error repeat. Every speaker with
any non-training annotation is excluded entirely, including mixed-split speakers.
Audio paths must resolve canonically inside help-train. Single-take student and
teacher greedy outputs retain complete ordered run strings. Cache keys include
audio, student manifest, teacher asset provenance and decoding policy hashes.

Both compared models have the same phone inventory, fragment support, two
probability tables, number of free parameters, pooled smoothing rule and 16 EM
iterations. Fragment support includes epsilon, all inventory phones and adjacent
phone pairs observed in the training partition's paired decodes.

The mediated fit separately estimates reference→teacher and teacher→student
tables with segmental forward/backward expectations. The direct control has the
same factorized reference→student capacity, but estimates both tables jointly
from reference/student pairs while leaving its intermediate run sequence latent.
Its intermediate codes receive no teacher-transcription supervision. Both models
see the same recording set and support construction.

Up to three deterministic partitions keep every take of a training speaker
together. Pooling masses 8 and 32 are selected by the combined held-speaker
student predictive log likelihood of the mediated and direct models. Validation
uses reference/student strings only, with no teacher available at inference.
The shared selected mass is frozen before refitting both models on all eligible
training speakers. Unsupported training or validation strings fail explicitly.
No partition selection, smoothing value or probability is fitted against practice.

Successful fitting would return `teacher-observation-models.json` and
`teacher-observation-fit-report.json` in LAB_OUT, with partition scores, hashes,
model provenance and parameter counts. Neither output was returned here.

## Local comparison

`local.ts` retains the pronunciation string actually traversed by each existing
winning DP witness. Substitution and reference deletion append the reference
phone; insertion appends none. A jump retains the earlier witness and appends its
landing phone without appending the unread jump interval. Every witness is paired
with the same complete student token string consumed since local-search creation.
Save/restore and recentering retain witnesses together with ordinary DP arrays.

For the existing best continuation and best restart, the comparison adds a
teacher-mediated likelihood residual against the matched direct control:

`corrected cost = existing cost − (UNIT / 3) × (log P_mediated − log P_direct)`.

The conversion uses the existing three-nats-per-edit convention. The direct
control defines the zero-residual comparison. Identical channels therefore retain
ordinary ranking, and exact corrected ties retain the ordinary winner. If either
candidate has an unsupported interval, neither receives a correction.

The original reference pronunciation symbols, duration deletion charges,
held-sound tolerance, movement edges and prices, surah-entry charges, unweighted
edit counts, commitment margins and checker rules retain their definitions.
The residual changes candidate selection; it does not waive an existing duration
charge, add a movement edge or veto a warning. This is a likelihood-ratio
correction to existing costs, rather than replacement of every acoustic edit price
with a generative negative log probability.

`TrackerOptions.teacherObservation` accepts the frozen paired models, or null.
`onTeacherObservationComparison` exports immutable diagnostics with the shared
observed string, both traversed strings, original costs, both channel likelihoods,
corrections and corrected ordering. It does not change event delivery or checker
windows. No runtime tracker code invokes the teacher.

`teacher-observation-model.ts` remains **null** pending real training. Installing
synthetic weights would not test the card, and scoring this unset default would
not validate teacher mediation.

## Validation

Ten new test bodies passed through Node 22 TypeScript stripping with a temporary
assertion adapter. This verifies source execution, not a semantic TypeScript build
or a complete Vitest run. The temporary copies and adapter were removed.

- Explicit composition matches lazy coefficients and retains total probability.
- Full-string sums retain probability mass across alternate segmentations.
- Invalid simplexes and inventory mismatches are rejected.
- Paired-data fitting produces normalized, equal-capacity mediated/direct models.
- Identical-channel local replay retains positions and original charges.
- Immutable comparison diagnostics cannot revise the decision through mutation.
- Rewind restores observations and reference witnesses together.
- A synthetic residual reverses a raw continuation/restart ordering.
- An integrated synthetic local replay restarts on its look-alike under the direct
  control and stays on its continuation under the mediated channel.
- Training eligibility excludes extra mistakes, repeats and held-out/mixed-split
  speakers; deterministic partitions keep each speaker's takes together.

The fitter passed a Node syntax check. `git diff --check` passed. No scorer, lab
tooling, manifest, evidence or test data was changed. New tests synthesize their
examples in code.

| Supervisor run | Requested work | Result |
| --- | --- | --- |
| 1 | Read-only authorized input inventory | NotFoundError before execution |
| 2 | Read-only inventory with a corrected multiline shell request | NotFoundError before execution |
| 3 | Required six-shard core/SDK build and tests | NotFoundError before execution |
| 4 | Core/SDK build and paired training-only fitting | NotFoundError before execution |
| 5 | Supplied target trace CLI | NotFoundError before execution |

Practice scores requested: **0 of 3**. Target wins, tracker/overall changes,
clean-session and studio effects are unknown. An infrastructure outage establishes
neither a premise kill nor a measured count of zero wins. No killed outcome is
written and no fold/discard decision is taken.

## Remaining validation

After supervisor recovery, run the training estimator and required checks:

```sh
python3 /Users/rock/ai/projects/tilawa/tools/autolab.py modal-run --timeout 60 -- 'pnpm --filter @tilawa/core... --filter @tilawa/sdk... build && TEACHER_OBSERVATION_CACHE=/scratch/tracker-teacher-mediated-observations/paired-decodes pnpm --filter @tilawa/sdk exec tsx src/teacher-observation-fit.ts'
python3 /Users/rock/ai/projects/tilawa/tools/autolab.py modal-run --shards 6 -- 'bash tools/autolab_tests.sh'
```

Freeze the returned selection before examining practice ordering. Inject the
returned pair through `teacherObservation` and capture
`onTeacherObservationComparison` while replaying the card's actual primary
student evidence. Compare the retained near/restart strings over the exact same
observed interval, including their raw and corrected costs. Replay with
`{ mediated: fitted.direct, direct: fitted.direct }` for the matched control.
Run the unchanged checker on each replay and count confirmed target warnings,
including the possible exposed listener warning. A changed cursor alone is not
a target win. Install the real frozen default only after that validation.

Apply the requested kill rule to measured results: if the premise fails or the
built change wins fewer than three targets, revert engine/tests and record a
killed outcome with the measured target count. If at least three targets win,
obtain the required practice score before considering narrowing; the supervisor
gate trades gains against losses. No constants should be adjusted against it.

## Limits

Real-data fitting, transfer to the target, full build compatibility and full tests
remain unavailable. The finite fragment support cannot express arbitrarily long
single-run outputs. The local comparison marginalizes emissions within each
retained reference witness, not all movement histories discarded by the original
DP. It uses completed traversed-phone emission strings, without an additional
latent unfinished-fragment frontier. These approximation choices are shared by
both channels but have not been validated on real streaming evidence.

The model uses phone/run identity, not teacher confidence or teacher durations;
original tracker duration charges remain active. Earlier local-start observations
are not added to the current interval. Prefix likelihood storage can grow with
the local history, so real CPU and memory measurements are required before any
performance claim. No independent listening opinion or label correction was
obtained. Synthetic passing tests do not establish a practice gain or a fold.
