# Delayed recognition copies of earlier phones

Status: needs-human. The optional runtime channel, training estimator, source-independent control,
traceback audit and unit tests are implemented. All four requests through the supplied supervisor
failed before command execution with `NotFoundError: No such file or directory`. Training,
target validation, full TypeScript build and core/SDK Vitest results remain unavailable.
The experiment card is unchanged.

## Prediction

Tracker −1, overall 0 on lab-v1.5u practice, targeting
`lab-v1.5-p46/phone/warning:skipped_passage/54:36:1/121.84`.
The overall forecast includes the documented nearby listener warning at 54:37:2 that a correct
binding may expose. No checker change attempts to hide that warning.

The supplied audits report that the near continuation remains available and loses by 0.3915
edits over the same complete observation interval. That is supplied evidence, not a number
reproduced here. Whether one of its insertion observations has a distinct earlier acoustic
source remains unmeasured.

## Runtime mechanism

`copy-channel.ts` defines a normalized observation distribution for insertion transitions.
An observation is either an ordinary insertion or a copy of a previously consumed observation.
The branches share probability mass; their likelihoods are summed, rather than adding a free
copy route to the minimum edit path. Both branches consume the new observation and advance no
reference phone. An insertion costs `-log(P(insertion) P(observation | history))`, expressed in
the existing three-nats-per-edit units, and remains strictly chargeable.

Sources lie within the local matcher's existing 16-observation recent horizon. Each must end
before the current token starts. Identical phone/start/end records count as one source,
excluding duplicated duration records of the same acoustic occurrence. Source weights combine
a learned token-lag distribution and manner-class weights and normalize over the eligible
sources. Copy outcomes share parameters by phonetic class, with a separately learned exact
identity branch. Empty history uses the ordinary distribution. The source-independent control
uses the same branches and copy kernel but draws source identity from a fitted pooled
distribution instead of the observed history.

`local.ts` computes the insertion distribution once per observation and applies it to every
candidate cell. It retains immutable observation snapshots across save/restore and window
recentering. Earlier inserted observations can themselves be sources: this is a recognition
process over consumed observations, without requiring the tracker to have assigned the source
to a correct reference phone. The history excludes the current and future observations.

Original held-sound insertion discounts retain their original costs. Reference deletion and
duration charges, match/substitution costs, jump edges and prices, surah-entry charges,
recombination and commitment rules retain their original definitions. A copy still counts as
one observed/reference edit in PER. `tracker.ts` passes token timestamps to local matching;
checker rules and window handling are unchanged.

`TrackerOptions.copyChannel` accepts either frozen model, or null. The default in
`copy-channel-model.ts` is **null** pending a real fit. No synthetic or target-derived model
was installed. Consequently the current default product invocation does not yet test the
proposed channel; scoring it would measure the unchanged channel. Default activation is an
unfinished dependency, not a measured zero-gain result.

## Training and control

`packages/sdk/src/copy-channel-fit.ts` opens only help-train labels/audio, the shipped reference
and the deployed listener at `models/spliced-clean-int8`. Eligibility requires explicitly
training-split, scored, none-labelled takes with no mistake or repeat annotations and no
exclusion. Every speaker with any non-training annotation is excluded entirely, including
mixed-split speakers. Audio must resolve canonically inside help-train. Decode caches include
model, decoder and audio hashes.

A normalized complete-string edit model runs forward/backward, marginalizing edit placement
and latent insertion/copy-source assignments. Fixed ordinary recognition/deletion emissions
derive from existing acoustic costs; insertion frequency, copy frequency, exact-copy probability,
class distributions and source selection are estimated. Hierarchical EM runs 24 iterations,
with pooled smoothing and normalized source-weight gradient updates. These fitting definitions
were chosen before any returned practice evidence.

Up to three deterministic speaker-grouped training partitions select pooling mass 8 or 32
separately for linked and independent models by predictive log probability. Every take of a
speaker stays in one partition. The independent model uses the same edit graph, emissions,
priors, exact-copy branch and class sharing. Selection then freezes before both models refit
on all eligible training speakers. Outputs are `copy-linked-model.json`,
`copy-control-model.json` and `copy-fit-report.json`, including provenance and partition scores.
No such outputs were returned, so no actual selection or training gain is claimed.

The normalized training edit model is a calibration model, not a replacement of every runtime
edit price. Runtime retains original duration discounts and the existing movement model;
the single-ayah training clips do not supervise passage jumps.

## Validation

Twelve new test bodies passed using Node 22's TypeScript stripping and a temporary assertion
adapter. This is focused source-runtime verification, not a TypeScript semantic build or a
full Vitest result. The adapter and copied sources were removed. Tests cover:

- Observation and source-selection normalization, including absent sources and independent control.
- Excluding current, overlapping, duplicate-timestamp and retired source observations.
- A cheaper source-linked copy, charged ordinary insertions and unchanged duration discounts.
- Complete-string probability normalization over alternate latent edit/copy paths.
- Latent fitting, normalized class backoff and unknown training-phone rejection.
- A learned copy dependence beating a fitted pooled insertion control on a new synthetic phone order.
- Source-history rewind and exact ordinary local-position equivalence without a model.
- A real synthetic word omission retaining its charged jump.
- Excluding mistake/repeat/unscored/excluded takes and held-out/mixed-split speakers;
  deterministic whole-speaker validation groups.

Node syntax checks passed for the fitter, audit and generated instrumented tracker/local
modules. `git diff --check` passed. No scorer, lab tooling, manifest, evidence or test data
was edited. New tests generate their synthetic observations in code.

Supervisor requests, all returning the same NotFoundError with no command log or usable output:

| Run | Requested work |
| --- | --- |
| 1 | Supplied target trace CLI at 121.84 s |
| 2 | Read-only target local-cost/traceback/source audit |
| 3 | Core/SDK build and training-only linked/control calibration |
| 4 | Required six-shard core/SDK build and tests |

Practice scores requested: **0 of 3**. Target wins, component and overall changes, clean/studio
effects and the fold verdict are unknown. An infrastructure outage does not establish either
premise kill. In particular, unavailable target evidence is not a measured count of zero wins.
No killed outcome is written and no fold/discard decision is taken.

## Remaining work after supervisor recovery

Run the final required checks and training through the supplied supervisor:

```sh
python3 /Users/rock/ai/projects/tilawa/tools/autolab.py modal-run --shards 6 -- 'bash tools/autolab_tests.sh'
python3 /Users/rock/ai/projects/tilawa/tools/autolab.py modal-run --timeout 60 -- 'pnpm --filter @tilawa/core... --filter @tilawa/sdk... build && COPY_CHANNEL_CACHE=/scratch/tracker-delayed-phone-copy-channel/copy-decodes pnpm --filter @tilawa/sdk exec tsx src/copy-channel-fit.ts'
```

Freeze the returned model selection before examining target preferences. The supporting
`.scratch/lab/copy-premise-audit.ts` reads the card's actual listener evidence and records
candidate edit tracebacks, insertion-source timestamps, raw relative costs and product events.
It instruments temporary copies of engine modules and removes them, leaving original source
and scorer untouched. `AUDIT_MODEL` optionally supplies either frozen model:

```sh
python3 /Users/rock/ai/projects/tilawa/tools/autolab.py modal-run -- 'AUDIT_SESSION=lab-v1.5-p46 AUDIT_AT=121.84 pnpm --filter @tilawa/core exec tsx /app/.scratch/lab/copy-premise-audit.ts'
```

Replay with each returned model over the identical interval. Establish that the near path
actually uses an insertion with a distinct earlier source and receives a relative advantage
over the independent control. Check the resulting confirmed warning, including the known
possible listener warning, rather than counting a changed cursor as a target win. Install
the frozen default only after real training and target validation.

Apply the user's premise/fewer-than-three-target-wins kill rule only to measured results.
If killed, revert the engine/tests and retain the measured write-up plus killed outcome.
If three or more targets win, obtain the required practice score before considering narrowing;
the supervisor's fold gate handles losses and the final decision.

## Limits

No real-data calibration, insertion-source attribution or independent audio opinion was obtained.
Distinct timestamps are eligibility evidence, not proof that a sound is an echo. Delay is measured
in consumed-token lag, with timestamp separation, rather than continuous acoustic latency.
Earlier history before a new local search is not seeded; sources begin with observations consumed
by that search. The model retains the original duration handling and does not learn duration or
movement prices. Full build compatibility, CPU performance and warning effects remain unverified.
