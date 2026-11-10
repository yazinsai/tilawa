# Joint phone coalescence

Status: needs-human. All three supplied supervisor requests failed before execution with
`NotFoundError: No such file or directory`. The channel, estimator and local integration are
implemented, but training calibration, target replay, default activation and the required
build/Vitest checks remain unfinished. The experiment card is unchanged.

## Prediction and premise

Tracker −1, overall 0 on lab-v1.5u practice. Target:
`lab-v1.5-p46/phone/warning:skipped_passage/54:36:1/121.84`.
The brief reports the near continuation available but 0.3915 edits behind the rhyme restart
over the same heard interval. That number is supplied evidence, not a measurement reproduced
here. A correct location can expose the documented nearby listener false wrong-word warning;
the forecast includes that possible loss.

The speculative premise is that two distinct reference articulations produced one decoded
token, but independent deletion and substitution transitions price that observation as separate
failures. The kill condition requires a training-frozen preferred coalescence explanation on
the near continuation and a resulting win over the restart. Neither condition could be measured
without the remote training data. No premise kill or recovered warning is claimed.

## Mechanism implemented

`packages/core/src/coalescence.ts` defines a normalized recognition-event channel. Each pair's
row partitions probability between an ordinary one-articulation event and each possible
two-reference-to-one-heard-token outcome. Exact phone pairs back off to fixed manner-class pairs,
then to pooled event counts. Classes depend on the phone inventory alone.

The training transducer aligns complete written and heard recordings with latent insertion,
deletion, ordinary recognition and coalescence paths. Forward/backward supplies fractional event
counts; hierarchical EM runs at most 30 iterations. Its ordinary emissions normalize the
existing exponential edit weights at the tracker's existing conversion of three nats per edit.
Insertion has fixed mass 0.1 and uniform phone outcomes, and the end state partitions insertion
and stop probability. These are training-model definitions, not practice-fitted parameters.
The pooled prior assigns 0.95 to ordinary events and 0.05 across merged outcomes, with two total
pseudocounts. Exact and class rows shrink by the training-selected pooling mass.

Local matching in `packages/core/src/local.ts`, called by `tracker.ts`, adds consuming transitions
from the immutable previous observation row at reference offset minus two. Such a transition
consumes both articulations and the current token exactly once and records one recognition
edit. Ordinary reference-consuming transitions receive `−log P(ordinary)` in existing edit
units. The merged transition's charge is the negative log likelihood ratio of its event
probability to exact ordinary recognition of the two consumed articulations, using the same
conversion. Probability allocated to a merge is removed from the ordinary branch; adding an
alignment route does not supply free counting weight.

The same channel applies to continued local paths and to the first recognition event of an
existing word-start restart. Candidate destinations, jump price/distance, surah entry prices,
search window, commitment margins and downstream handling retain their definitions. The DP
compares all candidates on the same sequence of heard observations. No checker code changes.

Identical adjacent phone identities, either identity touching a repeated reference run, and
pairs crossing a word boundary are excluded. This conservatively excludes pronunciation-generated
duration repetitions without collapsing them or removing their existing charges. A merged
event cannot absorb a written word boundary or a whole omitted word. Ordinary insertion and
duration edit definitions remain in place.

The optional `TrackerOptions.coalescence` carries a frozen model. There is no fabricated default:
ordinary product and practice invocations still use the original local costs until a returned
training model is installed. Scoring this unfinished branch's default engine would therefore
not test the hypothesis.

## Training isolation and model selection

`packages/sdk/src/coalescence-fit.ts` opens only help-train labels/audio, the shipped reference
and `models/spliced-clean-int8`. It decodes using the SDK listener and caches under the model
hash. Eligibility keeps scored, non-excluded `kind: none`, `split: train` takes with empty
events and non-errors. Any speaker with any non-training label is excluded completely, including
a mixed-split speaker. No practice evidence or release-held-out recordings are opened.

Up to five deterministic speaker-grouped partitions select no coalescence or pooling masses
8, 32 or 128 by held-partition predictive log probability. The no-coalescence model is an
explicit candidate and wins exact selection ties. Selection then freezes and the model refits
on all eligible training speakers. Outputs are `coalescence-model.json` and `fit-report.json`
in `LAB_OUT`, recording the listener identity/hash, eligible-label hash, recording and speaker
counts, and every partition result. No training output has been returned.

## Validation and limits

Seven core and two SDK focused test bodies passed locally under Node 22 TypeScript stripping
with a temporary assertion adapter for Vitest. They check:

- Normalized complete-string likelihood: summing all two-phone-inventory heard strings through
  length eight yields 0.9999980398649666, with the small remainder in longer insertion strings.
- Exclusion of repeated/duration symbols, identical identities and word boundaries.
- Exactly two reference phones and one observed token consumed by a merged event, one edit,
  and equivalent state after save/restore and replay.
- Identical ordinary local positions/costs with no coalescence selected.
- The same merged recognition event at a word-start restart.
- A genuinely omitted word still produces a charged movement jump.
- Normalized hierarchical EM rows, predictive improvement on synthetic merged recordings,
  phonetic-class backoff and rejection of unknown training phones.
- Training eligibility excludes mistakes, repetitions and non-training/mixed-split speakers.
- Deterministic partitions keep all recordings of a speaker together.

This is synthetic numerical/behavioral validation, not a TypeScript build or a required full
Vitest result. The temporary validation files are removed. `git diff --check` passes.

Supervisor requests, all failed before command execution with the same NotFoundError:

1. Inspect authorized help-train labels and scratch availability (one shard).
2. `bash tools/autolab_tests.sh` (six shards).
3. Core/SDK build followed by training-only calibration (one shard, 60-minute limit).

No model, labels, target trace, build logs or test logs were returned. Practice scores used:
**0 of 3**. Target wins, component delta, overall delta and gate verdict are unknown. An
infrastructure error supplies no evidence that the mechanism is false and is not a premise kill.

## Resume

After the supplied supervisor runner's missing resource is restored, run:

```sh
python3 /Users/rock/ai/projects/tilawa/tools/autolab.py modal-run --timeout 60 -- \
  'pnpm --filter @tilawa/core... --filter @tilawa/sdk... build && COALESCENCE_CACHE=/scratch/tracker-joint-phone-coalescence/clean-decodes pnpm --filter @tilawa/sdk exec tsx src/coalescence-fit.ts'
python3 /Users/rock/ai/projects/tilawa/tools/autolab.py modal-run --shards 6 -- \
  'bash tools/autolab_tests.sh'
```

Freeze the returned selection before opening the target. If enabled, supply that exact model
to both local candidates, inspect their complete common heard interval and the merged path's
reference ownership, then replay through the unchanged checker. Install a default only from
that frozen model; no practice-driven choice of mass or event price is authorized.

If no preferred merged explanation exists, the near continuation still loses, or the built
change wins fewer than three targets, follow the user's kill rule: revert engine changes and
tests, retain the measured write-up, and commit a killed outcome. If three or more targets win,
obtain the required practice score before a gate decision. No warning suppression or further
repair of the nearby listener false warning is part of this experiment.
