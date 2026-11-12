# Censored duration observations

Status: needs-human — the supervisor's Modal infrastructure failed before returning any worker
output. The untrained prototype and drafted unit tests were removed. This is not a premise kill.

## Prediction, before training or replay

Tracker −1 and overall 0 on practice phone. Correcting the continuation binding could remove
`lab-v1.5-p46/phone/warning:skipped_passage/54:36:1/121.84` and expose the documented nearby
listener false warning. This prediction and the earlier exact-duration prototype's 0.3479-edit
disadvantage come from the supplied card and experiment history, not new measurements here.

## Mechanism and inspected implementation

The current `trackerInput` strips frame posteriors but retains each decoded phoneme's inclusive
start/end frames. `tracker.ts` passes only the phone identity to `local.step`; `local.ts` applies
the ordinary weighted edit recurrence, including pronunciation-length deletion costs. There is
no current duration scorer to modify. The earlier tempo and duration-obligation branches named
in the history are unavailable in this checkout. Therefore the experiment needs both a trained
duration scorer and the otherwise identical exact-duration control, rather than a replacement
constant in an existing scorer.

A preliminary engine module represented every observed run with its emitted lower bound and
adjacent unassigned frames on an explicit common comparison horizon. A forward sum retained the
preceding sound's right extension as its state. The next sound could claim only the remaining
frames in their shared gap; frames could also remain unassigned. A joint left/right extension
prior was normalized over feasible configurations, preventing additional boundary choices from
supplying free probability. Reference duration obligations indexed separate discrete duration
distributions, and ordinary inserted observations used a pooled distribution. An exact control
used the same distributions with both extensions fixed to zero. Empty observations supplied no
fabricated sound or duration evidence.

The prototype included a weighted-count fitter with hierarchical smoothing. Its counts must
come from time-registered original/degraded training pairs; no counts were collected or fitted.
In particular, the original recording's own CTC spike boundaries cannot simply be called true
articulation boundaries: training must retain that uncertainty as well. Speaker partitioning,
paired decoding, boundary supervision, training-only model selection, reference-articulation
compilation and runtime tracker integration were not completed.

A temporary read-only traceback audit was drafted to retain candidate paths and observations
through the unchanged local recurrence, including snapshot restoration. It was not executed on
the target. No target gap, duration charge, candidate preference or warning was measured.

## Validation

Lightweight Node assertions exercised six proposed behaviors: inclusive run lengths and exclusive
gaps; rejection of overlapping runs; a shared-gap forward sum matching hand enumeration;
retention of distinct written length obligations in the exact control; agreement with exact
scoring when no extension frames exist; no duration evidence for a missing observation; and
normalized smoothed training distributions. These are synthetic checks of the preliminary
module, not core/sdk suite results, acoustic validation or evidence for the hypothesis.

The authorized supervisor received three requests, with no more than two outstanding:

1. Run 1: inspect the authorized help-train directory, label structure, available evidence and
   target evidence filenames. It failed with `NotFoundError: No such file or directory` before
   returning a worker log or any requested data.
2. Run 2: `bash tools/autolab_tests.sh`, six shards, ten-minute worker timeout.
3. Run 3: minimal `pwd` diagnostic, one shard, five-minute worker timeout.

Runs 2 and 3 also failed with exactly `NotFoundError: No such file or directory`. No request
returned a worker log or shard output, even for `pwd`. These failures do not establish a data,
code or storage defect; the earlier experiments' disk-full failures were not diagnosed or
reproduced here. No Modal credentials or alternative execution route were sought.

## Limits and continuation

Practice scoring used zero times. Target wins and score changes are unknown. Neither the
training-frozen comparison with the exact control nor the required target-win check has been
evaluated, so a zero-win claim or premise kill would invent a measurement.

Once supervisor execution works, collect authorized training pairs, freeze the censoring and
duration distributions using training-speaker partitions, and evaluate both local candidates
over the identical retained interval with the exact control. Keep genuinely absent articulations
as missing references, preserve movement costs and commitment margins, and run the core/sdk
suites. Apply the experiment's target-win stopping rule before any practice-score request.

No engine changes remain on this branch: shipping the untrained module would not implement the
training-frozen experiment. The experiment card, scorer, lab tooling, data, manifests and evidence
have not been edited. `.autolab/outcome.json` records `needs-human`, with no measured score or
target-win count.
