# Listener runlength supervision

Status: killed on the training premise. The experiment card is unchanged.

## Prediction and mechanism

The card predicts two false wrong-word warnings removed and one collateral wrong
call: listener −1 and overall −1. Its proposed mechanism removes
pronunciation-generated identity-label repetitions from CTC training, annotates
their length separately, and jointly trains an identity head and a duration head
with marginalized articulation boundaries. Genuine repeated articulations remain
separate. The comparison arm retains expanded identity labels with identical data,
capacity and auxiliary duration supervision; checkpoint selection must use only
training-speaker partitions. Tracker and checker rules remain unchanged.

The explicit card criterion requires at least two complete written-pronunciation
preferences recovered relative to that control on actual engine-requested windows.
The task also requires three target wins before practice scoring. A disproved
training premise permits stopping before either acoustic test.

## Training-premise audit

The deployed primary listener is `spliced-clean-int8`. Its release training record
in `docs/benchmarks/student-v2.md` and `student-v2.json` identifies the CTC target
as “spliced reference, runs collapsed,” with `target="spliced"`,
`init="librispeech-m"`, and `kd_weight=2`. It does not start from a previously
trained Quran student with expanded identity labels.

The current training code implements that recorded recipe:

- `modal/student_train.py` constructs each target with `spliced_target(ref, hyp)`
  before `ClipDataset` converts its characters to identity IDs for `ctc_loss`.
- `tools/student.py:spliced_target` returns `collapse(spliced(ref, teacher))`.
  `tools/per_breakdown.py:collapse` removes every consecutive duplicate character.
  Consequently, the resulting CTC target has no adjacent equal labels, including
  the pronunciation-generated long-vowel, nasal and shadda repetitions.
- The existing unit tests explicitly require reference and teacher-edge runs to
  collapse. Ordinary greedy CTC decoding separately preserves equal emissions
  separated by a blank; this is not evidence of expanded training targets.

The training target compiler also lacks articulation and word-boundary provenance:
it collapses all adjacent equal labels. That can motivate a different experiment
about preserving genuine repetition, but does not establish the card's proposed
artificial within-articulation blank requirement.

## Validation and limits

Two authorized CPU Modal diagnostics were requested: run 1 to inspect the shipped
checkpoint configuration, and run 2 to regenerate targets from `/student/s1`
with the shipped clean-match/train selection, 30-second cap and gold exclusions,
then execute `tools/test_student.py`. Requests were written at
2026-11-12 02:27:19 UTC and 02:28:16 UTC. Neither returned a result after more than
35 minutes; both request files were withdrawn. There are no checkpoint-inspection
results, corpus counts or completed Modal unit-test results to report. The
conclusion rests on the current compiler and the independently recorded release
recipe, not on those uncompleted diagnostics.

Soft top-five distillation still supervises framewise teacher probabilities and
can influence emission timing. It is not a duration head, and its effects are not
proved harmless by this audit. However, it does not place duration-expanded
identity sequences in the student's CTC target. Adding a duration head or altering
teacher timing supervision would need a separately justified training mechanism;
introducing an expanded-label control would not reproduce the shipped recipe.

No target recovery or aggregate score is claimed from source inspection. No GPU
training, checkpoint selection, practice score or release-held-out evaluation has
been performed. The inference-only topology probe's two local recoveries do not
establish that this student was trained with the same expanded-label topology.

## Outcome

The specified defect is absent from the shipped student's CTC supervision: every
adjacent duration-expanded label is removed before training. The artificial blank
separations proposed as the cause therefore cannot be imposed by those targets.
This is a premise kill, not a claim that an identity-plus-duration model cannot
help, and not a kill based on collateral losses. No matched models were trained,
so no actual engine-requested target preference comparisons were performed.

Zero target wins are claimed; the overall practice change is unmeasured. All
three permitted practice scores and the $90 GPU budget remain unused. No engine,
scorer, lab tooling, supplied manifest, test data or evidence was changed, and no
model bundle is declared. Because there is no behavior change, there are no new
unit tests or core/SDK regression-test claims. Only this write-up is committed;
the killed outcome is recorded in the ignored `.autolab/outcome.json`. No fold
was performed.
