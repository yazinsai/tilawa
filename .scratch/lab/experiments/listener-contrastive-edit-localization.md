# Contextual edit localization

Status: blocked by the supervisor compute service; acoustic premise untested.
The experiment card is unchanged. No engine change or model override is left
on the branch, and no fold or discard was performed.

## Prediction and mechanism

The card predicts two false warnings removed and one caught mistake lost:
listener −1, overall −1. A small acoustic/text head would predict edit location
and class jointly across a recording, using frozen shipped encoder features.
Scripted help-train locations would provide weak recording-level supervision,
with canonical acoustic boundaries marginalized rather than assuming performed
word transcripts. None-labelled and repeat recordings would be explicit controls.
Release-held-out speakers would be excluded; fitting, model selection and
substitution-preservation calibration would use disjoint training-speaker groups.

The jointly trained head would be distilled into per-word lexical-change scores
with independent sigmoid outputs, allowing multiple changes at inference. The
SDK would supply a span-specific veto certificate only for an existing
string-path wrong-word warning whose contextual score falls below the frozen
preservation operating point. Primary decoding, tracker events, graph
confirmations and insertion decisions would be preserved. An independently
trained per-word classifier would receive the identical acoustic/text features.

The card's diagnostic requires at least two targets rejected by the contextual
model while retained by that independent control. The task additionally requires
at least three target wins before continuing to practice scoring. Neither
condition has been evaluated; zero tested targets is not evidence of zero wins.

## Work and validation

Read `AGENTS.md`, the card, and the current listener, checker, model loader and
teacher-insertion certificate transport. The existing SDK path transports
auxiliary metadata on primary tokens through serialization and frame-cell
rechunking, providing a possible route for the proposed certificate.

Drafted a training utility and small tests. The draft exposes the pre-CTC frozen
encoder from a separate ONNX copy, pools acoustic/text features using canonical
CTC forward-backward occupancy, and fits recording-competition and independent
controls. It includes latent script-compliance supervision, explicit repeat
classes, grouped model selection, sigmoid distillation and substitution-based
calibration. This draft has not been trained, exported or connected to the SDK.

Python syntax compilation passed. Five small mathematical tests passed locally
using `arch -x86_64 python3`: boundary marginals matched exhaustive CTC path
enumeration, impossible alignments abstained, pooling retained multiple word
inputs, repeats mapped to an explicit control class, and speaker partitions were
stable. Native Python could not import its installed x86_64 NumPy extension;
the matching interpreter architecture ran the tests successfully. These checks
do not validate acoustic quality, ONNX feature discovery or model fitting.

Three requests through the user-specified supervisor failed:

| Run | Requested work | Result |
| --- | --- | --- |
| 1 | Inspect authorized help-train labels and model outputs | `NotFoundError: No such file or directory` |
| 2 | Minimal working-directory, storage and help-train inspection | Same error |
| 3 | Supplied six-shard `bash tools/autolab_tests.sh` | Same error |

All three returned only an error result, with no downloaded container log,
data listing, test output or acoustic result. Their result files report zero
spend. The failed path and underlying cause are not established; this must not
be described as a confirmed storage-capacity failure.

## Limits and disposition

No help-train audio was processed, no contextual or independent model was
fitted, no preservation point was frozen, and none of the four registered
incidents was evaluated. No practice-score invocation was made; all three
permitted practice scores remain unused. The required core/SDK checks were
attempted on Modal and could not run through the supervisor.

This is an infrastructure block, not a premise kill or a measured negative
result. The hypothesis remains open. The unvalidated engine draft and its tests
were removed from the source snapshot and retained under the ignored
`.autolab/localization-draft/` directory for a retry. There is no
`.autolab/model.json` and no declared incomplete bundle. No scorer, lab tooling,
supplied manifest, evidence, test data or experiment-card changes were made.

Resume by restoring the authorized supervisor compute route, inspecting the
actual training-speaker data and frozen feature graph, then completing training,
export, runtime certificates and the identical-feature control. The complete
bundle and actual SDK transport must be verified before reporting target wins
or handing the branch to fresh listener decoding and the fold gate.
