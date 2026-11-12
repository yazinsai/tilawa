# Cross-encoder acoustic bridge

Status: blocked by supervisor execution/transport; implementation incomplete.
No acoustic finding or deployable model exists for this experiment.

## Prediction

The unchanged card predicts listener −2 and overall −2 wrong calls on
lab-v1.5u, with substantial zero-gain risk. The motivating observation is the
earlier undistilled SSL ear's reported retention of ordered planted consonants
on five of six seams despite few eligible whole-word decodes. That observation
is supplied history, not a result reproduced in this worktree.

Declared baseline: `exp-listener-teacher-insert-marginal-gate-ad6eb8c0`.
The five registered targets are all unevaluated:

- `lab-v1.5-p07/phone/mistake:added_word/37:36:1/144.12`
- `lab-v1.5-p48/phone/mistake:added_word/12:76:1/39.98`
- `lab-v1.5-p56/phone/mistake:added_word/32:21:1/106.76`
- `lab-v1.5-p57/phone/mistake:added_word/51:18:1/95.26`
- `lab-v1.5-p57/phone/mistake:added_word/51:20:1/112.26`

## Mechanism and planned comparison

Freeze the independently speech-pretrained source encoder and the shipped
acoustic stem. Inject timestamp-aligned source features through a residual
bridge with a zero-initialized output projection before the shipped upper
encoder, then allow limited upper-block adaptation with one ordinary CTC head.
The controls must use the same recordings, trainable parameter count and update
budget: a shipped-feature bridge and a temporally shuffled source-feature bridge.
The source encoder and bridge must ship together as declared SDK assets and
produce the primary token/posterior stream on the existing 40 ms output clock.
Tracker, checker and insertion verification requirements remain unchanged.

Training and checkpoint selection belong entirely to speaker-separated training
partitions of authorized transcribed recordings and help-train training speakers'
none-labelled takes. Mistake scripts are not performed transcripts. Ordinary
speech and real pauses must remain in the preservation data, and release-held-out
speakers must remain excluded. Architecture and selection must be frozen before
target diagnostics. No alternate checkpoint may be substituted if the intended
source is unavailable.

The card asks for at least two targets to gain a previously missing planted
consonant relative to both controls. The task's final rules also require three
target recoveries before practice scoring. Neither condition has been tested.

## Preflight and validation

Read AGENTS.md and lab-experiment.json, and inspected the shipped training,
streaming export, listener frontend, SDK model loading and unit-test interfaces.
The shipped student/export interfaces take 80-bin fbank windows, carry explicit
encoder states, and emit one 40 ms CTC stream. Any bridge needs a bounded source
feature extraction path with explicit timestamp alignment, including isolated
gap decoding; a separately decoded phone string would not implement this card.

The referenced `exp/listener-ssl-ctc-undistilled-ear` branch is absent from this
checkout's refs, and its implementation/checkpoint provenance is absent from the
available git history. The supplied experiment history identifies w2v-BERT-2.0
as that ear's source. Preflight therefore requested the supplied model/prior-run
inventory and public `facebook/w2v-bert-2.0` repository metadata through the
authorized supervisor, rather than choosing another model.

- Modal request 1: CPU training image, supplied prior-run/model/cache inventory.
  Supervisor returned `NotFoundError: No such file or directory`, with zero
  reported cost and no command output or inventory artifact.
- Modal request 2: CPU training image, public source metadata/config retrieval.
  Supervisor returned the same error, with zero reported cost and no command
  output or source artifact.
- Modal request 3: L4 training image, combined source and inventory preflight.
  Still pending at the blocked handoff (2026-11-12 20:26 UTC), more than fifteen
  minutes after the request. No command logs or availability artifact returned.
- Modal request 4: ordinary image, supplied six-shard `tools/autolab_tests.sh`.
  Supervisor returned the same `NotFoundError`, with zero reported cost and no
  shard logs or test results.

Requests and supervisor results are retained in ignored `.autolab/` artifacts.
The wrapper also accesses the output directory after waiting for a sandbox, so
these bare errors cannot identify whether startup or output collection failed.
No source revision or checkpoint hash has been verified.
No source checkpoint was loaded, no training data was read by these failed runs,
and no architecture, bridge, controls or model bundle was built. There are no
engine changes to roll back and no new behavior requiring unit tests. Existing
tests were requested remotely but request 4 failed before returning any test
results. No passing result is claimed.

## Limits and disposition

No target was evaluated and no practice score was requested (zero of three used).
Neither acoustic recovery nor target wins, collateral losses, clean-session
warnings, studio drift, export parity or device cost has been measured. The
supervisor errors do not establish that the source repository/checkpoint does
not exist, and do not disprove the acoustic premise. This is an execution block,
not an acoustic kill or a measured zero-gain experiment.

The experiment card, scorer/lab tooling, test data, supplied manifests and
evidence are unchanged. No `.autolab/model.json` selects the main model as a
stand-in for an unbuilt bridge, and no fold was performed. Resume only with a
working authorized supervisor and reproducible intended source checkpoint;
then build and validate all three arms and the complete SDK bundle before any
practice evaluation.

Ignored `.autolab/outcome.json` records `blocked`, with zero targets evaluated
and zero practice scores. It deliberately does not report an acoustic kill,
target-win count or overall change, because none was measured. The outstanding
L4 request and the three failed results remain identified for recovery. An
optional asynchronous request for a recovery step within the supplied
supervisor interface had received no reply at this handoff.
