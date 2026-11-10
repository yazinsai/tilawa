# Foreground voice evidence

Status: ready-for-human

Blocked by the supervisor's Modal execution infrastructure. The acoustic premise
is untested; this is neither a killed experiment nor a validated candidate.
The experiment card is unchanged. No engine change remains on the branch.

## Prediction and mechanism

The registered prediction is listener −1 and overall −1, with one false warning
removed and no predicted losses. The four targets are p11 at 26:106:1, p36 at
55:48:1, p55 at 26:174:8, and realuser sob-049 at 92:13:1.

Build a compact foreground spectral dictionary from earlier confidently matched
speech. Decompose an existing accusation window into foreground and competing
speech with temporally continuous activations, supplying neither candidate word
to the separation. Freeze dictionary sizes, optimization settings and continuity
strength on help-train training speakers mixed with different training speakers,
retaining unmixed controls and speaker-disjoint validation partitions. Decode
the foreground reconstruction with the unchanged shipped model and veto only
after recovering the complete written pronunciation and removing rival support.
Primary tokens, tracker events, parallel-graph confirmations and insertion
decisions must retain their existing behavior.

The premise kill requires no independently identifiable competing voice at any
target's rival-supporting acoustic interval. The task also requires rollback
when the built change wins fewer than three targets. Neither condition has been
measured here. An execution error is not evidence for either condition.

## Work and validation

Read AGENTS.md, the card, the lab specification, the audio-review workflow, the
current checker sequence comparisons, the SDK streaming frontend and the
teacher-certificate transport. The lab builder describes same-speaker splices
and simulated room filtering plus noise from a noise-only bank. This provenance
does not establish whether another voice is present in the original recording;
it cannot by itself kill the hypothesis. No independent audio-review service
was exposed through the connected tools, and no audio was retrieved or listened
to. No listening question about a particular waveform can be resolved from
the text traces alone.

A temporary SDK probe implemented deterministic spectral clustering of earlier
audio, fixed foreground atoms plus learned competing atoms, nonnegative
activation updates with a quadratic temporal-continuity penalty, and
original-phase overlap-add reconstruction. Its API received waveform arrays
and decomposition settings, with no written or rival pronunciation. Five local
Vitest tests passed: insufficient-history abstention, separation of two distinct
simultaneous tones, sample count/source sum/input/dictionary preservation,
determinism, silence/empty-input handling, and invalid-input rejection. Strict
TypeScript checking of the module also passed. These synthetic tests establish
numerical mechanics only: tones do not demonstrate speaker identification or
the ability to preserve real consonants. The test settings were fixture values,
not training-frozen deployment settings.

The uncalibrated, unintegrated probe and its tests were removed from engine
source and retained under ignored `.autolab/foreground-prototype/` for resuming
the work. It never changed a checker decision, loaded an auxiliary model,
decoded a target, or generated a veto certificate. There is no model bundle or
`.autolab/model.json` declaration.

The authorized supervisor interface was used for every remote request:

| Run | Request | Result |
| --- | --- | --- |
| 1 | Inspect the four session records, available evidence and help-train layout | `NotFoundError: No such file or directory` |
| 2 | Four-shard lab-trace CLI using the card's listener and registered times | Same infrastructure error |
| 3 | Basic runtime check using the supported Node image: working directory, LAB_DATA listing, scratch capacity | Same infrastructure error |
| 4 | Required six-shard `bash tools/autolab_tests.sh` | Same infrastructure error |
| 5 | Retry data inspection using the supported training image | Same infrastructure error |

All five runs returned only `.autolab/modal-result-<n>.json` error objects, without
shard logs or diagnostic outputs. They do not establish which runtime file or
service was missing, and they do not justify attributing the current failure to
storage exhaustion. GPU usage was zero. Restoration of the supervisor's Modal
runtime/data access was requested while local work continued.

## Limits and next step

There are no training results, target recovery counts, full-suite results or
practice scores. Zero practice-score calls were used. Target wins and overall
change are unknown, not zero. No killed outcome is declared because the kill
criterion has not fired. No scorer, tooling, supplied data, manifest, evidence
or experiment-card edits were made, and no fold was performed.

Resume by restoring the supplied supervisor runtime, retrieving the exact
accusation windows and preceding accepted speech, and completing independent
overlap triage with calibrated mixed-speaker positives and unmixed controls.
Freeze all decomposition settings on training-speaker partitions before target
tests. Only after three or more target recoveries should the SDK certificate
path be integrated and the required practice score run. Without those steps,
scoring the main model would not evaluate the proposed separation mechanism.
