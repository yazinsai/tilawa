# Censored audio intervals

Status: ready-for-human

The experiment is blocked by the authorized Modal executor. No engine change
has been made, no target audio has been inspected, and no practice score has
been requested. This is not a premise kill or a measured zero gain.
The experiment card is unchanged.

## Prediction and mechanism

The card predicts listener −1 and overall −1: one omission false warning
removed without losses, with substantial risk of no gain. Its premise is a
demonstrable digital erasure overlapping a disputed spoken word, rather than
ordinary silence or acoustic misrecognition.

The proposed SDK detector requires constant-sample intervals with abrupt entry
and exit inside otherwise active speech. Detection and abstention must be
calibrated on authorized help-train training recordings with injected digital
dropouts and genuine pause controls. At an existing single-candidate omission,
complete-read and omit calculations would marginalize the same unavailable
frames under both hypotheses. An uncertainty certificate would be supplied
only when missing-observation treatment reverses the ordinary omission
preference. Primary tokens, tracker events, placement, and ordinary warning
requirements would be preserved. No replacement waveform would be generated.

The card's kill criterion is fewer than one target containing a qualifying
erasure that reverses the preference. The implementation brief also requires
at least three target wins before practice scoring. Neither condition can be
evaluated without actual audio and sequence evidence; failure to launch a
container establishes neither condition.

## Premise audit prepared

Read AGENTS.md, the card, the lab spec, and the audio-review instructions first.
Inspected the SDK's WAV decoder, streaming listener, frontend, and session
evidence path. The frontend uses a 400-sample window and a 160-sample hop at
16 kHz. Low energy and CTC blank are insufficient evidence of unavailable
samples.

A read-only diagnostic was prepared under the ignored
`.autolab/dropout_probe.py`. It scans exact sample equality without a
practice-selected duration threshold, reports constant-run lengths and boundary
jumps, and counts runs covering complete SDK frontend windows. File hashes and
WAV format metadata accompany the reports. Target references are diagnostic
arguments only. Supplied labels select every spoken occurrence of each target
word, its surrounding five seconds, and its whole ayah; both complete phone and
studio recordings are also scanned. Warning timestamps do not determine the
word's audio interval.

The requested targets were:

- `lab-v1.5-p50/37:113:10`
- `lab-v1.5-p65/4:12:40`
- `lab-v1.5-p65/4:15:6`

The audit deliberately precedes implementation. If no candidate constant
interval exists, there is no basis for missing-frame marginalization. If
candidates exist, training-only calibration must distinguish erasures from
natural pauses and establish the missing-frame mask before target preference
comparisons. Complete frontend-window coverage is a diagnostic, not a
calibrated detector or an uncertainty decision.

## Validation attempted and limits

All requests used the supervisor's documented `tools/autolab.py modal-run`
interface. No alternate Modal access or external data path was attempted.

| Request | Purpose | Result |
| --- | --- | --- |
| 1 | Inspect practice audio/label layout and help-train availability | `NotFoundError: No such file or directory` |
| 2 | Run the file-backed sample audit | Same executor error |
| 3 | Run the audit embedded directly in the shell command | Same executor error |
| 4 | Supplied six-shard `bash tools/autolab_tests.sh` harness | Same executor error; no tests executed |

Requests 1–4 produced no shard logs or returned audio diagnostics. The error
alone does not identify which remote file or resource is missing. It must not
be interpreted as a missing target recording. Request 2 additionally used an
ignored diagnostic path excluded by the source snapshot; request 3 corrected
that packaging mistake by embedding the complete source in a quoted heredoc,
and still failed before container output. The full corrected command remains
in `.autolab/modal-request-3.json` for a retry after executor recovery.

Small local diagnostic self-checks passed: changing samples, constant samples,
exact frontend-window grid coverage, boundary jumps, PCM16 WAV decoding, and
float32 WAV decoding. These checks validate the audit itself; they are not
engine tests, acoustic calibration, independent listening, or target evidence.
The core/SDK build and test suites remain unverified because request 4 failed
before executing the harness.

Targets evaluated: zero. Target wins: unknown. Training calibration: not run.
Practice scores used: zero of three. Overall and component deltas: unmeasured.
No model bundle or `.autolab/model.json` is declared. There are no engine or
unit-test behavior changes to retain or revert. No scorer, lab tooling,
supplied manifest, label, evidence, or card was modified, and no fold was
performed.

`.autolab/outcome.json` records `blocked`, not `killed`, with zero targets
evaluated and no claimed target-win count or practice delta. The diagnostic
source SHA-256 is
`bec0e241a1f811c21af2a5c0ddf3ed819242d7f2bfa848dd0a8b902bb39aba5b`.

Resume by restoring the authorized executor and rerunning the embedded audit,
then perform the training calibration and read-versus-omit comparison if its
premise survives. A blocked outcome must remain distinct from `killed` until
the requested premise or target-win measurements can actually be made.
