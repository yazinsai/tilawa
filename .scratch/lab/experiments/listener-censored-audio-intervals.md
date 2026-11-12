# Censored audio intervals

Status: ready-for-human

**Implementation cannot be completed in this environment:** every authorized
Modal request failed with `NotFoundError: No such file or directory` before
returning container output. This includes a minimal Python print command that
uses neither audio files nor repository code. No target audio or help-train
recording was inspected. This is an executor blocker, not a premise kill or a
measured zero gain. The experiment card is unchanged.

## Prediction and mechanism

The card predicts listener −1 and overall −1: one omission false warning removed
without losses, with substantial zero-gain risk. The premise is a demonstrable
digital erasure overlapping the disputed spoken word. Low energy, a long CTC
blank run, or a natural pause is insufficient evidence of unavailable samples.

The proposed SDK detector requires exact constant-sample intervals with abrupt
entry and exit inside otherwise active speech. Its detection and abstention
criteria must be calibrated using authorized help-train training recordings
with injected dropouts and genuine pause controls. Release-held-out speakers
must remain excluded. At an existing single-candidate omission, complete-read
and omit hypotheses would marginalize the same unavailable observations. Only
an omission whose ordinary preference reverses under that calculation would
receive an uncertainty certificate. Primary tokens, tracker events, placement,
and ordinary warning requirements would be preserved, with no replacement
waveform.

The card's kill criterion requires an actual target erasure and a preference
reversal. The implementation brief additionally requires at least three target
wins before scoring. Neither can be measured when the executor cannot run even
a print command. Target wins remain unknown; they are not zero.

## Executable prototype and why it was removed

The retry implemented SDK source and unit tests in commit `13417583`:

- `constantAudioIntervals` enumerated maximal exact PCM plateaus, recording
  half-open sample bounds, the held value, and entry/exit jumps. It explicitly
  did not certify equality alone as a dropout.
- `CensoredPosteriorFrames` supplied unit emission likelihood for every class,
  including blank, at certified unavailable frames. It reused the ordinary CTC
  recurrence, retaining elapsed time and its transitions instead of deleting
  frames or substituting observed silence. Stored posterior rows were unchanged.
- `omissionDependsOnMissingAudio` required finite scores and a strict change
  from omit preference to read preference, with both hypotheses using the same
  unavailable-frame mask. Intact audio with no mask received no exception.
- Unit tests covered exact and nonzero plateaus, stream edges, low-energy
  changing samples, exhaustive small CTC path sums, retained frame timing,
  unchanged source evidence, strict reversal and invalid masks.

Small local checks of that source passed using Node's TypeScript transform on
copies whose imports alone were redirected. They checked exact intervals,
exhaustive CTC path sums, unchanged evidence, retained frame timing, and
reversal. They did not run Vitest, the SDK/core suites, acoustic inference,
training calibration, target replay, or scoring.

This was an executable calculation prototype, not a completed warning-path
implementation. No calibrated detector or uncertainty certificate was connected
to the listener/session/checker path. Choosing dropout thresholds without the
required training controls could mistake genuine pauses for missing audio;
connecting that unvalidated mask to warnings would not implement the card's
specified mechanism. The authorized data and runtime are only on Modal, local
`node_modules` are absent, and alternate compute/data access is prohibited.
These constraints prevent completing and verifying the change here.

Commit `2cecf250` removed the prototype from the final engine diff. Its source
and tests remain recoverable from `13417583` for a continuation after executor
recovery. The rollback does not claim the mechanism failed.

## Validation attempts and limits

All requests used the documented supervisor `modal-run` interface. The original
attempt's requests 1–4 also returned the same executor error without logs.
This retry made four further requests:

| Request | Purpose | Result |
| --- | --- | --- |
| 5 | Embedded exact-sample target audit | `NotFoundError`, no shard logs |
| 6 | Six-shard core/SDK build and test harness | Same error; no tests executed |
| 7 | Six-shard harness with the SDK prototype and unit tests present | Same error; no tests executed |
| 8 | Minimal `python3 -c "print(12345, flush=True)"` | Same error; no output |

Each returned result reports zero dollars spent. The results do not identify
which external resource is missing. Request 8 demonstrates that the failure
does not require opening a target recording or importing the prototype. It is
not evidence that any recording contains, or lacks, a dropout.

The ignored read-only audit was corrected to resolve supplied session/audio
paths through the supplied manifest, rather than guessing audio filenames in
the sessions directory. No supplied manifest was edited. Its embedded retry
command is saved in `.autolab/corrected-dropout-request.json`, and its source
SHA-256 is `a28e84125961758babd40f67d1201acbad4da6584e16edd3c2e6f5d2798b73f6`.
That corrected audit has not executed remotely. It reports every labelled
spoken occurrence of the target words, nearby and whole-ayah windows, exact
run lengths and boundary jumps, WAV metadata/hashes, and complete frontend
window coverage. Labels select diagnostic windows only, never engine rules.

Targets evaluated: zero. Target wins: unknown. Calibration: not run. Practice
scores used: zero of three. Component and overall deltas: unmeasured. The full
core/SDK suites remain unverified. No model bundle or `.autolab/model.json` is
declared. No scorer, lab tooling, supplied data, manifest, evidence, or card was
modified. No fold was performed. `.autolab/outcome.json` records `blocked`.

Resume by restoring the authorized executor, running the corrected audit, and
calibrating the detector on authorized training recordings if the premise
survives. Then integrate certificate transport and the omission-only uncertainty
rule, validate target reversals and the complete SDK/core path, and apply the
brief's kill/scoring requirements. A blocked result must not be converted to a
premise kill without those measurements.
