# Tilawa

An offline engine that listens to Quran recitation, follows where the reciter is, and reports memorization and pronunciation mistakes.

## Pipeline

**Listener**:
The acoustic model that turns recitation audio into phoneme evidence.
_Avoid_: ASR, recognizer

**Teacher**:
The large listener (`muaalem-model-v3_2`) whose outputs are distilled into the student.

**Student**:
The small on-device listener distilled from the teacher, shipped in the `lite` and `full` tiers.

**Evidence**:
The listener's per-frame output: greedy phonemes with confidence and, optionally, top-k posteriors.

**Tracker**:
The component that finds and follows the recited passage from evidence and emits `position` events.
_Avoid_: aligner, locator

**Checker**:
A separate consumer of the tracker's locked alignment and the evidence that decides which deviations are recitation errors and emits checker events.
_Avoid_: grader, scorer

## Position

**Expected bookmark**:
Where the reciter is supposed to be, given the session's mode and passage.

**Observed bookmark**:
Where the evidence says the reciter actually is.

**Lock**:
The tracker state of being confident in one observed position.

**Jump**:
A committed move of the observed bookmark to a non-continuous passage.

## Errors

**Error class**:
One kind of mistake the checker can report, such as an omitted word or the ح/ه confusion. Each class is either supported or `unsupported`.

**Resolved**:
A suspected error the reciter fixes within the policy window, such as a restart on the right word or reciting skipped words before the next ayah ends.
_Avoid_: corrected, cleared

**Canonicalization**:
The listener's bias to output the correct recitation instead of what was actually said.

**Scripted-error recording**:
A volunteer reading of a generated script with planted errors, so the labels are known by construction.
