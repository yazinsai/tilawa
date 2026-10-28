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

**Disambiguation point**:
The first word at which the text recited so far occurs in only one place in the Quran. Lock delay is measured from it, and a confident lock on another place before it is a false jump.
_Avoid_: hook point

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

## Measurement

**Wrong call**:
A labelled mistake the engine missed, or a warning the engine raised where there was no mistake.

**Overall score**:
Wrong calls per 5 minutes of free recitation on the current test set version. Lower is better; it is the one number experiments try to move.

**Component score**:
How well one of listener, tracker or checker does its own job when fed perfect inputs from upstream, so its losses are its own.

**Attribution**:
The split of the overall score's wrong calls into those lost by the listener, the tracker and the checker. The three parts add up to the overall score.

**Spliced session**:
A test case cut from one reciter's continuous recording, minutes long, with mistakes, repeats and restarts planted by editing the audio, so the labels are known by construction.

**Test set version**:
A frozen set of test cases. Scores are only compared within one version; applying flags creates the next version.

**Practice set**:
The part of a test set version, split by reciter, that experiments iterate on with every failing case visible.

**Exam set**:
The rest of the test set version, shown only as numbers and scored at fold-in. The headline overall score is the exam score.

**Main line**:
The engine as it ships today, with every folded-in experiment. Experiments are judged against it.

**Flag**:
A reviewer's claim that a test case is bad data (wrong label, bad audio) rather than an engine failure. Flags are pending until applied to a new test set version.

**Experiment**:
One proposed change to one component, stated up front with its expected effect on that component's score and on the overall score, then measured.
_Avoid_: run, trial

**Fold in**:
Accepting an experiment into the main line, recording its actual effect on the overall score. The alternative is to discard it, keeping the result.

## Training data

**Bucket**:
Where a tlog clip lands after relabelling: `clean-match` (distillation), `deviant` (located, with mismatches; mined for natural errors), `ambiguous` (no single location), or `reject`.

**Text-only label**:
A label that gives a clip's text but not its location, for clips whose candidate locations all have the same canonical phonemes, like the ar-Rahman refrain.

**Speaker cluster**:
A group of tlog clips with similar speaker embeddings. tlog has no speaker IDs, so splits are made per cluster.
