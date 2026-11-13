# Diacritic-order supervision

Status: killed on the premise. No engine change, head refit, model export or practice score.

## Preregistered prediction

Listener −1 and overall −1, speculatively removing the false wrong-word warning at
`realuser-v1-sob-049/phone/warning:wrong_word/92:13:1/36.44`, with no predicted losses.

The proposed cause was inconsistent doubled-noon supervision from canonically equivalent
Arabic combining-mark orders. The intervention would compile a base letter and its marks
as a semantic record, correct only demonstrated ordering differences, then compare a
frozen-encoder phone-head refit with an identical-recording, identical-update-count
original-target control. Training and checkpoint selection would remain within authorized,
speaker-grouped training partitions, excluding release-held-out speakers.

## What the target-construction path actually does

`modal/student_train.py` does not compile Arabic orthography. Its dataset converts an
already compiled phone string to inventory IDs. The default target is the teacher's
acoustic greedy output. In the spliced arm, `tools/student.py:spliced_target` substitutes
reference phones between teacher anchors and collapses adjacent repeated phones; soft
KD retains duration evidence.

Those reference phones originate in `tools/student_manifest.py`, which reads the
versioned reference index and calls `reference_phonemes`. Span labels select slices of
the index's existing `p` strings using word offsets. Text-only labels are already phone
strings; they are returned directly, not phonetized. `modal/student_prep.py` joins these
references to teacher outputs without processing Arabic marks. Orthographic shadda is
absent from the 42-phone inventory and would fail direct ID conversion.

The upstream index builder, `tools/build_index.py`, uses `quran-transcript==0.6.4` and
the library's Uthmani text. `modal/tlog_pass.py` likewise obtains its reference text from
`Aya(...).get().uthmani`; the reference does not generate the teacher's acoustic outputs.
There is a known ordering precaution in `tools/build_phrases.py:normalise`: place shadda
before its vowel before phonetizing phrase text. This is evidence of a possible raw
phonetizer limitation, rather than evidence of mixed orders in the student targets.

## Validation

A Unicode-category scan grouped every base letter with its following marks in the
existing reference, without editing the asset:

| Check | Result |
| --- | ---: |
| Reference words | 77,433 |
| Shadda-bearing clusters, all letters | 22,678 |
| Vowel-before-shadda clusters | 0 |
| Doubled-noon clusters | 4,098 |
| Distinct doubled-noon word spellings | 870 |
| Doubled-noon vowel-before-shadda clusters | 0 |
| Words changed by the existing shadda-order correction | 0 |

The noon clusters were `نَّ` (3,676), `نِّ` (269), `نُّ` (143), `نًّۭ` (2), `نًّ` (1),
`نٌّۢ` (1), `نٌّۭ` (5), and `نًّۢ` (1). Every one already places shadda before the vowel
or tanween. All continuous and isolated reference phone strings successfully converted
through the student inventory. The reference asset identifies its compiler as
`quran-transcript 0.6.4`; its SHA-256 is
`f236384ea91e81ad554ba2777bc4073c3ff41e34120343a8a474571c577b7727`.

Thus the demonstrated ordering correction changes no source words for the existing
reference, and there is no raw mark-order processing inside the student target path to
fix. For unchanged reference and teacher strings, both CTC-target arms are identical
for every recording. A head update could still change predictions, but that would not
test the proposed correction to inconsistent supervision.

An authorized supervisor compiler probe requested a pinned installation, original/NFD
equivalence comparisons, inventory checks and the compiler source. Run 1 failed before
producing command output with `NotFoundError: No such file or directory`. Run 3 retried
the same probe and failed with the same error, also without command output.
Core/SDK validation was requested with six shards of
`bash tools/autolab_tests.sh` (run 2); it also failed before executing with the same
supervisor error. No passing test result is claimed.

## Decision and limits

Kill on the absent training-target premise, rather than claiming the raw phonetizer is
order-invariant. No incorrectly ordered doubled-noon training source was demonstrated,
so there is no supported corrected-target arm to refit. No engine changes were made,
and no new behavior tests are needed. The experiment card, inventory, assets, timing,
tracker, checker and warning rules remain unchanged.

This audit does not establish how the external phonetizer handles every canonically
equivalent ordering, does not inspect historical training artifacts unavailable here,
and does not explain the target recording's acoustic confusion. An actual authorized
training source with equivalent spellings that compile to different accepted targets
would be new evidence for the premise. No practice incidents were evaluated, no training
or selection used practice audio, and no overall score delta is reported. `targets won: 0`
records that this killed experiment delivered no changed model or recovered warning.
