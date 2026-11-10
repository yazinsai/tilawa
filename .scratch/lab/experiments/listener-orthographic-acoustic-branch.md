# Orthographic acoustic candidate branch

Status: blocked by the supplied execution service; not implemented or acoustically evaluated. The experiment card is unchanged.

## Prediction and mechanism

The card predicts two recovered additions and one collateral wrong call:
listener −1 and overall −1. The proposed explanation is that vowel, madd and
nasal realization variation fragments the surface-phone spelling used by the
existing seam proposer.

The experiment requires a separate acoustic encoder with non-autoregressive
CTC targets of normalized Arabic letters and spaces, plus auxiliary
surface-phone CTC supervision. Its control must use the identical recordings,
speaker partitions, encoder capacity and training schedule, with surface-phone
targets. Both checkpoints must be selected entirely within training-speaker
partitions. Authorized transcribed Quran recordings and help-train training
speakers' none-labelled recordings are eligible; mistake scripts are not
performed transcripts, and release-held-out speakers must not enter either arm.

Only complete extra-word hypotheses from the existing engine-selected seam
windows may become new candidates. The original-audio teacher insertion gate
and the existing repeat, prefix, local-text and non-Quran exclusions remain
required. Primary phone decoding, tracker input and current warning paths must
be preserved. The auxiliary branch cannot authorize a warning by itself.

## Source audit

`packages/sdk/src/seam-insertion.ts` discovers current candidates with a
read-only free tracker/checker replay, then evaluates a warm original-audio
teacher crop with `seamInsertionMarginal`. Its current candidates require a
complete student-phone spelling before teacher verification. This makes the
card's alternative lexical representation a testable hypothesis; inspecting
the source does not establish that it recovers a target.

`modal/student_train.py` provides the training encoder and augmentation recipe.
`tools/student_manifest.py` records phone references and cluster identifiers in
the student corpus. Building letter targets therefore requires auditing the
supplied corpus and its correspondence to Arabic word text, rather than
assuming its phone-reference strings are independent orthographic transcripts.
The runtime data audit was submitted before training or SDK asset declaration.

## Execution and validation limits

The supplied supervisor command failed before returning container output:

- Run 1: training-image corpus/storage audit, one CPU shard, ten-minute limit.
  Result: `NotFoundError: No such file or directory`.
- Run 2: normal-image working-directory/data/storage probe, one CPU shard,
  five-minute limit. Result: the same `NotFoundError`.
- Run 3: L4 training-image working-directory/data/storage probe, one shard,
  five-minute container limit. Still pending without container output after
  eleven minutes of wall time; it performs no training or writes. Its pending
  status does not establish that the training runtime works.
- Run 4: the supplied `bash tools/autolab_tests.sh` harness, six shards,
  five-minute limit. Result: the same `NotFoundError`, with no test output.

No corpus, speaker exclusion, matched-arm training, checkpoint selection,
ONNX export or acoustic premise test has completed. No target was evaluated,
and no practice score was requested. There is no measured target gain or
overall change; zero evaluations must not be described as zero recoveries.

The card's premise criterion cannot be applied without the trained branch and
matched control on actual engine-selected windows. Likewise, the task's
three-target minimum cannot be tested here. This is an infrastructure block,
not a killed acoustic hypothesis.

No engine, scorer, lab tooling, manifest, test data, evidence or card was edited.
No auxiliary asset was declared, and `.autolab/model.json` was not created:
running the primary model would not evaluate this experiment. There is no
incomplete model bundle to hand back. Implementation and its required unit
tests must resume after the supplied execution service can start a container.
No fold or discard was performed.

The supervisor's result files for completed runs report zero actual GPU spend;
run 3 has a small outstanding reservation. No killed outcome was written,
because neither acoustic stopping condition was measured. Restore the supplied
runtime and its mounts, then retry the corpus audit before implementing or
scoring the experiment.
