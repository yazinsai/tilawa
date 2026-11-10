# Orthographic acoustic candidate branch

Status: blocked by the supplied execution service on the implementation retry.
The experiment is not implemented or acoustically evaluated. The experiment
card is unchanged.

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
  five-minute container limit. Its completed result now reports the same
  `NotFoundError`, with no container output and $0.9929 charged.
- Run 4: the supplied `bash tools/autolab_tests.sh` harness, six shards,
  five-minute limit. Result: the same `NotFoundError`, with no test output.

## Implementation retry (11 November 2026, Bahrain)

The previous review requested engine code or an explanation of why the change
cannot be implemented. This retry read `AGENTS.md` first and submitted fresh
requests through the supplied supervisor, without modifying that supervisor or
trying another execution route:

| Run | Request | Completed result |
| --- | --- | --- |
| 5 | One training-image CPU shard, ten-minute limit: print runtime paths and available storage, then inspect the first supplied training parquet's schema and metadata | `NotFoundError: No such file or directory`; no container output; $0 charged |
| 6 | Six ordinary-image CPU shards, ten-minute limit: `bash tools/autolab_tests.sh` | `NotFoundError: No such file or directory`; no build or test output; $0 charged |
| 7 | One L4 training-image shard, five-minute limit: print working directory, CUDA availability and whether the supplied training/scratch mounts exist | Pending after more than nine minutes of wall time, with no container output |

Run 5 never returned its first working-directory print. The failure therefore
does not establish which remote file, application or volume is missing, and
does not establish a disk-space problem. Run 6 also failed before returning
test output; no test has passed or failed on its assertions in this retry.
The corresponding supervisor result files are local, untracked
`.autolab/modal-result-5.json` and `.autolab/modal-result-6.json`.
Run 7's pending status does not establish that a GPU container started; it
performs no training or writes. Its eventual result can be retrieved with the
supplied `python3 /Users/rock/ai/projects/tilawa/tools/autolab.py modal-result 7`
command. The stated five-minute limit has not bounded the startup/result wait.

The mechanism cannot be completed here because the authorized training audio,
speaker metadata and prior checkpoints are supplied only inside those remote
containers. This worktree has no training checkpoint and no `node_modules`;
its only tracked ONNX model is the tiny unit-test fixture. Until a container
starts, the required speaker exclusions and letter transcripts cannot be
audited, the matched encoders cannot be trained or selected on training-only
partitions, and a complete exported asset bundle cannot be verified. The
user explicitly requires the complete bundle before handoff. Declaring a
missing branch or scoring the baseline would leave that requirement unmet.

Source inspection also identifies a necessary integration detail for the
eventual implementation. The current checker exposes `seamInsertion` only
after exact phone spelling and student neighbour certification. The new
letter branch must receive the existing seam bounds before that spelling
filter; otherwise the remaining candidate-free targets would never be
consulted. Its complete word hypotheses must then pass the unchanged
original-audio teacher gate and exclusions through the existing certificate
route. This observation describes the required wiring; it is not a measured
acoustic recovery or an implemented change.

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

No killed outcome was written, because neither acoustic stopping condition was
measured. The completed run 3 charge updates the previously pending reservation;
runs 5 and 6 report $179.01 of the $180 GPU budget remaining. Restore the supplied
execution service and its mounts, then complete the corpus audit, matched-arm
training, bundle verification and actual-window premise check before scoring.
