# Degradation continuation for a frozen phone head

Status: needs-info

The implementation is retained, but the experiment is **unmeasured and blocked
before training**. No new model bundle exists, and this branch is not ready for
an acoustic practice evaluation. The experiment card is unchanged.

## Prediction and mechanism

The card predicts one recovered addition with no losses: listener −1, overall
−1. Weak consonants might stay in a blank-dominated CTC alignment basin when a
head encounters difficult audio immediately. Establishing transcript-supported
emission paths on easier views before advancing to degraded views may lead to a
different alignment basin. This is an optimization-order hypothesis, not a new
decoding or warning rule.

`modal/student_degradation_head.py` caches full recordings through the frozen
exported encoder. Each recording has five fixed views: original, 25%, 50%, 75%
and 100% waveform interpolation toward one fixed degraded endpoint. The endpoint
uses the existing phone chain: direct-path-trimmed room response, RMS matching,
fourth-order 120–7000 Hz zero-phase filtering, 10 ms noise-edge fades, active-speech
SNR sampled from 8–22 dB, and peak limiting to 0.89. Parameters depend only on
training recording identity and a fixed seed. No practice audio or labels enter
the cache, sampler, objective or checkpoint selection.

The first course presents all original views, then successively more degraded
views. The matched control shuffles the exact same recording/view occurrences.
Subsequent courses use the same balanced shuffled order in both arms. Every
complete course includes all levels, ordinary speech and each recording's real
pauses. Both heads start from the same shipped readout and use the same CTC plus
full-frame shipped-output KL, Adam updates, recording weights, learning rate and
gradient clipping. No dropout, pause mask, new positives, loss reweighting or
alignment regularizer differs between the arms.

The fixed progression is specified before any target decoding. A single common
checkpoint is selected by mean validation CTC plus preservation loss across
both arms, only at completed courses with identical exposure multisets and update
counts. The default is at most four courses, one pass per level. Optional stage
passes are fixed before a run; they must not be chosen from practice results.
An atomic selection record refers to both checkpoint hashes, so an interrupted
pair save cannot select unmatched arms. An interrupted partial course cannot
become a selected checkpoint.

The data catalog admits only explicitly authorized performed transcripts and
help-train training speakers' none-labelled takes without planted events,
repeats, exclusions or unsupported labels. A speaker tagged non-training anywhere
in either catalog is excluded from both. Paths must remain inside their supplied
training roots, duplicate recordings are rejected, and checkpoint selection uses
a recording-independent speaker partition. Canonical held-sound runs collapse
within each none-labelled ayah, while equal phones across an inter-ayah pause
remain separate occurrences. The cache command requires both source categories;
the availability of the authorized corpus catalog could not be inspected.

Cache provenance includes source model, labels, references, audio, augmentation
banks, transcript targets, speaker partition and cached-file hashes. Cache work
can resume only with identical provenance. Cache and refit share a deadline of
at most 24 hours from cache creation.

`modal/student_frozen_head.py` exposes the head-input activation in the existing
streaming ONNX graph and reconstructs its linear weights, including dynamic-int8
readouts. Export replaces only the final phone readout with a float linear head;
the quantized encoder, input/output names, inventory, padding and 40 ms clock
remain intact. The exporter checks head-logit parity and exact float/integer
recurrent-state parity over four carried-state windows. It preserves auxiliary
files and requires both unchanged insertion-teacher files before creating the
bundle. Primary graph, generated manifest and copied assets receive checksum
verification. Neither tracker, checker nor SDK decoding/verification code changes.

## Validation and limits

Eight local synthetic checks passed: first-course progression, identical exposure
multisets, shared subsequent courses, grouped speaker selection, rejection of
mistake/repeat prompts, cross-catalog held-out exclusion and root containment,
waveform endpoints/clock, and exact equality of the degradation endpoint with
the existing implementation. The final suite also contains five Torch/ONNX
checks for enumerated CTC paths with repeated phones, full-frame pause
preservation, identical per-example gradients, common transactional checkpoint
selection and quantized-head export with carried float/integer states and
unchanged auxiliary files. Those five checks remain unrun because the authorized
worker did not start. Python syntax compilation passed.

The default local Python's NumPy installation has an incompatible CPU
architecture. The tiny waveform tests passed under `arch -x86_64 python3`; no
training, real-audio inference or heavyweight dependency installation ran on the
Mac. Torch, ONNX and ONNX Runtime are absent locally.

Four supervisor requests failed with the same response:
`NotFoundError: No such file or directory`. Each result reports $0 compute and
contains no worker output or returned `log.txt`:

| Request | Work requested | Result |
| --- | --- | --- |
| 1 | Supplied data/model/runtime probe | Failed before worker output |
| 2 | Required six-shard core and SDK build/test harness | Failed before worker output |
| 3 | Retried data/model/runtime probe | Same failure |
| 4 | CPU dependencies and the new numerical/export unit suite | Same failure |

The request/result JSON files are retained under ignored `.autolab/`. No core or
SDK test pass is claimed. The data mount, training catalog, augmentation banks,
source encoder readout and complete insertion-teacher assets could not be
verified. No cached features, trained heads or exported bundles were produced.

The acoustic kill criterion has **not** been tested. Recoveries must be assessed
from complete-session SDK decoding after training-only checkpoint selection,
against both shipped and shuffled-control exports. The task requires at least
three target wins before a practice score. Target wins and aggregate changes are
unknown; a failed worker launch is not evidence of zero recovery or a disproved
premise. No practice scores were used; all three remain available. No killed
outcome or `.autolab/model.json` is declared.

## Remaining execution

Restore the supplied supervisor path and inspect its authorized data/runtime.
Run the required core/SDK tests and the new numerical/export suite there. Supply
the authorized training catalog and existing augmentation banks, then run the
cache, train and export commands documented in the training module. Export both
selected arms and verify the complete bundles, including the unchanged insertion
teacher. Freeze selection before full-session target comparison. Apply the
user's target-win floor, and score practice only if that floor is met. Declare
the continuation bundle through ignored `.autolab/model.json` only after those
files exist and verification succeeds.

No scorer, lab tooling, supplied test data, manifest, evidence or experiment-card
edits were made. No fold was performed.
