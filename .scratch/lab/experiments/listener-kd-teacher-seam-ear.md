# KD teacher as a tracker-bounded seam ear

Status: implemented, with the rejected lab model-selection override removed.
The acoustic premise passes on seven targets and the packaged model catches
six. The teacher arm has no full-practice score or fold verdict.

## Preregistered prediction

Practice lab-v1.5, listener −2 wrong calls, overall −1. Base commit
`7928bbdf`. The experiment card is unchanged.

## Mechanism

Load the existing KD teacher (`obadx/muaalem-model-v3_2`) once in fp32.
At an ayah seam or waqf-marked inner seam, use the shipped tracker's step
onto the previous word and step onto the second word after the seam.
Decode a warm crop starting four seconds before the former and ending
300 ms after the latter. Replace only seam-local checker tokens and rows,
feeding the folded lexical-spell and added-word support paths unchanged.
Keep the main string, tracker input and other evidence paths unchanged.
No training or threshold calibration is performed.

The SDK loads an optional `teacher.onnx` beside the unchanged main model.
It records tracker steps and rejects spans crossing a lock drop. The
teacher uses its own saved SeamlessM4T feature recipe: Kaldi log-mel up
to 8 kHz, unbiased per-bin crop normalization and stride-two stacking.
Teacher evidence travels on the first recorded token as an auxiliary
payload; the main phonemes and posterior rows remain intact. A second
instance of the unchanged checker receives bounded replacements. Only
its added-word event histories at the matching seam are selected; all
other warning classes come from the original checker. The original
tracker receives no auxiliary evidence.

In this experimental implementation audio-mode addition events are
delivered at `Session.end()`, with their original evidence timestamps.
Other runtime event classes retain their original delivery. Replay uses
the exact original tracker-update batches, including the final batch.
Where seam crops overlap, the later crop owns both tokens and rows;
speech is never duplicated across the two crops. This permits
the same recorded evidence to be replayed by the unchanged lab code,
but does not establish streaming addition-warning latency.

The preregistered acoustic kill is fewer than four of the eight targets
with at least two planted consonants in order inside the replacement span.
If that premise holds, the built mechanism still needs at least three
target wins before a practice score is warranted.

## Validation method

Supervisor Modal run 3 replayed the shipped `spliced-clean-int8` phone
evidence through the free tracker. Labels were used to select and report
target cases, never to define crop boundaries. All eight targets have
both required tracker steps. At p55 the previous-word step falls 440 ms
after the planted word starts; its warm crop includes those earlier frames,
but the replacement span does not.

The teacher measurement uses float32 parameters, features and logits,
with autocast and TF32 disabled. It checks the teacher's inventory against
`ref-v2`, greedily collapses non-blank runs and keeps top-five posterior
rows. Both models use the fixed 40 ms output frame clock; final receptive
field loss is not stretched over the audio. Consonant longest common
subsequence is reported both across the tracker span and inside the
labelled planted interval. The latter distinguishes actual inserted speech
from coincident consonants in neighbouring canonical words.

The supervisor rejected GPU run 4 because GPUs are restricted to bet
cards. Run 5 performs the same fp32 inference on Modal CPU.

Supervisor Modal run 2 established the original test baseline. Run 16
built core and SDK and ran all six shards with the implementation:
717 core tests and 20 SDK tests passed; one existing help E2E test was
skipped. One core shard hit Vitest's `onTaskUpdate` RPC deadline after
every test passed, which the supplied test harness explicitly accepts.
New tests cover bounded replacement, unchanged tracker and other warning
classes, folded neighbour rejection, feature normalization and padding,
greedy collapse, and rejection of spans across a lock drop. Run 18 built
core and passed all three seam-ear tests after adding the overlap
regression. Run 20 then built both packages and passed all 20 SDK tests
(one help E2E test skipped) for the exact-batch replay change.

## Acoustic result

Run 5 completed successfully on CPU. Seven of eight tracker spans pass
the two-consonant criterion. The fp32 teacher is not uniformly fragmenting
these insertions as its students do.

| Target | Planted word | Teacher tokens in the labelled plant | Consonants in order, replacement span / labelled plant |
| --- | --- | --- | --- |
| p07, 37:36:1, 144.115 s | سليم | سَرِۦۦم | 2 / 2 |
| p48, 12:76:1, 39.975 s | كذب | جَدِر | 1 / 0 |
| p49, 26:123:1, 152.245 s | كريم | َرِۦۦم | 3 / 2 |
| p50, 37:99:1, 146.465 s | وقفوهم | وَقِفُۥۥهُم | 5 / 5 |
| p55, 26:193:1, 269.44 s | قليلون | قَلِۦۦلُۥۥۥۥن | 2 / 4 |
| p56, 32:21:1, 106.76 s | طين | طِۦۦۦۦن | 2 / 2 |
| p57, 51:18:1, 95.26 s | قبل | قَبڇل | 3 / 3 |
| p57, 51:20:1, 112.26 s | قبل | قَبل | 3 / 3 |

p49's ك is emitted just before the labelled onset and lies inside the
replacement span. p55's earlier ق and first ل lie in the warm crop but
outside the replacement span. The table keeps those boundary effects
visible instead of changing either span.

## Result and limits

Runs 12 and 13 exported the real fp32 teacher with external ONNX weights
and checked the SDK's front end plus ONNX inference against run 5 on all
17 measured crops. Frame counts and argmax sequences were identical on
every crop (zero differing frames). The primary ONNX and manifest are
copies of `spliced-clean-int8`, unchanged. The auxiliary graph and fp32
weights total about 2.43 GB. They are an analysis payload, not a viable
device-budget result.

Runs 10 and 15 replayed the seven target sessions using teacher tokens
and rows in the measured tracker spans. Run 15 used the production
seam-only checker wrapper and verified byte-identical non-addition
event histories. Six of the eight targets became caught calls:
p07 37:36:1, p49 26:123:1, p55 26:193:1, p56 32:21:1, and both p57
targets. p48 and p50 remained missed. These are target diagnostics,
not a full-practice score. Other word-1 additions in those sessions were
also examined; the already-caught p07 37:18:1 was lost.

The initial replay excerpt showed only two confirmations because it
stopped six seconds after each plant. Full event histories recovered
four more: p49's confirmation arrives at 174.92 s after the first
suspected event at 155.44 s. The existing scorer associates that history
with its initial suspicion, so the short excerpt was insufficient.

The acoustic criterion is weaker than lexical spelling or confirmation:
p07 passes it while substituting ر for ل; p48 does not pass it at all.
Teacher agreement is correlated model evidence, not an independent review
of the audio or labels. Hearing a word does not show that the tracker-span
replacement reaches a folded emission path.

The teacher has a different architecture and front end and is evaluated
on warm crops; the shipped comparison is its recorded stream. The result
does not isolate a KD-loss defect from student capacity, quantization or
inference context. It does establish that the shared teacher can retain
consonants missing from some student decodes.

Run 19 loaded the actual ONNX bundle once per shard and fed the seven
target sessions' phone audio through the SDK, with every eligible seam
enabled. It generated 15–48 seams per session. Both actual runtime event
histories and the recorded evidence replay caught the same six targets
as run 15; p48 and p50 stayed missed. Total measured SDK compute ranged
from 67 to 248 seconds per session on Modal CPU. Run 17 had reached
inference but its temporary report driver called the scorer API without
the reference index; run 19 corrected that diagnostic error.

Run 22 built both packages and repeated the audio runs after the
exact-batch replay fix, paired with the same current SDK and primary
backend with the teacher disabled. All seven sessions had exactly equal
primary phoneme values (identity, confidence and frame bounds), posterior
IDs and probabilities, tracker events and non-addition warning events.
Both runtime and recorded replay still caught six of eight targets.
The direct comparison against older `spliced-clean-int8` cached evidence
in run 19 had returned false; run 21 found small confidence differences
and a few token changes in that older cache. The paired current-model
control establishes the experiment's isolation without attributing those
cache differences to the teacher. Temporary probe drivers were removed.

Practice score 1 was requested after the six-target result. It returned
overall +0 and listener +0: 33 wrong calls (22 misses, 11 false warnings),
charged 32 listener, 1 tracker, 0 checker. This verifies the path with no
auxiliary evidence and is not a score of the teacher arm. The supplied
`serve_practice` implementation replays the card's existing listener
evidence through `lab run`; it does not decode `.autolab/model.json`.
The requested score therefore must be identified by the evidence it actually
uses. No narrowing or threshold change was made against this unchanged-evidence
result. The teacher's full-practice losses, clean-session warnings, studio
result and reader robustness remain unmeasured.

## Diff-review correction

The previous attempt added `.autolab/model.json` to point the scoring stage at
the exported bundle. Diff review rejected that file as a prohibited change
to lab control. This revision deletes it without adding another scoring
override. It also removes the exporter’s copies of the practice model and
manifest: the exporter now produces only the generic teacher graph and
weights, independent of `LAB_DATA` and any practice model name.

The SDK still loads `teacher.onnx` once when it is beside the caller's primary
model. It continues to load the primary model specified by that caller's
manifest. No adjacent teacher means the ordinary shipped path. Supplying the
auxiliary graph is an SDK deployment prerequisite; this branch does not
redirect the supervisor to the diagnostic bundle. The remaining allowed
scoring path replays the existing shipped evidence and cannot measure this
arm without auxiliary evidence already present. Repeating that score would
not resolve the missing teacher-arm measurement. One practice-score request
has been used in total; no additional score is claimed here.

Supervisor Modal run 23 rebuilt both packages and completed all six test
shards with exit code 0: 718 core tests and 23 SDK tests passed, with one
existing help E2E test skipped. Three new loader tests verify that an adjacent
teacher is loaded once across repeated crops without replacing the caller's
primary backend, that an absent teacher preserves the ordinary path, and that
an inaccessible installed teacher reports an error. Local Python syntax and
`git diff --check` also passed. No acoustic inference or threshold changed in
this correction; the saved six-target result still applies to the explicitly
supplied diagnostic bundle, subject to the unmeasured practice limits above.
