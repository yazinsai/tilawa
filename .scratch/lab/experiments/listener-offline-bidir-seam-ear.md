# Full bidirectional student seam ear

Status: built and retained for the supervisor's fresh listener decode and fold gate. Acoustic premise and target-warning checks passed. The first requested practice score returned noise on the card's existing listener evidence. The experiment card is unchanged.

## Prediction and mechanism

Predict listener −2 and overall −1 wrong calls on lab-v1.5u, with two
of six named additions recovered and about one loss. The task's continuation
rule requires three retained target detections before practice scoring.

The primary student uses streaming chunk masks and chunkwise convolutions.
Its fixed-window ONNX export cannot become a full-lookahead model just by
buffering more audio. This experiment trains a separate copy with
`Zipformer2(causal=False)`, unrestricted attention and ordinary symmetric
convolutions, then exports a variable-length offline ONNX graph without
streaming cache inputs.

Training uses gold-excluded clean-match real recitation, random 2–8 second
sample crops, and starts after natural teacher-blank pauses. Canonical phones
are Viterbi-aligned to the retained teacher posterior rows. Only complete
canonical phone spans inside each crop enter CTC. KD maps each crop frame to
the nearest source teacher frame. The shipped RIR, noise, gain and
SpecAugment calls and settings are unchanged. No practice labels, splice
positives or teacher relabels enter training.

The intended secondary inference span runs from each detector-labelled pause
at an ayah or waqf-marked inner seam to 300 ms after the primary tracker's
step onto the second written word. Its phones are judged only by the folded
soft-cost lexical surplus rule and existing neighbour, prefix, back-run and
non-Quran exclusions. Primary phonemes, posterior rows, tracker events and
support decisions must remain unchanged.

## Validation and limits

Modal run 7 passed the core and SDK builds, all 756 core tests and 30 SDK
tests, including the five new offline decoder/span tests; one optional SDK
integration test was skipped. The supplied harness accepted its documented
Vitest worker RPC timeout after every affected test passed. No test deadline
or enforcement was changed. A subsequent diagnostic invocation in shard 0
failed because ignored analysis files were omitted from the source snapshot;
that failure did not occur in either test suite.

Run 3 passed all four Python tests, including the distant-future perturbation
check. It trained 2,400 updates on 12,762 gold-excluded clean-match recordings
(30.0025 hours), with 53,844 eligible natural gap starts. The original full
student supplied 669 compatible state entries. The symmetric convolutions
needed 64 new entries; the 160 streaming convolution entries were not loaded.
CTC fell from 0.702 at update 20 to 0.171 at update 2,400; KD fell from 1.445
to 0.437. These are training losses, not engine scores.

Run 12 evaluated 512 held-out real crops: 2,871 edits / 13,583 reference phones
= 21.14% PER. This is a crop diagnostic with incomplete edge phones excluded
from the reference, not release PER or a practice wrong-call score.

Run 14 exported the same checkpoint at 69,361,638 bytes, SHA-256
`9f3ca5df47543a825b182a13941c192452790441d3e1e0efcd23d728c5afb56d`.
Native/fp32 ONNX parity passed at 200, 201, 202, 397, 401, 515, 738, 801 and
914 fbank frames, covering different subsampling remainders and a span longer
than the training crops. Maximum fp32 log-probability error was below 1e-5;
random-input int8 argmax agreement was 100%. Native/int8 target-crop agreement
ranged from 97.46% to 100%.

Run 13 discovered the crops from primary evidence and tracker events. Holds
across silence do not break a seam, and a revisit of word 1 before word 2
retains the original seam entry. Tests cover both, skips and relocations. All
six targets have a span that covers their planted interval. Labels select
and report cases only; they never set the engine's crop bounds.

The run-14 int8 census found two planted consonants in order inside four of
six insertions, exceeding the kill threshold of three. Counting the entire
crop would claim five: the second 51:20 target borrows its qaf from outside
the planted interval, so that does not establish recovery.

| Target | Best decode inside the planted interval | Ordered planted consonants |
|---|---|---:|
| p07 37:36:1 | سَلِۦۦم | 3 |
| p48 12:76:1 | َ | 0 |
| p55 26:193:1 | قَلِۦۦلُ | 3 |
| p56 32:21:1 | طِۦۦۦۦن | 2 |
| p57 51:18:1 | قَبڇل | 3 |
| p57 51:20:1 | َبڇ | 1 |

This is evidence that the trained full-context copy can recover acoustic
material at these sites; it does not prove right context is the only cause,
that every recovered string is a whole lexical word, or that clean seams
will stay quiet. No checkpoint or training constant was chosen against these
results.

The runtime declares the int8 ear separately in the SDK registry. The normal
Node loader loads it lazily beside the caller's unchanged primary model. It retains the whole crop and checks its prefixes longest first for a whole
word under the folded spelling costs. CTC blanks inside held vowels do not
split these candidates, and internal substrings are never enumerated. Tokens
travel as `seamSpellings` metadata, survive JSON serialization, and enter only
the folded lexical surplus exclusions and confidence check. The existing
phrase costs also reject a prefix belonging to a complete non-Quran formula
in the same crop.
Primary tokens and posterior arrays are preserved. Live audio supplies its
original tracker events for crop bounds. Like the existing auxiliary teacher
path, secondary warnings are delivered by session `end()`; their evidence
time is the seam's word-two step plus 300 ms.

Run 19 verified JSON-preserved primary evidence, tracker positions and all
non-addition checker events. Its initial utterance-segmentation implementation
produced two new target warnings (p07 and 51:18). The per-frame baseline replay
already reports p55, so that is not an additional win in this diagnostic.
The implementation incorrectly split p56's recovered طين at a long CTC blank
run inside the word. That extra condition was not the folded spelling rule;
it was removed. The complete secondary crop now supplies longest-first whole
prefixes to the same lexical and phrase exclusions. No training, model or
numeric decision constant changed after the census.

Runs 18 and 20 together passed all 756 existing core tests, three added core
tests, and all 35 SDK tests, with one optional SDK skip. Run 18's two new-test
failures were fixture/assertion defects fixed in run 20; its documented
Vitest RPC timeout occurred only after every test in that shard passed.
Run 22 passed the four revised core regression tests and all 35 SDK tests.
Together with the existing-core checks, that is 760 core and 35 SDK tests
passed, with one optional SDK skip. The serialized replay again preserved
primary evidence, all tracker events and every non-addition checker event.
It generated four new target-location/class matches: p07's سليم, p56's طين,
51:18's قبل, and a fragile 51:20 match whose observed phones are ممَ rather
than the planted قبل. Three warnings therefore have the recovered planted
word shape. The fourth does not establish the acoustic premise at that site
and must be judged by the practice gate, together with warnings elsewhere.

The previous attempt assembled `offline-bundle` in this experiment's
persistent scratch, with byte-identical primary `model.onnx` and
`manifest.json` copies of the supplied spliced-clean-int8 export and the
separate declared `offline-student.onnx` ear. Review rejected redirecting
`.autolab/model.json` to that directory. The selector is now restored
byte-for-byte to its inherited `{"path":"bundle"}` value. The declared
offline SDK asset belongs beside the primary model in that existing bundle;
no evaluation configuration change is needed. The baseline teacher was
regenerated with the unchanged
`student_teacher_export.py`: its 2,422,885,632-byte parameter file has the
baseline SHA-256 `d3f24efba836219f0c556ad25e3a1046f9caf36b2ed639cd283f795c78da58b4`.
The regenerated graph has different serialization bytes (10,559,071 bytes,
SHA-256 `6b8d451823dbbd0a42411a36ce10ad26e5013b170999296c40a3715b5ac67220`).
Run 23 checked it against the same native fp32 teacher at 121, 300 and 401
feature frames: maximum log-probability error 5.34e-5 and 100% argmax agreement.
The existing teacher registry entry, weights, decoding and gate code are
unchanged. No supplied manifest was edited.

## Requested practice score

Practice request 1 returned for commit `32b114fc`:

| Wrong calls | Main | This branch | Change |
|---|---:|---:|---:|
| Overall | 24 | 24 | 0 |
| Listener | 23 | 23 | 0 |
| Tracker | 1 | 1 | 0 |
| Checker | 0 | 0 | 0 |

Both runs have 13 missed mistakes and 11 false warnings. The reported verdict
is noise. One of the three permitted practice requests was used; no constants
or engine code changed in response.

This request does not measure the new ear across practice. Inspection of
`serve_practice` in `tools/autolab.py` shows that it invokes `lab run`, which
replays the listener named in the unchanged experiment card,
`exp-listener-teacher-insert-marginal-gate-ad6eb8c0`. It neither copies the
declared bundle nor generates new listener evidence. The supervisor's
`after_tests` phase performs that copy and fresh SDK decode after the executor
finishes, then scores with the resulting listener. No scorer, tooling, card or
stored evidence was changed to bypass this separation.

The acoustic census passes the premise on four targets; the serialized target
replay adds three warnings with the recovered planted word shape, plus one
fragile location match. Retain the implementation and the declared ear in
the inherited `bundle` for the supervisor to measure the new evidence on all practice sessions and apply the
fold gate. The new ear's overall gain, losses, clean-session behaviour and
studio behaviour remain unmeasured by this requested score. No fold verdict
for the freshly decoded ear is claimed.

## Review correction

The previous diff changed lab model selection from `bundle` to
`offline-bundle`. That exceeded the permitted engine scope. This revision
restores `.autolab/model.json` exactly to the inherited file. It leaves
`lab-experiment.json`, scoring code, supplied manifests and evidence untouched.
The trained ear remains a declared SDK auxiliary asset, loaded lazily from
the primary model's directory by the normal Node loader. No acoustic training
or checker decision was changed in response to review.

Modal run 25 rebuilt core and SDK and passed all 760 core and 35 SDK tests,
with the one optional SDK integration skip. The supplied harness accepted
its documented `onTaskUpdate` RPC timeout only after all 134 tests in the
affected core shard passed. No tests, test deadlines or harness rules were
changed for this correction.

Run 24 found no `bundle` directory in this experiment's persistent scratch.
Run 26 materialized that inherited path from the already-built model files.
The primary model SHA-256 is
`714e98addee73b1b8781ad7d99b6d3c745acd9116638baf4e12891f53a15af1b`;
the unchanged manifest SHA-256 is
`785f5eb4724bc0ecd7217b48066aa34ba6dd8ba77f144f6f422c2436a1fc442f`.
Both match `/lab/data/models/spliced-clean-int8` byte-for-byte. The offline
asset hash matches the SDK registry, and the baseline teacher files were
retained. The run then stopped on a diagnostic read of `.autolab/model.json`,
which is omitted from Modal's source snapshot; this occurred after successful
installation and hash checks. Local byte comparisons verify that the selector
and experiment card both match commit `607471c1` exactly.

Run 27 passed the normal `loadNodeModel` smoke check against the inherited
`bundle`: it retained primary model `spliced-clean`, attached both declared
auxiliaries, decoded a one-second input through the offline ear to 23 frames
at 40 ms with the expected inventory, and released the backend successfully.
No further practice-score request was made: this correction changes neither
model weights nor engine decisions, and the earlier requested replay result
remains 24 → 24. Fresh decoded practice and fold-gate results remain the
supervisor's responsibility.

## Two-hour check: variable-length export repair (6 November)

The saved p30 phone/studio errors were reproduced on the existing exported model
with generated long feature sequences, independently of labels. The export traced
Zipformer's prefilled 3,999-row CompactRelPositionalEncoding cache. At longer
runtime lengths it sliced too few rows, causing attention Reshape_3 to fail.

`test_student_offline_export.py` reproduces the actual exporter/module seam: it
failed at 2,001 frames (one positional row rather than 4,001). Clearing the
positional cache immediately before tracing generates its table with a dynamic
Range graph. The regression passes at 401, 1,999, 2,001, 3,201 and 6,401 frames;
all four existing offline tests also pass on Modal. No runtime crop, weights,
training data, labels or decision thresholds changed.

The unchanged 2,400-step checkpoint was re-exported into a separate directory,
with all 11 native/fp32/int8 parity checks passing, including 4,007 and 6,409 fbank
frames (the latter exercises the original requested 6,401-position reshape).
Maximum fp32 error: 0.000010014; all random-input
int8 argmax agreements 1.0. New auxiliary: 68902541 bytes, SHA-256
`853c0eeebdafc57f7f775c5b65c58cd782895960af0d70ea6e6dd6e622df2f2e`. `bundle-shape-fix` preserves every primary/teacher file byte-for-byte
and replaces only this declared offline ear. The old bundle remains available.
The inherited tracked protocol stub is removed; the untracked descriptor names
the actual repaired export. Normal autolab validation/gate decides its result;
no fold or engine gain is claimed here.
