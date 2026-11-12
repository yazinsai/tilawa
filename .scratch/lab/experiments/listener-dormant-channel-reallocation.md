# Dormant upper-channel reallocation

Status: training implementation added; acoustic validation and measurement remain
blocked by the supplied supervisor. No replacement model is declared and no
target gain is claimed. The experiment card is unchanged.

## Prediction and mechanism

Experiment `listener-dormant-channel-reallocation`, listener, explore. Predict
one recovered addition, listener −1 and overall −1, with substantial zero-gain
risk. Redundant upper acoustic capacity might leave weak consonants poorly
represented even though the student has sufficient nominal width.

The implementation recycles internal feed-forward expansion channels in the
last two Zipformer stacks. These channels do not change residual width, cache
geometry, alphabet, posterior clock, tracker, checker or insertion verification.

## Implementation

`modal/student_reallocation.py` implements padding-free, per-recording
post-SwooshL activation RMS and outgoing-vector contribution RMS. A candidate
must remain below one tenth of its layer's median activation and one millionth
of its layer's outgoing RMS on **every** observed recording/view. These fixed,
conservative admission limits are not fitted to practice.

SwooshL is `log(1 + exp(x − 4)) − 0.08*x − 0.035`; it has no zero region.
Negative preactivations are not dormant by definition. Zeroing an outgoing
column removes its previous contribution, even if that contribution is small.
Consequently, admission statistics alone never establish preservation.

A training-only CTC Viterbi alignment locates poorly supported transcript
consonants. Incoming weights are initialized as normalized templates of their
upper-block input vectors, at the existing activation shoulder, using the
median incoming-weight norm. Outgoing columns start at zero. No extra channel,
compensation branch or inference hook is added.

`modal/student_reallocation_train.py` provides the executable comparison:

- Enroll clean-match s1 Quran recordings and audited help-train none takes.
  Reject mistakes, repeats, unsupported/excluded takes and duplicate audio.
  Discover non-training speaker clusters and gold exclusions before enrollment;
  exclude whole Quran clusters containing excluded clips. Missing speaker
  provenance stops the run. Partition enrolled speakers deterministically
  into fitting and selection groups; both must contain both corpora.
- Verify that the source checkpoint reproduces the installed listener's
  exported initializer tensors and streaming state geometry. Require the
  insertion teacher and its external tensors and record source asset hashes.
- Cache original and augmented views once, using the existing noise, RIR,
  gain and SpecAugment recipe. Use fitting speakers for the channel census
  and weak-consonant template selection.
- Construct ordinary continuation, dormant recycling and equally sized
  random-channel recycling arms. Both recycling arms use the same templates
  and zero-outgoing operation. Random channels are selected without looking
  at practice. Their initial acoustic drift is recorded as a control result;
  random live-channel recycling is not claimed to preserve the function.
- Before dormant refitting, require full streaming probability drift at most
  `1e-6` and identical frame argmax on both views of every enrolled recording,
  including selection speakers and real pauses. This is empirical preservation
  on those inputs, not an algebraic guarantee on arbitrary future audio.
- Freeze the acoustic stem and lower stacks; refit the last two stacks,
  output downsampling and phone head with ordinary transcript CTC, the
  existing top-five KD where available, and whole-output KL preservation
  across speech and real pauses. All three arms receive identical cached
  recordings, objectives, dropout draws, clipping, learning rate and 60 updates.
- Select one shared checkpoint count by mean selection-speaker objective
  across the three arms. Practice data never enters selection. Upper-only
  snapshots and shared frozen stems limit CPU memory.
- Export all three complete streaming int8 bundles with the existing
  MatMul-only quantization recipe. Check FP32 streaming parity, finite int8
  output, inventory/state geometry, file hashes and unchanged auxiliary assets.

Preparation saves each recording, channel census and initial-parity result.
Matched refits save only after all three arms complete an update; completed
exports can be resumed. A separate cumulative budget counts preparation,
refitting and export across invocations within one day. No incomplete export
is automatically installed for scoring.

## Validation and execution

Local validation passed **15 tests**, with **4 Torch tests skipped** because
Torch is unavailable locally. Tests cover negative SwooshL activations, outgoing
contributions, intermittent activity, speaker exclusions, CTC alignment against
enumerated paths, repeated-articulation blanks, probability/clock preservation,
audio containment and invocation boundaries. The pending Torch behaviors are
zero-contribution initialization, rejection of non-preserving initialization,
outgoing learning, and frozen-stem/refit snapshot behavior.

The native Python installation has incompatible NumPy extensions. The small
NumPy tests passed with `arch -x86_64 python3 -m unittest discover -s modal -p
test_student_reallocation.py -v`. Compilation and the runner's help/import
smoke check also passed. These are source/unit checks, not acoustic validation.

Earlier supervisor runs 1–4 failed with `NotFoundError: No such file or
directory`. This implementation attempt requested default-image inventory
(run 5), training-image inventory (6), the required six-shard core/SDK harness
(7), Torch behavior tests (8), and the executable training workflow (9).
All five also failed with the same error before command output or shard logs;
the result files report zero charged compute. A final updated Torch test
request is run 10. The error does not identify the missing supervisor resource.

The training request used the documented prior-checkpoint location:

```sh
python3 /Users/rock/ai/projects/tilawa/tools/autolab.py modal-run \
  --image train --timeout 60 -- \
  'python3 modal/student_reallocation_train.py \
    --checkpoint /student/prior-runs/spliced-clean/ckpt.pt \
    --source-model "$LAB_DATA/models/spliced-clean-int8" \
    --out "/scratch/$AUTOLAB_EXP/reallocation" --budget-seconds 3000'
```

No remote checkpoint, authorized corpus, channel statistics, preservation
result, arm training, export or SDK test result was obtained. Runtime integration
with the actual pinned model and data schemas remains unverified. There is no
evidence that dormant capacity exists or that this intervention recovers a
consonant. The strict census/preservation requirements may admit no channel.

Practice-score requests used: **0 of 3**. Full-session comparison of the five
registered targets against both exported controls has not run. Target wins,
collateral losses and overall score change are unknown, not measured zero.

## Disposition

The previous documentation-only attempt is now replaced by engine training
code and behavior tests. The requested trained, scored experiment is still
incomplete because the only authorized heavy-compute route cannot execute.

Once execution is available, run the full comparison and verify all bundles.
Then replay the five complete sessions through the ordinary SDK with the
dormant, continuation and random models. Apply the user's final stopping rule:
stop and revert engine changes if the premise fails or fewer than three
targets win; otherwise request the official practice score before deciding.

No killed outcome is justified by unavailable execution. No
`.autolab/outcome.json` or `.autolab/model.json` is written, so main-model
decoding cannot be mistaken for evaluation of recycling. The source
implementation remains on this branch for review and completion.
No scorer, lab tooling, supplied manifest, test data or evidence was changed,
and no fold was performed.
