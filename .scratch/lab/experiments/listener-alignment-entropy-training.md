# Transcript-conditioned alignment entropy training

Status: implementation prepared; blocked by supervisor execution infrastructure.
No trained model, acoustic target result, or practice score is available. The
experiment card remains unchanged. This is not a premise kill.

## Prediction

The card predicts two recovered added-word mistakes and one collateral wrong
call: listener −1 and overall −1. Training may concentrate a correct transcript
on brittle temporal placements even when more valid alignments are supported.
The proposed intervention changes that training responsibility distribution.

## Mechanism

`modal/student_alignment_entropy.py` computes exact transcript-conditioned CTC
alignment entropy. It expands the transcript into alternating blank and phone
states, permitting stay, advance and the legal two-state skip. Equal adjacent
transcript phones require an intervening blank. A path can emit only phones
in its supplied transcript, in order; there is no wildcard, extra-phone arc,
output-probability smoothing, or explicit blank penalty.

Forward and backward calculations use the entropy semiring represented by log
path mass and conditional mean log path weight. Entropy is `log Z − E[log W]`.
The manual backward uses forward/backward occupancy and the covariance between
path log weight and each frame emission. It avoids retaining a differentiable
operation graph for every temporal state. Padding is excluded from both path
mass and gradients; impossible transcripts raise instead of silently yielding
a zero objective.

The identity-neutral baseline is the same valid CTC graph evaluated with equal
emission weights: its entropy is the log count of valid temporal paths. This
retains transcript length, repeated-phone topology and recording duration.
The regularizer minimizes `(baseline − conditional entropy) / valid frames`.
The baseline is independent of the model and has no gradient; subtracting it
centres the reported quantity, rather than changing the entropy gradient or
proving that increased entropy improves recognition.

The shipped `student_train.losses` supports an optional
`alignment_entropy_weight`, default zero. Architecture, alphabet, front end,
clock, ordinary decoding, tracker, checker and existing confirmation bars are
unchanged.

`modal/student_entropy_finetune.py` prepares a matched experiment. It admits
clean-match, train-split Quran recordings and help-train training speakers'
none-labelled takes without errors or repeats. Help takes must also pass the
importer's extra-mistake audit (`scored: true`) and contain no unsupported
mistakes. It excludes any Quran speaker
cluster represented in another split or the gold exclusion list, and any help
speaker represented outside train. It does not convert mistake scripts into
performed transcripts. Existing reference run-collapse is retained for held
phone notation. Data identities and speaker partitions are recorded with each
run; deterministic speaker hashing makes an internal fit/validation split.
The supplied data's actual schema, eligibility counts and checkpoint identity
could not be inspected because remote execution never began.

Both arms start from one supplied checkpoint, train on identical batches and
the shipped RIR, noise, gain and SpecAugment chain inherited from that
checkpoint's configuration. They share Python and PyTorch random states for
stochastic encoder choices and dropout. CTC, available cached teacher top-5 KD, and KL preservation against the
frozen initial student apply to both. The entropy arm alone adds the regularizer.
Prepared defaults are weight 0.05, KD weight 0.3, preservation weight 1.0,
AdamW learning rate 1e−5, three epochs, four audio hours and 80 padded seconds
per batch. These choices are fixed before any target inspection. Checkpoints
are selected independently within the training-speaker validation partition by
PER plus 0.1 times preservation KL. The chunk/left-context regime is 16/128.
No release-held-out or practice recording enters this program.

Export uses the existing streaming ONNX wrapper and dynamic int8 quantization.
It checks alphabet, blank, frame and feature geometry, streaming window and
state contract against the supplied base bundle, and retains that bundle's
auxiliary files so the folded insertion path remains installed. This export
code has not run; no `.autolab/model.json` is declared, and no incomplete model
bundle is presented as a result.

## Validation and limits

Numerical unit tests enumerate all short CTC paths and compare entropy, path
partition, neutral baseline and gradients with an independent exhaustive sum.
They also check finite-difference gradients, repeated labels, an empty
transcript, variable recording lengths, packed targets, nonzero blank IDs,
invalid transcripts and single-path entropy. All eight tests passed locally
on Python 3.13 and PyTorch 2.15.0 (8.805 seconds after imports). These results do not
establish compatibility with the supervisor's Python 3.12 / PyTorch 2.8.0 GPU
image. Python syntax compilation and `git diff --check` also passed.

The tests also cover exclusion of unaudited help takes and held-out
speakers, and the exact loss/gradient difference between continuation and
entropy training using an independent exhaustive CTC path sum. Command:

```sh
PYTHONPATH=modal:tools OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .autolab/entropy-venv/bin/python -m unittest modal/test_student_alignment_entropy.py -v
```

Four supervisor requests failed before their commands executed, each with
`NotFoundError: No such file or directory` and no command log:

1. Training-image data/checkpoint and storage inventory, 10-minute timeout.
2. Training-image numerical unit tests, 10-minute timeout.
3. The required six-shard `bash tools/autolab_tests.sh`, 20-minute timeout.
4. Minimal ordinary-image `pwd`, 5-minute timeout.

Each returned USD 0, GPU budget spent 0 and GPU budget left 65. The failure
affects even a command with no dependence on experiment code. No alternative
execution service, out-of-worktree data, or Modal credentials were used.
A fifth ordinary-image `pwd` retry was requested at 04:24:31 UTC on 2026-11-11.
It produced no supervisor result after more than seven minutes. Its local
waiting process was stopped; the supervisor request remains pending and had
no GPU reservation.

Training, matched continuation, export parity, full-session SDK replay and the
core/SDK test suites therefore remain outstanding. No target audio was decoded
or used to choose a weight or checkpoint. Practice scores used: 0 of 3. Target
recoveries are unknown, not zero. There is no independent listening judgment.

The card's acoustic premise requires at least two targets to gain a previously
missing planted consonant in order against the matched continuation. The task's
stop rules also require a practice score before disposition if three or more
targets win. Neither condition has been measured. A trained bundle must be
completed and verified before model selection or scoring; evaluating the default
main listener would not evaluate this training change. No fold or discard is
authorized or performed.

## Resume once supervisor execution works

First inspect the actual s1/help schema, eligibility counts and available
checkpoint lineage with the authorized supervisor. Establish that the selected
parent checkpoint exports the installed primary, rather than choosing a parent
by target performance. Run the numerical tests in the training image and the
required six-shard core/SDK suites. Then run the matched recipe, using that
verified parent:

```sh
python modal/student_entropy_finetune.py \
  --checkpoint /student/prior-runs/<verified-parent>/ckpt.pt \
  --out /scratch/listener-alignment-entropy-training/matched \
  --export --base-bundle /lab/data/models/spliced-clean-int8
```

Defaults and selection policy above are fixed. Verify both complete exported
bundles and their SDK loading, clock and streaming parity before paired
full-session target decoding. Record recovered previously missing planted
consonants against the matched continuation, then follow the card and task's
target stop rules. Declare only the verified primary entropy bundle in ignored
`.autolab/model.json`; no asset has been generated by this implementation turn.
No outcome JSON labels this infrastructure block as a premise kill.
