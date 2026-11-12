# Distillation target-domain contract

Status: killed on the conversion premise. No engine change retained, head refit, model export or practice score.

## Preregistered prediction

On lab-v1.5u practice, listener −1 wrong call and overall −1, with one recovered added word and zero predicted losses. The proposed cause was treating already-normalized teacher probabilities as logits, flattening weak consonant supervision.

## Proposed mechanism and stopping decision

Distinguish teacher logits, log probabilities and probabilities, converting each exactly once to the distribution consumed by the existing KD loss. Only if that fixes an actual transformation would this experiment freeze the encoder and compare corrected-target and current-conversion phone-head refits with identical data and update counts, selecting checkpoints within training-speaker partitions.

The transformation is absent in the current source. A probability-domain correction would reproduce the current teacher distribution, which fires the card's first kill criterion. The three-target minimum for continuing cannot override a false conversion premise. No replacement training objective or unrelated acoustic change was attempted.

## Source audit

- `modal/tlog_pass.py`, `TeacherPass.infer`: raw model outputs named `logits` become `torch.softmax(lg.float(), dim=-1)` once. `run_shard` selects the phoneme distribution's top five entries and stores probabilities as float16 bytes, with uint8 class IDs.
- `modal/student_prep.py`, `prep_shard`: `top5_ids` and `top5_p` are copied from the teacher pass without conversion.
- `modal/student_train.py`, `load_clips` and `ClipDataset.__getitem__`: probability bytes are decoded as float16 and cast to float32. `collate` pads them; `losses` gathers teacher rows using the existing frame mapping. None applies softmax, log-softmax or temperature scaling to the teacher probabilities.
- `tools/student.py`, `soft_target_loss`: the teacher weights are `q = top_p.float() / top_p.float().sum(-1, keepdim=True).clamp_min(1e-6)`. Cross-entropy is `-(q * student_log_probs.gather(-1, top_ids)).sum(-1)`, averaged over valid frames. Thus every retained nonzero probability ratio is preserved, apart from the existing float16 storage rounding. Conditioning on retained top-k mass is the documented objective, not probabilities treated as logits.
- The student head's `log_softmax` operates on student logits. The separate `modal/student_teacher_export.py` wrapper applies `log_softmax` to raw teacher logits for SDK evidence. `student_train.transcribe` exponentiates student log probabilities before writing the same top-five probability format. These are distinct, correctly ordered conversions.

## Validation

The existing regression test `tools/test_student.py::test_soft_target_loss_is_cross_entropy_to_renormalised_top_k_over_valid_frames` explicitly checks that teacher weights `[0.6, 0.2]` become `[0.75, 0.25]` and that padding contributes no loss. Against student probabilities `[0.25, 0.5]` in that class order, its expected cross-entropy is 1.21301.

A lightweight standard-library arithmetic calculation reproduced 1.2130075659799042. Applying the suspected extra softmax instead gives weights `[0.598687660112452, 0.401312339887548]` and cross-entropy 1.1081258442029223. Thus the existing test's expectation distinguishes the proposed defect. This calculation illustrates the formula; it does not execute the tensor loss helper.

Two validation requests used the authorized supervisor:

- Modal request 1, CPU training image: a temporary synthetic audit of 512 valid frames, float16 probability-byte round-trip, loss and consumed-target gradients, padding, and an erroneous-softmax negative control, followed by `python3 -m pytest -q tools/test_student.py`.
- Modal request 2, six shards: `bash tools/autolab_tests.sh`, the prescribed core/SDK builds and tests.

Both requests returned `NotFoundError: No such file or directory`, with zero reported cost and no output directories or command logs. Neither the tensor audit nor the Python/core/SDK tests could be verified. No runtime pass, numerical target delta or test failure is inferred from that infrastructure error. No alternate Modal route was attempted.

No engine behavior was changed, so no permanent unit test change was needed. The temporary audit file under `tools/` was removed; there are no engine changes to revert.

## Limits

The audit checks source and synthetic target arithmetic, not the provenance or contents of historical training caches. It does not assess the information discarded by top-k storage, teacher accuracy, acoustic augmentation alignment or other training hypotheses. No training data, held-out speaker audio, practice incident audio or evidence was read for this audit.

The five target spans were not replayed because the proposed target-domain correction is a no-op. Zero target recoveries were demonstrated; this is not a measured practice delta. No practice score was requested, and no model bundle or `.autolab/model.json` was declared. The final tracked change is this write-up; the experiment card and engine remain unchanged.
