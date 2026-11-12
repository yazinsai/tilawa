# Global loss denominators in student training

Status: killed on the premise. The shipped training recipe has no independently normalized gradient workers or accumulation steps. No engine change or replacement listener is retained.

## Preregistered prediction

Listener −1 wrong call and overall −1 on lab-v1.5u. The registered cluster was listener/added_word/missed, with five targets: p07 at 144.12 s, p48 at 39.98 s, p56 at 106.76 s, and p57 at 95.26 and 112.26 s. The forecast was uncertain and below the standalone fold threshold.

## Proposed mechanism

Represent each acoustic loss as a numerator and denominator; aggregate denominators across gradient workers and accumulation steps before scaling gradients. Preserve the unpartitioned effective-batch objective and existing recording weights. Only if the current recipe actually differs from that objective, compare a bounded frozen-encoder phone-head refit with an identical-data, identical-update-count current-aggregation control, selecting within speaker-grouped training partitions and excluding release-held-out speakers.

## Premise audit

The documented `spliced-clean` reproduction command in `docs/benchmarks/student-v2.md:38` uses `modal/student_train.py`, with the full model, spliced CTC targets and KD weight 2. This is the only checked-in student trainer. The export path rebuilds this model from the saved training configuration (`modal/student_train.py:497`; `modal/student_export.py`).

The trainer requests one H100 (`modal/student_train.py:384`) and constructs one model and optimizer. Its `workers=12` setting is passed to PyTorch's DataLoader (`modal/student_train.py:453`); those processes load and collate complete batches, not losses or gradients. The duration-bucketed batch plan is consumed by one training loop. That loop zeroes gradients, backpropagates one batch loss and steps the optimizer (`modal/student_train.py:460–472`). There is no gradient accumulation, distributed model wrapper or gradient collective in the student training paths. Parallel preprocessing and decoding do not aggregate training gradients.

For a complete batch, let `F = sum(out_lens)`, `C_i` be recording i's unnormalized CTC loss, and `K_it` its valid-frame teacher cross-entropy. The actual objective is:

```
L = ctc_weight * sum_i(C_i) / max(F, 1)
  + kd_weight * sum_i,t(valid_it * K_it) / max(F, 1)
```

CTC uses `reduction="sum"` and divides once by total valid output frames (`modal/student_train.py:306–335`). KD renormalizes each teacher top-k distribution, then divides the masked frame sum by `valid.sum()` (`tools/student.py:77–85`). The mask is `t < out_lens[i]`, so this is the same denominator for valid positive-length model outputs. Padding is excluded. There is no mean of independently normalized recording or worker losses. Length-dependent contributions already belong to this complete-batch objective; changing those weights would be a different experiment.

## Validation

The source audit traced both loss reductions, the complete-batch collation and the optimizer update loop. Searches of `modal/student_*.py` and `tools/student*.py` found no distributed gradient wrapper, collective or accumulation path. The trainer, loss helper and experiment card were checked byte-for-byte against HEAD and remain unchanged.

A temporary read-only numerical audit was prepared to extract the actual `losses` and `collate` functions without importing the Modal app. Its synthetic recordings had 3, 7, 17 and 29 valid frames, unequal teacher lengths and blank-heavy long examples. It would compare full-batch loss and head gradients against frame-weighted reconstructions over four disjoint partition plans, vary KD weight/delay, and compare DataLoader worker counts 0 and 2. Independently averaging partition means was included only as a counterfactual; that operation does not occur in the trainer. The script's Python syntax passed a local AST check, but the numerical audit did not execute, so no gradient-equality measurements are claimed.

Supervisor requests 1 and 3 attempted the unchanged six-shard `bash tools/autolab_tests.sh` core/SDK build and test suite; requests 2 and 4 attempted the numerical audit, with the retry passing the script directly in the command. All four returned `NotFoundError: No such file or directory`, with no command logs or audit artifacts. The supervisor reported no compute charge. The missing resource was not identified, so these are unavailable checks, not test failures or passes. Local PyTorch is unavailable; no dependencies were installed and no local training was attempted.

No unit tests were added because no engine behavior changed. The existing core/SDK suite could not be verified through the supplied executor. The kill rests on the source-level absence of gradient partitioning, not on an unexecuted numerical comparison.

## Decision and limits

The premise kill criterion fires before refitting: there is no partition-dependent normalization discrepancy in the shipped recipe. Introducing distributed training or accumulation merely to replace a hypothetical local-mean reduction would create a new recipe rather than repair the current one.

No training recordings, help-train takes or practice audio were consumed. No checkpoint was selected or exported, no model bundle is declared, and no practice score was requested. The outcome's `targets won: 0` denotes no delivered target wins; it is not an acoustic evaluation of a refitted model. No listener or overall score delta is claimed.

This audit establishes the checked-in recipe and loss semantics, not historical execution logs or an undocumented trainer fork. It does not establish why the five additions lose consonants, nor whether a future multi-GPU implementation would normalize correctly. Stochastic augmentation, encoder batch effects and ordinary SGD differences between successive optimizer batches are outside this loss-reduction hypothesis.
