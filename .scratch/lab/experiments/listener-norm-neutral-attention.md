# Norm-neutral acoustic attention recheck

Status: incomplete; supervisor execution is unavailable. No acoustic result or
practice score is established. The experiment card is unchanged.

## Prediction and mechanism

Predict one recovered added word and no lost incidents, listener −1 and overall
−1. The speculative premise is that sustained sounds produce high-norm attention
keys which outcompete weak inserted-word consonants. Normalize each content
attention query and key immediately before their dot product, retaining raw
cached keys, all value vectors, positional scores, context, and the 40 ms output
clock. Apply this unchanged-weight graph only as a seam-window recheck; primary
tokens, tracker input and existing candidates must remain intact. Newly spelled
words must pass the existing lexical, local-text, repeat, prefix and non-Quran
exclusions and original-audio teacher insertion verification.

The drafted exporter finds the pinned streaming Zipformer's content-attention
MatMul and its positional Add. It inserts vector normalization on the content
branch only, without modifying model initializers or state outputs. Its fixed
per-head cosine scale is to be calibrated exclusively on authorized training
recordings. For query-centered unit-content logits U, positional logits P and
original full logits O, solve

`a² E[U²] + 2a E[UP] + E[P²] = E[O²]`

for a positive scale a. This includes positional covariance, rather than assuming
that matching content spread also matches complete attention-logit spread.
Calibration must exclude padded keys. An unchanged-weight temperature-only
control scales the original full unmasked logits by one fixed multiplier per
head, chosen to match the norm-neutral arm's training entropy. Padding masks
remain unchanged in both arms. These quantities have not been measured or fitted.

The paired acoustic check must establish previously missing planted consonants
in order, with recovery unique to norm-neutral attention on at least two targets
under the card. The task additionally requires stopping below three recovered
targets; three or more requires a practice score before a gate decision. Neither
floor can currently be evaluated.

## Validation and limits

Read AGENTS.md, the export and streaming paths, seam discovery, certificate
transport and original-audio teacher verification before drafting the transform.
Pinned upstream attention separates content Q/K selection from positional
selection and value retrieval; normalizing immediately before content MatMul
therefore permits the intended intervention without normalizing the key cache.

Behavior tests have been drafted for query/key norm invariance, raw cache
preservation, retained value magnitude, finite zero-norm vectors, unchanged
padding masks, a temperature-one identity control, and spread/entropy calibration.
Python syntax compilation passes. Synthetic ONNX Runtime checks are pending in
a worktree-local isolated environment; they require no real audio or model
inference. Discovery has not been verified on the actual shipped graph.

Supervisor Modal runs 1 (model/data inventory) and 2 (the required six-shard
baseline build/test harness) both returned `NotFoundError: No such file or
directory`, with no command logs or downloaded outputs. Runs 3 (a minimal `pwd`
request) and 4 (training-image inventory and graph tests) returned the identical
error. Restoration of the supplied supervisor environment was requested. This
is an execution failure, not evidence against the acoustic premise.

The existing certificate consumer requires a spelling in primary tokens. Merely
attaching a certificate for a newly spelled recheck word would not make the
change effective. Seam-window discovery and a consumer preserving unchanged
exclusions still need implementation if the paired acoustic test clears its
floor. No SDK wiring or complete model bundle has been produced; evaluating the
main listener would not evaluate this experiment.

No practice-score allowance has been used. No target recoveries, aggregate
changes, clean-session precision or studio behavior have been measured. No
scorer, lab tooling, supplied model manifest, test data, evidence or card edits
were made. No fold or acoustic kill is justified by the results available so far.
