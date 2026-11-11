# listener-coded-phone-discrimination

Status: killed on mechanism, before acoustic training or practice scoring.

## Prediction

The card predicts listener −1 and overall −1: recover two additions while
risking one collateral wrong call. The proposed primary listener replaces its
categorical phone projection with balanced error-correcting binary codes,
including blank, normalizes their Bernoulli likelihoods into phone probabilities,
and retains CTC, the shipped augmentation and ordinary-reading preservation.
The requested comparison is an equal-parameter categorical head on identical
authorized training data. The experiment card was not edited.

## Mechanism audit

The specified normalization removes the Bernoulli discrimination mechanism.
Let C[c,k] be the fixed binary code for phone c, and let z[k] be a bit logit.
Its unnormalized log likelihood is

```text
s[c] = sum_k C[c,k] log sigmoid(z[k])
             + (1-C[c,k]) log sigmoid(-z[k])
     = sum_k C[c,k] z[k] - sum_k softplus(z[k]).
```

The last term is identical for every phone, including blank. Normalizing over
phones therefore gives exactly `log_softmax(C z)`. With the proposed final
linear bit projection `z = W h + b`, this is a categorical linear head with
effective weights `C W` and bias `C b`. Code balance, Hamming separation and
the number of bits do not prevent this cancellation.

An exactly matched categorical control can retain the same trainable W and b
and apply the fixed C projection directly before softmax. It has the same
parameter count and the same function. Its normalized-output Jacobian is also
identical: `d log p[c] / d z[k] = C[c,k] - sum_j p[j] C[j,k]`.
Consequently, CTC and any preservation/KD objective operating on those phone
probabilities have identical gradients in the two parameterizations. Sharing
initial parameters, minibatches, augmentation, dropout and optimizer makes the
training problem identical; this does not require acoustic or passage labels
to establish.

This is a finding about the specified Bernoulli-versus-categorical mechanism,
not a claim that changing a head's parameterization can never affect training.
A different code factorization or optimizer geometry could affect a comparison
with an unfactored categorical layer. That would not demonstrate independently
supervised binary error correction. Adding a separate bitwise loss or a
nonlinear bit network would change the experiment's mechanism; neither was
substituted for the card's CTC recipe.

## Validation

Inspected `modal/student_train.py::build_model`, `losses`, and
`modal/student_export.py::wrap`: the current head is dropout followed by a
linear projection and log-softmax, and both training and streaming export
consume the normalized phone distribution. The existing SDK reads those
same output columns; no SDK adaptation can restore a term canceled inside
that distribution.

A small standard-library numerical diagnostic used a label-independent,
fixed-seed 43-by-64 binary code. Each phone had exactly 32 one bits; every
bit had 21 or 22 one labels; minimum pairwise Hamming distance was 20.
Both candidate and factorized categorical control would have 16,448 trainable
head parameters at the shipped 256-channel output. The diagnostic evaluated
61 synthetic frames at each of five logit scales, using stable log-sigmoid
calculations, a blank-inclusive CTC forward sum with repeated target labels,
and the full normalized-output Jacobian at one frame per scale.

| Logit scale | Maximum probability difference | CTC loss difference | Maximum Jacobian difference | Argmax changes |
| --- | --- | --- | --- | --- |
| 0.1 | 2.50e−16 | 2.84e−14 | 5.55e−16 | 0 |
| 1 | 1.44e−15 | 5.68e−14 | 4.44e−16 | 0 |
| 10 | 6.99e−15 | 0 | 1.28e−16 | 0 |
| 100 | 3.81e−19 | 0 | 2.13e−25 | 0 |
| 1000 | 3.42e−44 | 0 | 0 | 0 |

These checks support the algebra; 305 synthetic frames are not practice
targets or an acoustic benchmark. The diagnostic and its JSON results are
retained, untracked, in `.autolab/coded_head_identity.py` and
`.autolab/coded-head-identity.json`. Run it with
`python3 .autolab/coded_head_identity.py` from this worktree.

The supplied supervisor service failed before executing any command:

| Request | Requested validation | Result |
| --- | --- | --- |
| 1 | Read-only runtime/storage probe, 1 CPU shard | `NotFoundError: No such file or directory` |
| 2 | PyTorch CTC/preservation gradient and ONNX head equivalence, training image, 1 CPU shard | Same infrastructure error |
| 3 | Required `bash tools/autolab_tests.sh`, 6 shards | Same infrastructure error |

The PyTorch/ONNX checks and core/SDK suite therefore did **not** run. The
standard-library check above ran locally and took about a second; it was
not acoustic processing or training. No GPU was requested or charged;
the supervisor reported $60 of $60 remaining. A preliminary local NumPy
diagnostic could not import the installed architecture-mismatched extension;
it was replaced with the standard-library check rather than altering the
environment.

## Verdict and limits

Stop under the user's instruction to kill when the mechanism is wrong. The
prescribed normalized Bernoulli CTC head is exactly an equal-parameter
categorical control, so it does not instantiate the proposed independent
binary-discrimination comparison. No primary model was trained or exported;
no production head, SDK, tracker, checker, insertion-verification, test,
scorer, lab-tooling, supplied manifest or evidence change remains.

Targets acoustically evaluated: **0 of 6**. Empirically demonstrated target
recoveries: **0**. Practice scores requested: **0 of 3**. Overall/listener,
studio and clean-session deltas are **unmeasured**, rather than measured zero.
The stop is the mathematical mechanism finding, not collateral losses,
the three-target recovery floor, or the Modal infrastructure error. No
`.autolab/model.json` is declared, and no fold was performed.

The killed outcome is recorded in untracked `.autolab/outcome.json`.
