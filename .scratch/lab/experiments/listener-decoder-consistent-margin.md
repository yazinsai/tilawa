# Decoder-consistent max-path phone-head margin

Status: training implementation and synthetic validation complete; real training,
export and target validation blocked by supervisor execution failures. No model
is declared. The experiment card is unchanged.

## Prediction and premise

Predict listener −1 and overall −1, with one added-word recovery and no losses.
This forecast is within the noise band and below the fold threshold. The premise
is that ordinary CTC can increase a transcript's aggregate probability while its
best complete frame path still loses to the unrestricted greedy frame path.

The implementation's two-frame example illustrates the distinction: each frame
has blank probability 0.6 and phone probability 0.4. The phone transcript has
aggregate probability 0.64, versus 0.36 for silence, but its best frame path has
probability only 0.24. Greedy therefore produces silence. This example proves
the objectives differ; it does not establish this defect on any practice target.

## Built mechanism

`modal/student_margin.py` computes exact best correct and best incorrect complete
CTC frame paths. Prefix states alternate blank and the most recently emitted
transcript phone. An incorrect emission enters an absorbing reject state;
incomplete prefixes also reject at the final frame. Equal-label continuations
and blank-separated repeated articulations follow ordinary CTC semantics.

Loss augmentation is the bounded binary sequence loss: zero for the correct
transcript, one for any incorrect transcript. The structured hinge is
`max(0, 1 + score(best incorrect) − score(best correct))`. It compares maxima,
never summed competitor probabilities. Search covers the complete vocabulary
and all valid timings without beams, lexicon constraints or competitor sampling.
The path search uses detached scores; gathering the winning complete paths from
the live head output gives an exact max-path subgradient. Softmax's shared
normalization cancels in the score difference.

The automaton has at most `2U + 2` states, O(TUV) work and O(TU) backpointer
storage. Fixed bounds are 1,500 valid frames and 512 target articulations.
Oversized or impossible recordings fail instead of truncating transcripts,
changing timing or approximating the search. Empty transcripts use the real
all-blank path and compete against every nonempty decode.

`modal/student_margin_train.py` exposes the shipped ONNX encoder's native
features and caches three fixed views per recording: original audio and two
views using the shipped waveform augmentation and SpecAugment recipe. Only
the existing affine phone head is fitted. Both arms start from its dequantized
native weights, retaining the quantized encoder. Removing the head's dynamic
activation rounding is common to both arms and cannot be credited to the margin.

Both arms receive identical recordings, cached views, seeded recording order,
initialization, SGD updates and clipping. Both compute CTC, path-margin and
shipped-posterior KL preservation losses; the ordinary control gives the margin
coefficient zero. Each recording has one update per epoch, averaging its views.
CTC and margin divide by transcript length, or valid frame count for silence.
The fixed schedule is 12 epochs, learning rate 0.02, clipping bound 5 and margin
coefficient 1. There is no practice-based constant or strength search.

Whole training speakers are hash-partitioned into fit and selection groups.
Selection averages CTC + unit margin + preservation equally over selection
speakers, choosing the pair's minimum mean objective at one shared epoch.
Epoch zero is eligible. Both selected heads have the same update count.
Speech PER and silence emissions are reported separately. A cumulative 24-hour
preparation/refit/export deadline refuses incomplete or unmatched results.

## Data and export contracts

Preparation admits only help-train `none` takes from explicitly training
speakers, with scored status and empty events, non-errors and unsupported lists.
Mistake scripts and repeats are not substituted for performed transcripts.
Authorized cached recordings are enrolled through JSONL containing speaker,
training split, authorization, verified performed phone transcript, audio path
and audio SHA-256. Verified real pauses have empty transcripts and `kind=silence`;
synthetic all-zero audio is rejected. Duration-expanded reference notation is
converted using the SDK's held-phone convention, retaining genuine gemination.

Speakers from every nontraining split in either corpus, plus an explicit release
exclusion list, are removed before admission. Each partition requires both
speech sources and real pauses. Duplicate audio hashes and cache checksums
prevent duplicate weighting and changed training views. These enrollment
contracts have synthetic checks, but their compatibility with the actual remote
corpus is unverified because data inspection never executed. No claim is made
that an enrolled input list or feature cache currently exists.

`modal/student_head_export.py` exposes native features and replaces only the
existing projection. It preserves the alphabet, frontend, 40 ms output clock,
stream window, recurrent state interface and terminal log-softmax. Existing
decoder, tracker, checker and insertion verification code are untouched.
The unchanged insertion teacher and its external weights must be present,
copied and checksum-verified. Existing models and manifests are never rewritten;
new exports get their own manifests and model SHA-256.

Preparation requires exact parity of original outputs and recurrent states
after feature exposure. Both exported heads must reproduce their fitted
probabilities on every selection view within 2e-4 before a final
`training-report.json` is written. A model declaration is permitted only after
both complete bundles and their auxiliary assets pass verification.

The runner exposes `prepare --model --help --transcribed --exclusions
--augmentation --out` and `fit --prepared --out`, for the supplied supervisor's
training image. Preparation and export support were adapted from
`exp/listener-unconditional-count-training`; its count objective is not included.

## Validation and limits

All 24 local synthetic tests passed with no skips. The suite exhaustively
enumerates tiny frame paths, including empty targets, repeated phones, alternate
blank indices, two-class inventories, incorrect incomplete prefixes and extra
emissions after completion. It compares exact max scores and loss-augmented
scores, checks hinge gradients against exhaustive finite differences away from
ties, and verifies padding and common-logit invariance. A tied initial gradient
fixture was corrected: central finite differences average subgradients at a tie,
whereas the exact Viterbi implementation validly selects one maximizing path.

Tests also check fixed encoder/preservation gradients, ordinary-control gradient
parity, matched examples and shared selected updates, budget expiry, speaker
exclusion, transcript/audio enrollment, float and int8 head extraction, unchanged
ONNX state outputs, and complete teacher/external-weight bundle copying.
Local validation uses an ignored workspace virtual environment with Torch 2.8.0,
ONNX 1.19.0, ONNX Runtime 1.22.1, NumPy 2.5.4 and Python 3.13. No recording
inference, feature-cache preparation or training ran on the Mac. Compilation
and whitespace checks pass.

Four supplied supervisor requests failed before command execution, each with
`NotFoundError: No such file or directory` and USD 0.00:

1. Run 1: supplied data roots and dependency discovery.
2. Run 2: training-image discovery of caches, help labels and the shipped bundle.
3. Run 3: required six-shard core/SDK build and test harness.
4. Run 4: the new Python numerical and export suite on the training image.

The required core/SDK suites therefore remain unverified. There is no real
trained margin head, matched ordinary head, verified replacement bundle or
full-stream target decode. No practice score was requested; all three remain
available. Target wins, losses and listener/overall changes are unknown.

This is an infrastructure blocker, not a measured zero gain or an acoustic kill.
No `.autolab/model.json` or killed outcome is written. Scoring with the main
model would not evaluate the new objective.

## Remaining experiment work

Restore the supplied supervisor, inspect and enroll authorized training data,
prepare the common native-feature cache, fit the matched pair and verify both
complete bundles. Compare all five registered targets using correctly ordered
full-stream tokens and planted consonants relative to the ordinary refit;
isolated reference-selected crops cannot establish recovery.

Apply the user's final stop rule: fewer than three target wins or a disproven
premise requires engine rollback, a result-only commit and a killed outcome.
At least three wins requires the supplied practice score before deciding or
narrowing. Collateral losses belong to the fold gate; a new false warning on a
fully clean session is a hard stop. Neither decision is supported yet. No fold
was performed, and no scorer, lab tooling, test data or supplied evidence changed.
