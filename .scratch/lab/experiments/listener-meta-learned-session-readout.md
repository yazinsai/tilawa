# Episodically learned session readout

Status: blocked on the supervisor's Modal runner. Implementation is preserved;
no trained bundle or measured acoustic result is available.

## Prediction and mechanism

Predict two false warnings removed and one caught substitution lost: listener
−1, overall −1. The card is unchanged. The task's explicit continuation rule
requires three target wins before a practice score, overriding the card's
two-reversal premise check.

A separate copy of the shipped encoder graph exposes frozen pre-CTC features.
A deterministic projection retains 32 coordinates, normalized using only each
training fold. A linear pronunciation readout has those coordinates plus a bias;
its initialization and positive diagonal adaptation regularizer are learned.
The primary model and its streaming state remain separate.

Only help-train training speakers' none-labelled recordings supply canonical
supervision. Recording exclusions, unscorable takes, labelled mistakes and
repeats cannot supply support. Speakers in any nontraining partition are
excluded. Three deterministic speaker folds select the starting regularizer
from 0.01, 0.1 and 1 using held-speaker query loss within the training set.

Exact transcript-conditioned CTC forward/backward produces soft phone
occupancies, including blank and repeated articulations. An episode uses a
chronological prefix of accepted words as support and complete later accepted
words as queries. Support must already have been available and finish at least
300 ms before a query word. Each support word contributes equal total mass,
then divides that mass over its frames.

Adaptation solves a ridge system whose target is the log soft occupancy. Its
prior mean is the learned initialization and its precision is the learned
positive diagonal regularizer. Query cross-entropy differentiates through the
solve to both parameters. A fixed quarter-weight unadapted query loss anchors
the initialization. Training uses 60 epochs with a fixed optimizer recipe;
neither passage identities nor query features or expected query phones enter
the support update. The SDK implements the same normal equations with a
Cholesky solve.

The checker records completed zero-cost word visits in locked or declared
context, requiring existing 0.8 token confidence and no accusation. Their
availability timestamps come from the finalized evidence clock. At a pending
string-path wrong word, the support bank excludes speech unavailable at the
comparison's onset, speech within 300 ms of the comparison, and speech
overlapping known accusation intervals with the same collar.
Feature availability additionally includes the entire encoder input window and
the final 25 ms fbank frame: earlier frame indices alone cannot authorize
support whose acoustic features could contain the disputed audio. Training
episodes use the same availability adjustment.

Both readouts compare the complete written and already-accused pronunciations
on identical existing warning frames. All phone boundaries are marginalized
with a timing prior normalized on completion. A certificate requires the
unadapted readout to prefer the rival strictly and the adapted readout to prefer
the written pronunciation strictly. Missing support, incomplete paths and
nonfinite scores abstain. Certificates bind the exact word, frame interval,
complete sequences and causal support ranges.

The checker retains baseline internal confirmations and ownership, changing
only the public wrong-word verdict to uncertain. Graph confirmations are
exempt. The SDK preserves primary tokens, tracker events and insertion
decisions. Its current audio-session integration computes the auxiliary recheck
at session completion and publishes matching downgrades; timely streaming
vetoes are not established by this implementation.

## Validation and limits

The export utility is designed to check eight consecutive streaming windows,
including every primary cache output, for bitwise parity between the original
graph and its feature-exposing copy with optimizations disabled. It persists
the original primary bundle, frozen feature graph, fitted heads, SHA-256 asset
descriptor and training report, then verifies ordinary SDK loading. A declared
missing, corrupt or mismatched side asset fails loading.

Added tests cover gradients through the support solve, query-label isolation,
chronological word episodes, speaker partitions, ridge numerical behavior,
equal word weighting, repeated-phone occupancies, causal support eligibility,
exact-window certificates, finite strict reversals, feature/primary streaming
geometry, graph exemptions, insertion preservation, asset rejection and
audio-session downgrade publication.

Local Python syntax checks and `git diff --check` pass. These checks do not
establish TypeScript build or unit-test success. Modal requests 1 (data audit),
2 (six-shard tests) and 3 (training/export) failed before execution with
`NotFoundError: No such file or directory`. Their result files contain no shard
logs and report zero compute spend. Requests 4 (test-harness retry) and 5
(corrected training/export retry) failed with the identical preexecution error.
Their commands did not run either. Later static review additionally tightened
feature availability to exclude encoder input containing disputed audio; the
latest source and tests still require execution when the runner is restored.

No practice score has been used. The four target windows remain unmeasured.
There is no verified bundle and no `.autolab/model.json` declaration.
Primary-only decoding would not evaluate this experiment. Infrastructure
failure supplies no premise evidence and is not reported as zero target wins
or an acoustic kill.

Once supervisor computation works, run the latest six-shard tests and the
training/export command, verify all bundle files and normal SDK loading, then
inspect the four targets with the generic SDK comparison CLI using their
actual audio and stored evidence. Only then declare the verified `bundle` in
untracked `.autolab/model.json`. Fewer than three actual target wins requires
reverting the engine changes and recording a kill; three or more requires a
practice score before deciding. No fold has been performed.

The training/export command to reissue through `autolab.py modal-run` is:

```bash
python3 -m pip install --quiet numpy onnx onnxruntime &&
python3 -m pip install --quiet torch --index-url https://download.pytorch.org/whl/cpu &&
pnpm --filter @tilawa/core --filter @tilawa/sdk build &&
python3 -m unittest discover -s modal -p test_student_session_readout.py &&
python3 modal/student_session_readout_export.py \
  --model "$LAB_DATA/models/spliced-clean-int8" \
  --labels "$LAB_DATA/help-train/labels.jsonl" \
  --out /scratch/listener-meta-learned-session-readout/bundle
```

Use `--timeout 60`; the output persists in the experiment scratch directory.
The required test command is `bash tools/autolab_tests.sh` with six shards.
The generic diagnostic CLI is
`pnpm --filter @tilawa/sdk exec tsx src/bench/session-readout-cli.ts`, accepting
`--model`, `--audio`, `--evidence` and `--out`. Its comparison records contain
support provenance and both readout preferences, while its before/after replay
checks tracker and insertion parity. None of these blocked steps is certified
as passing, and this branch is not evaluation-ready without its complete bundle.

Limited none-labelled speaker coverage, short recordings without usable
support/query pairs, erroneous accepted labels, lossy feature projection and
poor learned transfer remain risks. Training-speaker validation cannot
guarantee preservation of genuine substitutions. No scorer, lab tooling,
supplied manifest, evidence or test data was changed.
