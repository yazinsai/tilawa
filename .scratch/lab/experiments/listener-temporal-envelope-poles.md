# Temporal-envelope poles

Status: infrastructure-blocked draft. The scorer, trainer, SDK integration and
unit tests are implemented, but no corpus fit, complete model bundle or target
measurement exists. The experiment card is unchanged.

## Prediction and mechanism

The card predicts two false warnings removed and one caught substitution lost:
listener −1 and overall −1. Its proposed cause is that spectral summaries hide
brief closures and nasal transitions within sustained speech. Frequency-domain
linear prediction (FDLP) instead fits an all-pole representation of each
frequency band's temporal energy envelope.

`modal/student_envelope.py` implements a CPU-only trainer. Every 40 ms it takes
a centered 128 ms waveform interval, transforms it with a DCT, and fits
regularized prediction along frequency in six fixed subbands. Band powers and
stable reflection coefficients describe the envelope. The matched control
uses ordinary 25 ms frame-energy measurements at uniformly spaced positions
within the identical interval. Both arms include waveform energy, have the
same number of feature dimensions and Gaussian parameters, and use identical
phone/silence timing graphs.

Only explicitly training-split, none-labelled help-train recordings can enter
fitting. A speaker with any nontraining row is excluded entirely, as are
explicitly excluded speakers, audited-out recordings and unscored takes.
Events, repeats, unsupported annotations or a missing single canonical ayah
prevent a take from being treated as clean. Missing speaker/split metadata
fails closed. These checks occur before opening ineligible audio. Training
does not read practice audio or labels.

Phone-conditioned diagonal Gaussian models are fitted with transcript
forward-backward occupancies and four EM updates. Proportional placement is
used only for initialization; later updates marginalize boundaries. Recording
contributions are balanced by speaker. The silence states are optional, while
every written phone, including repetitions, is obligatory. Sequence scores
divide out the timing graph's partition to avoid a pronunciation-length
advantage without acoustic evidence.

The preregistered training search is order {2, 4}, LP ridge {0.001, 0.01} and
Gaussian shrinkage {8, 32}. Three deterministic speaker folds compare generic
one-phone pronunciation contrasts, with each speaker weighted equally and
both arms contributing to the selection objective. The chosen configuration
is refitted on eligible training speakers before any target inspection.
The export records exclusions, speaker folds, recording hashes, validation
results, fit history, package versions and the emission asset's checksum.
This search has been implemented but has not run on help-train.

## Engine integration

The SDK implements the same numerical features and complete-pronunciation
forward sums. Its optional Node bundle loader verifies the emission file's
declared size, SHA-256, schema, feature dimensions, inventory, sample rate and
frame clock. A declared missing or corrupt file is an error; an ordinary
primary-only bundle has no active envelope scorer.

The checker exposes an inspection hook only for an otherwise supported,
confirmed string-path wrong word. It supplies the existing support comparison's
complete written/rival strings and exact frame bounds, including its existing
neighbor context. The SDK emits a certificate only when the temporal arm
strictly prefers written and the control strictly prefers rival. Ties,
nonfinite scores, unavailable frames and untrained phones abstain. Certificate
matching requires the same word, pronunciations and frame bounds.

The checker retains its baseline internal confirmation and ownership ledger,
and publishes an uncertain verdict for a vetoed wrong word. This preserves
subsequent candidate generation and insertion ownership. Parallel-phrase graph
confirmations are exempt. Primary tokens, confidence values, posterior rows and
tracker input are preserved. Audio-backed sessions can score live candidates;
recorded evidence also carries exact-window certificates through the existing
token metadata route for later replay.

`modal/student_envelope_compare.ts` is a generic diagnostic of that SDK route.
It discovers comparisons through the actual tracker/checker, loads the verified
bundle, reports both arms, and checks primary evidence, tracker events and
non-wrong-word mistakes for equality. It contains no registered target identities
or target-dependent inference rules. It has type-checked but has not run on
practice recordings.

## Validation

Local validation used ignored dependencies under `.autolab/`, since the
worktree arrived without node_modules:

- All 762 core assertions passed across the original full run and targeted
  timeout-only reruns, including six new certificate tests. The first run
  exceeded 5 s deadlines for ten assertions and 15 s deadlines for two existing
  startup assertions, and reported two Vitest RPC timeouts. The affected
  assertions passed with longer local deadlines; repository tests, assertions
  and fixtures were unchanged.
- The SDK's 37 assertions passed across its full run and timeout-only reruns;
  one optional help integration was skipped. Coverage includes latent-boundary
  sums checked against exhaustive paths, obligatory repeated phones, abstention,
  strict two-arm reversals, streaming feature coordinates and asset integrity.
- Six Python tests passed, including soft-occupancy fitting and whole-speaker,
  audited-recording and non-none exclusions before audio access.
- Python/SDK synthetic-waveform parity passed for both orders and 16 sequence
  comparisons: maximum feature difference 7.060663165248116e−11 and maximum
  score difference 1.6086687537608668e−9.
- Strict TypeScript source checking and `git diff --check` passed. The local
  checker used source workspace aliases and declarations for unavailable
  web/native runtime packages; this is not a completed production build.

Supervisor requests 1–5 all failed before command execution with
`NotFoundError: No such file or directory`, reporting zero compute cost.
Requests 1–2 attempted metadata inspection, 3 and 5 requested the supplied
six-shard test harness, and 4 requested the CPU training fit. There is no
container training log or fitted artifact from those requests.

Consequently the actual staged corpus schema and eligibility counts remain
unaudited, no speaker-grouped model selection has occurred, and none of the
four registered targets has a measured comparison or warning reversal.
Targets won is **unknown**, not zero. No practice score was requested.
Synthetic tests establish numerical and transport behavior, not the acoustic
premise, aggregate gain, clean-session behavior or studio safety.

The user requires at least three target warnings removed to continue and a
practice score before deciding when three or more are won. Neither the card's
two-reversal premise nor that stricter continuation condition can be assessed
without fitting and exact-window measurements. Infrastructure failure is not
an acoustic kill result, so the draft is retained without claiming success or
writing a killed outcome.

## Bundle and resumption

There is no `.autolab/model.json` and no complete bundle under the persistent
experiment scratch directory. Evaluating the main listener without trained
envelope assets would not evaluate this change. The ignored outcome file marks
the draft blocked and omits fictitious measurement counts.

After the supervisor's worktree access is repaired:

1. Audit only help-train metadata, confirming that training speakers' none
   recordings have the expected schema and that held-out speakers are absent.
2. Run the fixed CPU training search and save its report before inspecting any
   practice comparison:

   ```sh
   python3 /Users/rock/ai/projects/tilawa/tools/autolab.py modal-run --timeout 60 -- 'python3 modal/student_envelope.py --help-dir "$LAB_DATA/help-train" --out /scratch/listener-temporal-envelope-poles/fit && cp /scratch/listener-temporal-envelope-poles/fit/envelope-training-report.json "$LAB_OUT/"'
   ```

3. Assemble a persistent `bundle/` with unmodified copies of the selected primary
   model and manifest, every auxiliary teacher file needed to preserve current
   insertion behavior, and the trained `envelope-asset.json` and
   `envelope-emissions.json`. Verify all declared files and primary/teacher
   hashes through the normal SDK loader before declaring the bundle path.
4. Run the supplied six-shard tests and the generic comparison diagnostic on
   all four target sessions' actual engine-requested windows. Verify primary
   and protected decision equality, and count warnings removed rather than
   individual comparisons reversed.
5. Apply the user's continuation rule. Fewer than three target warnings won:
   revert engine changes, record the measured premise kill and commit. Three
   or more: declare only the verified complete bundle in the ignored model
   selector and request a practice score before deciding; the fold gate owns
   the aggregate tradeoff.

The current implementation keeps session audio and frame features and has not
been profiled for device memory or latency. Its independent acoustic models
could erase a genuine substitution. Those remain limitations to measure, not
reasons to invent a result or alter the training search against practice.
