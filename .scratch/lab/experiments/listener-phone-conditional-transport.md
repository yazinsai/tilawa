# Phone-conditioned acoustic feature transport

Status: blocked before training and target measurement. This is a draft implementation, not a scored or shippable model change. No complete model bundle or `.autolab/model.json` was produced. Running the primary listener without the side assets would not evaluate this experiment.

## Prediction and stopping rule

The card predicts two listener wrong-word false warnings removed and one caught substitution lost: listener −1, overall −1. Its three named targets are `lab-v1.5-p11/phone/warning:wrong_word/26:106:1/104.32`, `lab-v1.5-p36/phone/warning:wrong_word/55:48:1/282.64`, and `realuser-v1-sob-049/phone/warning:wrong_word/92:13:1/36.44`.

The later implementation instruction requires at least three target wins before proceeding to a practice score; fewer than three would kill the built change. That overrides the card's two-reversal criterion. No target comparison or practice score has run, so neither the forecast nor the stopping criterion has been measured. Target wins are unknown, not zero.

## Implemented mechanism

`modal/student_transport_export.py` adds a pre-CTC encoder output to a separate copy of the selected primary ONNX graph. Before writing feature metadata, its export path checks every original output and recurrent cache bit-for-bit over eight sequential random streaming windows with ONNX Runtime optimizations disabled. A fixed random projection reduces the encoder output to 32 features. The primary graph, its manifest and its optional insertion-teacher assets are copied unchanged into the output bundle.

The training extractor takes explicit help-train phone labels and a separate authorized studio label file. Both use training-speaker splits, canonical none-labelled transcripts and the native streaming feature clock. Laptop takes cannot substitute for studio references. Non-training speakers are excluded across the entire input collection. Scripted substitutions supply only comparisons already requested by the unadapted checker at the labelled word; scripts never supply a performed pronunciation or force its boundaries.

`modal/student_transport_phone.py` computes exact transcript-constrained forward/backward CTC occupancies. Full posterior state mass contributes to each phone and its nearest left/right vowel-quality group. Sampling balances phones, vowel contexts and recordings rather than using hard engine word visits. A studio-trained 32→48→vocabulary readout is frozen. Two identically initialized 32→16→32 residual adapters receive the same source/studio feature samples and update counts. The treatment uses balanced Sinkhorn couplings restricted to identical phone/vowel groups; the control pools those exact samples without that restriction. Distance, entropy scale, identity penalty and optimizer settings are shared. Each deployed adapter consumes only acoustic features.

Three speaker-grouped training folds select regularization and a shared sequence-scoring temperature. Preservation-only speakers also receive folds. Selection considers held-speaker soft phone likelihood and reversals on engine-requested training substitution comparisons. The final heads, their selection report, extractor and checksummed side-asset descriptor are written together. The ordinary SDK loader validates complete files, hashes, inventory, dimensions and the exact primary SHA, and rejects missing or corrupt declared assets.

The SDK scorer compares both complete existing pronunciations on exactly the same existing support window. It marginalizes CTC boundaries under the same normalized timing model, conditioned on completing the entire pronunciation, so extra timing paths do not create a length reward. A certificate requires finite scores, treatment preference for the written sequence and control preference for the already-accused rival. The checker matches the exact word, both sequences and frame bounds; graph-confirmed substitutions are exempt. It keeps the internal confirmation/ownership ledger so vetoes cannot change insertion generation or merges.

The session integration advances an independent feature stream before passing each primary audio chunk to the live checker. An existing string-path confirmation can therefore be vetoed before its public confirmed event is emitted; its internal ownership remains intact. The extractor follows the primary fbank windows, recurrent caches, resampling and final padding clock. Session completion attaches the live certificates and any existing stored-window rechecks to recorded evidence. Primary decoding and tracker inputs remain unchanged. The initial completion-only scaffold was replaced during review because withdrawing a warning at `end()` would still allow an earlier live false confirmation.

## Validation actually completed

- Python syntax compilation passed for the trainer, exporter and training tests.
- Four small training tests passed locally using the installed x86 Python environment: ambiguous soft-state mass and vowel-context accounting; phone/recording balance independent of recording length; full-pronunciation timing normalization; and domain-balanced speaker folds including preservation-only speakers.
- `git diff --check` passed. The experiment card, benchmark code, supplied labels, manifests and evidence were not changed. Two unrelated end-of-Quran guards missing from the reused checker scaffold were restored to baseline.
- Added core tests exercise discovery without verdict drift, finite treatment/control reversals, exact-window and complete-pronunciation matching, unchanged internal confirmation events, graph exemption and neighbouring insertion preservation. Added SDK tests exercise candidate-order symmetry, impossible-sequence abstention, incremental/batch primary streaming clock/cache parity, declared-asset rejection, stored certificates and vetoes before live warning publication. These suites have not executed.
- Added training tests for Sinkhorn marginals and a synthetic cross-phone swap that unconditional coupling can match but conditional coupling cannot. These Torch-dependent tests have not executed.

## Supervisor failure and outstanding validation

All authorized supervisor requests failed before command output with `NotFoundError: No such file or directory`; each reported zero compute billed:

1. Request 1: read-only listing of the provided data/model directories and experiment scratch space. It returned no listing.
2. Request 2: the required six-shard `bash tools/autolab_tests.sh`. It returned no test results.
3. Request 3: a minimal `pwd` probe. It also returned no command output.

The failure of the minimal probe distinguishes this from a failed training command or failing engine test. An asynchronous clarification requested a supervisor setup fix and the location/provenance of authorized training-only studio recordings. No alternative compute access or benchmark training source was attempted.

To resume, repair supervisor startup and identify the authorized studio corpus; then extract the two domains, run the complete test suites and Torch mechanism checks, train/select the two matched arms, export and verify all declared assets, and diagnose the three target warnings using their actual complete checker-requested comparisons. Only a measured three-target recovery authorizes proceeding to the practice score under the latest instruction. No practice scores have been used.

The acoustic premise remains untested. Missing studio coverage, uncertain transcript occupancies, phone-balanced readout calibration and small-data adaptation limit this draft. The side stream adds a second encoder pass and retains projected features across the session; latency, memory and actual streaming parity still need measurement. No release-safety, studio-safety, target-recovery or preservation result is claimed.
