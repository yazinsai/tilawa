# Pair-specific nonlinear phone tournaments

Status: blocked, unmeasured. The implementation and training pipeline are prepared, but the permitted supervisor could not start any Modal job. There is **no fitted bundle**, no `.autolab/model.json`, and no acoustic target or practice result. Decoding with the main listener would not evaluate this experiment. This is an infrastructure blocker, not a premise kill or a deployable experiment result.

## Prediction and decision rule

Card: listener-pairwise-phone-tournaments, listener / explore. Prediction: two false wrong-word warnings removed with one caught substitution lost, listener −1 and overall −1. The card and all supplied labels, manifests, evidence and lab tooling remain unchanged.

The hypothesis is that the multiclass objective neglects rare consonant contrasts. Pair-specific balanced objectives on the same frozen encoder representation might distinguish a written consonant from the acoustic model's rival. That hypothesis remains untested on real audio.

The card's diagnostic asks for two written-pronunciation reversals against the matched control. The user's continuation rule takes precedence: measure actual removal of all three target warnings before taking a practice score; fewer than three warrants a measured kill and rollback. An infrastructure error is neither a zero-target result nor evidence that the acoustic premise is wrong. Zero practice scores have been used.

## Training mechanism

`modal/student_pairwise_export.py` prepares a separate copy of the shipped streaming graph with its pre-CTC encoder representation exposed. The primary graph and manifest are copied without modification. The exporter checks all original outputs and updated caches bit-for-bit over eight successive random windows with ONNX optimizations disabled. It also carries source subdirectories and auxiliary insertion assets into the bundle, and checks normal SDK loading and insertion-teacher presence before reporting success. None of those real-graph checks could run here.

The SDK extraction CLI processes only help-train training speakers. Every speaker represented by a non-training label is excluded from both supervision and validation. None-labelled recordings containing scripted events or repeats are rejected. Their complete canonical pronunciations condition exact CTC forward/backward soft occupancies; no hard alignment or mechanically corrected mistake transcript supplies phone supervision. Frozen native encoder features receive a fixed, seeded 32-dimensional projection.

Training greedy frame decodes and soft occupancies supply the confusion table, with each recording contributing equal total mass. Both consonants of a mined pair require positive occupancy from at least two training speakers. The union of each eligible consonant's two strongest observed confusions defines the pairs. Zero-count contrasts are not invented, and practice incidents never select pairs.

Each pair has its own nonlinear `Linear → Tanh → Linear` binary classifier, balanced by each class's expected soft occupancy mass with pair-specific sampling. An independently fitted conventional multiclass MLP uses the same features and balanced occupancies. Its trainable parameter count exactly equals the sum across all binary classifiers; integer-compatible hidden widths and, where required, one bias-free binary output achieve parity without dummy parameters. These independently trained binary losses are not algebraic combinations of multiclass logits.

Fixed seed 1729, 18 epochs and learning rate 0.002 apply to both arms. Weight decay is selected from 0.0001 and 0.01 through three speaker-grouped folds. Pair mining, feature normalization and model architecture are recomputed without each validation fold. Real scripted substitutions are weak preservation checks only: the unmodified engine must first request an overlapping written-versus-rival comparison. Their scripted words are never assumed to have been performed. Selection minimizes held-speaker reversal rate on those comparisons, then mean speaker soft NLL. Both pairwise and equal-parameter control validation NLLs and capacities are recorded. No available preservation comparison, insufficient speaker coverage, or no mined confusion stops export rather than silently supplying an unvalidated scorer.

## Runtime mechanism

`packages/sdk/src/pairwise-phone.ts` reconciles the independently fitted log odds through least-squares Bradley–Terry scores in each connected phone tournament. It normalizes each component while preserving the multiclass control's total probability mass for that component; unpaired phones and blank retain the control probabilities. This gives one coherent phone distribution even when binary cycles disagree.

The checker exposes only complete pronunciation comparisons for otherwise confirmed, acoustically supported string-path wrong words. It uses the exact existing warning window, including the existing contextual or joint pronunciation where applicable. A substitution diagonal in the edit script is unnecessary. Both written and already-accused pronunciations use the same encoder rows and complete CTC boundary marginalization. The scorer uses identical normalized outgoing timing priors conditioned on completion, avoiding a pure path-count reward for a pronunciation with more legal boundary paths. Unknown phones, invalid features and impossible complete sequences abstain.

A certificate requires strict disagreement: the pairwise scorer prefers the complete written pronunciation while the matched multiclass control prefers the rival. The certificate records both comparisons, word identity, pronunciations and exact frame bounds. The checker validates the reversal and exact identity before changing the public verdict to uncertain. Internal confirmation/ownership state remains baseline so later candidate generation and insertion decisions are preserved. Parallel-graph confirmations are exempt.

Certificates travel as optional token metadata through stored evidence. Sessions also review original live evidence before gap splicing and auxiliary insertions, so a graph-confirmed live warning cannot be vetoed merely because an auxiliary replay aligns differently. Primary acoustic outputs, phones, confidence values and tracker input are untouched.

The node loader recognizes a separate hashed descriptor declaring the fitted heads and feature graph, validates their sizes, hashes, inventory, primary graph checksum, dimensions and parameter parity, and fails on incomplete declarations. A directory without that descriptor follows the baseline loader. No descriptor or model path has been declared in this worktree because no model was trained.

The exact-window callback, session evidence transport and feature-export scaffolding reuse plumbing from `exp/listener-vicinal-phone-head`. The training objectives, sampling, confusion selection, tournament reconciliation and matching control implement this card's pairwise mechanism.

## Validation and blocker

All six local Python mathematical tests passed with `arch -x86_64 python3 -m unittest discover -s modal -p test_student_pairwise_phone.py`. They exercise soft occupancies against exhaustive CTC paths, sparse contradictory-cycle reconciliation, complete-sequence acceptance, training-confusion eligibility, rejection of unobserved contrasts and parameter parity. A separate dependency-free check also passed exact parameter parity in 20 architectures. The x86_64 invocation is required here because the installed NumPy extension has that architecture; the default arm64 invocation previously failed during import before any assertions ran.

A supplemental native Node assertion harness ran the SDK/checker unit-test bodies with Node's TypeScript transform and strict assertions, without installing dependencies. Fourteen scenarios passed: candidate symmetry and strict reversal, normalized inconsistent tournaments, complete pronunciation timing, abstention, metadata-only transport, schema/parameter checks, primary frame/cache parity, exact-window discovery and identity matching, public verdict downgrade, actual insertion preservation, graph exemption and EOF session downgrade. This is not a Vitest run or a TypeScript type check. The harness is an untracked local diagnostic under `.autolab/pairwise-native-check.mjs`.

All four Python files compiled, all twelve changed TypeScript files parsed with Node's TypeScript transform, and `git diff --check` passed. Full dependency-based tests, builds, real ONNX loader tests and training remain outstanding.

Three permitted supervisor requests failed before execution with exactly `NotFoundError: No such file or directory`, zero compute charged:

1. Read-only help-train/model census.
2. Six-shard `bash tools/autolab_tests.sh` for the required core/SDK builds and full suites.
3. Training-image mathematical tests, builds, extraction, fitting and complete bundle verification.

The errors are retained in `.autolab/modal-result-1.json`, `modal-result-2.json` and `modal-result-3.json`. No remote tests, data census, feature extraction, fitting, graph export or acoustic replay ran. The failed training request was made before the final mining/report/export refinements; it must be rerun from the committed source when the supervisor works.

## Limits and outstanding work

The runtime currently buffers audio and performs the auxiliary feature pass at session completion. Stored replay can consume the resulting certificates, but live users can receive a warning before its EOF downgrade; this does not establish a streaming-time veto. Memory grows with recording length and the separate feature pass adds compute. Small-data contrast coverage, calibration, speaker generalization, real substitution preservation and actual target removals are unresolved.

Resume with the authorized supervisor: run the full suites and `python3 modal/student_pairwise_export.py --model /lab/data/models/spliced-clean-int8 --out /scratch/listener-pairwise-phone-tournaments/bundle --labels /lab/data/help-train/labels.jsonl`. Verify source auxiliary assets, real graph/cache parity, trained parameter parity, provenance, validation report and normal SDK loading. Only after the complete bundle passes should `.autolab/model.json` declare `{"path":"bundle"}`. Inspect all three target comparisons and actual final warnings using the SDK CLI with the supplied audio/evidence paths. Apply the user's continuation rule, and practice-score only if all three targets are removed.

No killed outcome is written: target gains and losses are unknown, and infrastructure failure does not meet the premise kill rule. The optional engine implementation is retained for resumption, not presented as a measured or release-ready result.
