# Bounded sequence-score readout

Status: blocked by supervisor execution. The implementation is prepared, but
training, acoustic measurements and official validation have not run. No model
bundle is declared. This branch is not ready for scoring or folding with a new
listener. The experiment card is unchanged; this is not a measured kill.

## Prediction and mechanism

The card predicts listener −1 and overall −1: a few poorly explained recordings
may dominate an acoustic head trained with log loss. The task's stopping rule
requires at least three target wins before a practice score. Neither the
premise nor a target win count has been established.

`modal/student_bounded_sequence_export.py` exposes the deployed encoder's final
projection input in a separate ONNX graph. Dynamic activation quantization is
included through dequantized activation codes; effective projection weights
include weight scales and zero points. The exporter checks all primary outputs
and caches for exact equality over eight sequential random streaming windows,
then checks that the recovered weight and bias reconstruct the logits. Those
model-dependent checks are implemented but remain unexecuted.

The SDK utility caches complete usable help-train none-labelled recordings on
the deployed streaming clock, with independent encoder caches. It excludes
non-training and explicitly excluded speakers, unusable recordings, mistakes
and repeats. Canonical authorized text supplies the complete original
transcript, including vowels and duration labels. Recording weights are
retained when present, otherwise each recording has weight one. No practice
case, warning label or held-out release speaker enters fitting.

Four equal-capacity linear heads start at the same effective shipped
projection. Each matched pair differs only in its training objective:

- Brier: the transcript, up to K distinct competing collapsed transcriptions,
  and one remaining-probability bucket form categorical outcomes.
- Log loss: negative exact log probability of the identical transcript, with
  the identical recording weight.

For each recording, a frozen reference-free prefix search proposes candidates
from the shipped posteriors. Search width is 32 and per-frame branching is
eight, plus blank. The transcript is inserted once and duplicates are removed.
K is fixed at eight competitors for the main pair and 32 for the enlarged-set
pair. Both sets use the same candidate ordering and include the full transcript.
The candidate proposal scores are not used as training probabilities.

Every named probability is an exact full-vocabulary CTC forward sum. With the
transcript first, the loss is

`(1-p[0])² + sum(p[1:]²) + (1-sum(p))²`.

Named masses are never conditioned on the candidate set, length-normalized,
temperature-scaled or substituted with individual best alignments. The loss
lies between zero and two. This is a proper categorical score for the fixed
named-outcomes-plus-bucket partition, rather than the full uncoarsened sequence
Brier score. Enlarging that partition can change learning; the second pair
tests whether the proposed preference reversal survives that change.

Fitting uses float64 CPU CTC, Adam at 0.0003, the same recording order and
update count for all arms, and a default 30 epochs (hard cap 120). No per-item
gradient clipping or recording reweighting changes the control's influence.
A deterministic speaker-grouped training-only fifth selects checkpoints by
each arm's own proper validation score, including the initial checkpoint.
Feature normalization uses fitting speakers only. Reports retain speaker
partitions, parameter counts, candidate limits, input checksum, selected
epochs, exact initial transcript log-probability quantiles and gradient norms.
The latter expose saturation if transcript masses are too small to learn;
such saturation is currently a possibility, not an observed result.

## Engine behavior

The checker exposes only its already-supported string-path wrong-word
comparisons. The SDK compares the complete written and already-accused
pronunciations over those same frames. A veto requires finite strict written
preference from the Brier head while its matched log-loss control prefers the
rival, with the same reversal under the enlarged-set pair. Unknown phones,
impossible hypotheses and inconsistent windows cannot certify a veto.

Certificates retain word identity, both complete strings, pronunciation parts,
frame limits and all eight scores. The checker validates them again and applies
them only to the identical comparison. Its internal confirmation ownership
remains intact, preserving subsequent issue merges and graph decisions.
Primary tokens, confidences, posteriors and tracker events are unchanged;
graph confirmations and insertion decisions keep their existing paths.

Audio sessions buffer the separate recheck and retract matching public
string-path warnings at session completion. Original live evidence is replayed
with and without certificates to isolate those retractions from gap splicing
and auxiliary insertions. This implementation has session-end latency;
streaming intervention latency and device resource use remain unvalidated.

The Node loader requires both hashed side assets and an exact primary-model
checksum match. Missing or corrupt declared assets fail loading. Bundle export
copies the entire existing directory, preserving any insertion teacher and its
external data, then adds the extractor, four heads and side descriptor without
editing the supplied primary manifest. A training-only parity fixture compares
all four exported JavaScript readouts with their PyTorch outputs.

## Validation and blocker

Supervisor requests 1 and 2 (data/dependency audits) and request 3 (the required
six-shard `bash tools/autolab_tests.sh`) returned
`NotFoundError: No such file or directory`, with no command output and $0
reported compute. The minimal `pwd` request, number 4, failed identically, independently
of repository commands and data paths. An environment repair was requested. No alternate Modal route or external dataset was sought.

Lightweight local checks passed:

- Python syntax checks for the trainer, extractor/bundler and mathematical
  tests; TypeScript syntax checks for all 11 changed source/test files;
  whitespace checks for the diff.
- The actual TypeScript CTC implementation matched exact enumerated mass for
  all 15 possible collapsed strings of a four-frame, three-class example,
  including empty and repeated-label strings.
- All nine assertions in the new core/checker and SDK mathematical test files
  passed using Node's native TypeScript transform and a small assertion runner.
  These cover strict matched-pair reversals, enlarged-set disagreement,
  unnormalized bucket loss, invalid labels, eligible training takes, exact
  certificate matching, stale frames and graph preservation.

These source checks do not replace TypeScript builds or the official core/SDK
Vitest suites. Audio-session and asset-loader tests were added but could not
run. PyTorch objective/gradient tests, deployed graph parity, label eligibility
audits, fitting, exported-readout parity and all four acoustic target replays
remain pending. No practice score was requested; all three permitted scores
remain available. No target was claimed won or lost.

`.autolab/model.json` and a killed outcome were not written. Without a complete
verified bundle, decoding the main model cannot evaluate this experiment.
Execution failure supplies no evidence to discard the scientific premise.

## Resume after execution is repaired

1. Inspect the supplied model and help-train labels/dependencies through the
   supervisor; run `bash tools/autolab_tests.sh` with six shards and
   `python3 -m unittest discover -s modal -p test_student_bounded_sequence.py`.
2. Export an extractor under the experiment's persistent scratch directory:
   `student_bounded_sequence_export.py --model ... --out ...`.
3. Cache eligible complete takes with the SDK's `bounded-sequence-cli.ts
   --model ... --extractor ... --labels ... --excluded-speakers ... --out ...`.
   Fit with `student_bounded_sequence.py --observations ... --metadata ...
   --out ...`, keeping the frozen recipe and training-only selections.
4. Verify all four heads with the CLI's `--verify ... --fitted ...` mode.
   Package the complete directory through the exporter's `--bundle ...
   --fitted ...` mode, and verify normal Node loading with every declared file.
5. Use the CLI's generic `--audio ... --evidence ... --fitted ...` route to
   inspect all actual confirmed comparisons in the target sessions. Keep
   diagnostic case selection outside engine code. Confirm before/after
   warnings and identical tracker and non-wrong-word decisions.
6. If the premise is false or fewer than three targets are won, revert engine
   changes and commit a measured kill with `.autolab/outcome.json`. Otherwise
   declare the verified complete bundle with untracked `.autolab/model.json`
   and run the authorized practice score before considering narrowing.

No scorer, lab tooling, supplied manifests, evidence, test data or experiment
card was edited. The prepared implementation is retained for resumption;
there is no acoustic result supporting a fold or an experimental kill.
