# Acoustic recurrence graph

Status: needs-human — implementation draft; Modal execution is unavailable.
The experiment card is unchanged. No acoustic outcome or practice gain has been
measured, and the draft is not ready for supervisor scoring.

## Prediction

The accepted prediction is listener −1 and overall −1, with no forecast losses.
The mechanism tests whether within-occurrence nonadjacent acoustic relationships
can reverse a complete written-versus-accused pronunciation comparison while the
otherwise matched local-emission control still prefers the accusation.

The card names fewer than two qualifying target windows as its premise kill.
The execution instructions additionally require at least three target wins to
continue to a practice score. Neither threshold has been evaluated.

## Implementation draft

`modal/student_recurrence.py` accepts only none-labelled, training-split
help-train recordings. Canonical CTC forced alignment supplies dense acoustic
phone supervision, preserving repeated pronunciation labels. Internal blank
intervals are assigned to their nearest articulation. It fits standardized,
16-dimensional pooled SDK log-mel observations with phone-conditioned diagonal
Gaussian local emissions. For each word occurrence it fits cosine self-similarity
between frames separated by at least one intervening frame, conditioned on their
pronunciation-phone pairs. Unsupported pairs share same-phone or different-phone
priors; the phone-independent similarity distribution supplies a matched null.

Three folds partition whole speakers. Shrinkage candidates 8, 32 and 128 are
selected by held-speaker predictive similarity likelihood, and the final fit uses
all eligible training speakers. The relation weight is fixed at one, the search
retains eight paths per phone frontier, and windows are bounded at 256 native
frames. None of these choices has been inspected against practice results.

The SDK scorer consumes every phone of both complete pronunciations, including
repeated articulations. Its bounded monotonic segmentation pool is shared by
the local and relational calculations. Similarity edges are evaluated only
inside the original accused-word interval, while any identical context in the
existing acoustic support window remains part of both local sequence scores.
Per-frame degree normalization prevents quadratic accumulation over held sounds.
The relational score adds pair-conditioned log likelihood relative to the
phone-independent similarity null.

The core discovery callback exposes an otherwise supported string-path
comparison's exact pronunciations, acoustic support frames and word interval.
A certificate is consumed only for that exact tuple, with finite scores, a local
preference for the accused pronunciation, a reversed relational preference and
positive written relationship evidence. Parallel-phrase confirmations bypass
this path. Primary phonemes and posteriors are unchanged; metadata uses the
existing token-preserving evidence transport.

The Node loader can verify a declared scorer's byte count, SHA-256, inventory
and parameter dimensions. No scorer descriptor is registered: no fitted asset
exists yet. The session draft retains audio privately when needed, enriches
recorded evidence at completion and replays the unchanged tracker/checker to
downgrade vetoed wrong words. Other live events and auxiliary insertion publication
remain on their original path. This completion-time integration still needs
end-to-end verification and does not establish immediate streaming warnings.

## Validation and blocker

All three authorized supervisor requests failed before execution with
`NotFoundError: No such file or directory` and produced no container logs:

1. Run 1: read-only LAB_DATA/help-train/model layout inspection.
2. Run 2: the supplied six-shard `tools/autolab_tests.sh` build/test harness.
3. Run 3: minimal working-directory, LAB_DATA and scratch inspection.

The supervisor result files report zero compute cost. This failure does not
establish absent training data, absent recurrence evidence or a target kill.

Lightweight local checks passed: Python syntax compilation; SDK invariants for
complete repeated phones, nonadjacent recurrence, adjacent-frame exclusion,
affirmative reversal and parameter validation; all three new core certificate
tests for read-only discovery, positive reversals and exact-frame/sequence
matching; and `git diff --check`.

The executable local checks used Node's TypeScript transformation on temporary
copies with import suffixes adapted; they did not run Vitest or replace the
required Modal build/test harness. Existing suites, loader behavior, session
replay and numerical fitting remain unverified.

No help-train recording or target window was accessed. No model was fitted,
no scorer bundle or `.autolab/model.json` was declared, and zero practice
scores were requested. Target wins and overall change are unknown. The local
data-preparation draft remains under ignored `.autolab/recurrence/`; it has not
run. No scorer, lab tooling, supplied manifest, test data or evidence was edited.

## Resume requirements and limits

Restore the supervisor snapshot/run storage, then inspect the authorized data
layout. Decode none-labelled training speakers, excluding any speaker assigned
outside the training split; fit and freeze the scorer before inspecting target
comparisons. Obtain candidates from the actual unchanged engine rather than
manually chosen target boundaries. Verify at least three actual target vetoes
before invoking practice scoring. If the built premise fails, revert the engine
draft and record a killed outcome with the measured target count.

If it survives, finish the primary-plus-side-asset bundle, retaining all existing
teacher files and the primary model/manifest unchanged; register the fitted
side asset with its integrity metadata and declare the verified bundle only in
untracked `.autolab/model.json`. Check primary evidence, tracker events, graph
confirmations and insertion decisions, and run the full required tests and
practice score.

Dense phone supervision can inherit forced-alignment errors. Pair distributions
may poorly transfer from real learner clips to degraded reciter audio. The
bounded pool approximates the boundary marginal and may omit useful placements.
Positive written relational evidence limits intervention but does not prove
that a genuine substitution will survive. These are unmeasured limitations,
not reasons to kill without the required target and gate results.
