# Lexical identity and pronunciation factors

Status: needs human — supervised Modal infrastructure failed before execution. The premise is untested; this is not a killed experiment.

## Prediction

The card predicts two false warnings removed and one caught substitution lost: listener −1, overall −1. It proposes four phone wrong-word false warnings as targets. Its reversal prerequisite is two targets; the task's continuation rule additionally requires at least three target wins before practice scoring.

## Mechanism

Train a small head on frozen streaming-student encoder features, separating lexical identity from pronunciation variation. None-labelled and tajweed-labelled help-train recordings share their canonical lexical identity. Substituted-word scripts supply weak change supervision: marginalize canonical versus scripted readings and segment boundaries, rather than declaring a script to be the performed transcript. Exclude vowel-labelled recordings and all release-held-out speakers.

An auxiliary pronunciation task retains tajweed information in the nuisance factor. A separation penalty and nuisance lexical adversary discourage that factor from deciding lexical identity. Choose regularization using speaker-grouped partitions of training speakers alone. Compare against an otherwise matched unfactorized control.

At existing string-path support comparisons, score complete written and already-accused pronunciations on the same existing acoustic interval. A certificate may veto only when the lexical factor strictly prefers the written sequence and the control strictly prefers the accusation. Primary decoding, tracker events, graph confirmations and insertion decisions must remain intact.

## Work and validation

Read AGENTS.md and the unchanged experiment card. Inspected the SDK's existing teacher-certificate transport, native evidence rechunking and the checker's sequence-support windows. Prepared a partial implementation of comparison discovery, score-certificate transport, an auxiliary frozen-feature loader, and a CPU training utility with latent script compliance and speaker-grouped regularization selection. Python syntax compilation passed. No training or runtime correctness was established.

The supervisor rejected all three requested runs before executing their commands:

| Request | Purpose | Result |
| --- | --- | --- |
| 1 | Inspect help-train labels, available model bundles and Python dependencies | `NotFoundError: No such file or directory` |
| 2 | Required six-shard `bash tools/autolab_tests.sh` | `NotFoundError: No such file or directory` |
| 3 | Minimal `pwd` infrastructure probe | `NotFoundError: No such file or directory` |

Each response reported zero compute cost and provided no command log or shard output. These failures therefore establish an infrastructure block, not a missing training-data category, a failed test, or a failed acoustic premise. No target was inspected or evaluated, no model was trained or exported, and no practice score was requested; all three practice-score allowances remain available.

## Limits and handoff

All draft engine changes were reverted because the complete auxiliary bundle could not be trained or verified. The branch contains this write-up only. No model selection file declares an incomplete bundle. The original card, manifests, scorer, lab tools, test data and evidence remain unchanged. No fold or discard was performed.

An ignored `.autolab/implementation-draft.patch` retains the partial draft for resumption. It is not ready to score: feature-graph compatibility, actual help-train coverage, separation quality, unit tests, streaming event handling and complete-bundle parity still need validation. Restore the supervised runner, inspect the training-only inputs, finish the implementation and tests, freeze the training-selected head, then measure the four exact engine-requested target comparisons. Apply the premise/target-win stopping rule only after those measurements. If at least three targets are won, run practice scoring before evaluating collateral losses.
