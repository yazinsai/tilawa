# Lexical identity and pronunciation factors

Status: needs human — supervised Modal execution is unavailable. The acoustic premise is untested; this experiment has not been killed on its premise.

## Prediction and mechanism

Predict two false warnings removed and one caught substitution lost: listener −1,
overall −1. Four phone wrong-word false warnings are registered. The card asks
for at least two unfactorized-rival to lexical-written preference reversals;
the task additionally requires at least three target wins to continue to practice
scoring. Neither condition has been measured.

Train a small acoustic-text head on frozen streaming-student encoder features,
with separate lexical and pronunciation factors. None-labelled and
tajweed-labelled help-train recordings supervise shared lexical identity;
substituted-word scripts provide weak change supervision. Marginalize canonical
versus scripted readings and segment boundaries rather than declaring the script
a verified performed transcript. Exclude vowel-labelled recordings and
release-held-out speakers. A pronunciation-task loss retains tajweed variation
in the nuisance factor, with separation regularization selected solely on
speaker-grouped training partitions. Use an otherwise matched unfactorized
control.

At existing string-path wrong-word comparisons, score complete written and
already-accused pronunciations over identical frames. A certificate can veto
only when the lexical head strictly prefers the written pronunciation and the
control strictly prefers the accusation. Preserve primary decoding, tracker
events, parallel-graph confirmations and insertion decisions.

## Work and validation

Read AGENTS.md and the unchanged card. The earlier attempt had recorded three
supervisor failures before execution and left an ignored partial draft. This
retry restored and revised that draft: frozen-feature graph extraction and
loading, latent-compliance CTC training, lexical/control sequence comparisons,
and metadata transport through ordinary evidence. Inspection found and corrected
a draft scope error that excluded lexical string warnings, and added draft
session handling to defer wrong-word publication until review while reusing the
original tracker events. These are draft changes, not validated behavior.

Python syntax parsing passed. No training, TypeScript build, unit test, target
comparison or runtime parity check executed successfully. This retry used only
the prescribed supervisor interface:

| Request | Purpose | Result |
| --- | --- | --- |
| 4 | Inspect help-train labels, exported models and Python dependencies | `NotFoundError: No such file or directory` before execution |
| 5 | Six-shard `bash tools/autolab_tests.sh` | Same error before execution |
| 6 | Minimal `pwd` workspace probe | `NotFoundError: No such file or directory` before execution |

Requests 4–6 returned zero compute cost, with no command log or shard output.
The earlier requests 1–3 had the identical failure. This establishes an execution
infrastructure block, not absent training supervision, failed tests or a failed
acoustic premise. No practice score was requested; all three allowances remain.

## Why this retry cannot ship engine code

The required learned factors cannot be instantiated from source code alone. The
authorized help-train audio, label schema and speaker partitions, and the actual
frozen encoder graph, are available through the supervised container rather
than this worktree. The failed data probe provided none of them. Consequently
it is impossible here to verify usable none/tajweed/substitution supervision,
exclude held-out speakers from the fit, identify compatible encoder features,
select regularization, train both heads or export their complete bundle. Target
preference reversals and preservation of primary outputs also remain unknown.

Leaving a loader or certificate hook in place without those trained assets would
run the main listener without evaluating the proposed mechanism. The task
explicitly requires finishing and verifying the complete bundle before handoff.
For that reason all unverified engine changes were removed again and no
`.autolab/model.json` declares an incomplete model. This write-up is the only
tracked change; its lack of engine code is an infrastructure limitation, not a
claim that the experiment was implemented or scored. The card, supplied
manifests, scorer, lab tools, data and evidence were not changed.

The revised partial implementation is retained in the ignored
`.autolab/implementation-draft.patch` for resumption. It still needs actual input
schema validation, training-partition checks, meaningful unit tests, training
and SDK feature parity, model/export integrity, and end-to-end warning review
checks. After supervised execution is restored, finish those steps, freeze
selection entirely on training speakers, and test the four actual engine-requested
comparisons. Apply the premise/target-win rule only after those measurements;
if three or more targets are won, request practice scoring before assessing
collateral losses. No fold or discard was performed.
