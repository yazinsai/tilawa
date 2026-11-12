# False-sequence unlikelihood head refit

Status: blocked by the authorized Modal supervisor before data inspection or
implementation. Unmeasured; the experiment card is unchanged.

## Prediction and mechanism

The card predicts listener −1 and overall −1 on lab-v1.5u. Freeze the shipped
encoder and mine complete erroneous phone sequences behind confirmed wrong-word
accusations on help-train training speakers' none-labelled recordings. Exclude
sequences compatible with existing lawful pronunciation variants. Retain positive
recordings of the targeted pronunciations and ordinary speech and pauses, and
preserve shipped outputs on mistake-labelled takes rather than treating their
canonical written text as a performed transcript.

Refit the phone head with transcript CTC plus sequence unlikelihood: penalize the
summed probability of every CTC path spelling a mined erroneous sequence, not
individual phone classes. Compare with a positive-only head using identical data
and update counts. Partition and select by training speaker, excluding release
held-out speakers. Export the selected head with unchanged vocabulary, timing,
tracker, checker and confirmation rules, retaining every inherited auxiliary asset.

The task's final stopping instruction requires at least three of the four target
confirmations removed on complete streaming replay relative to the matched
positive-only head. Fewer than three wins or an invalid premise kills the
experiment; at least three wins requires an official practice score before any
decision. This supersedes the card's narrower zero-win criterion. No training
setting may be selected using practice results.

## Work and validation limits

Read AGENTS.md, the experiment card, the lab spec and audio-review instructions.
Inspected the existing streaming listener, session finalization, reference word
forms and frozen-head capture/export utilities on prior experiment branches.
Existing frozen-head utilities provide capture and export paths that enforce
encoder-tensor preservation; that code inspection does not establish compatibility
with the supplied model, the training premise or any target gain.

Prepared an ignored training-only mining probe in
`.autolab/unlikelihood-mine.ts`. It separates training speakers before mining,
checks existing lawful forms, searches for an unambiguous complete observed phone
sequence in SDK evidence and looks for training-only positive recordings. It has
not executed, and its eligibility rules still need data-backed review before use
in a frozen training plan. No head refit, checkpoint selection or bundle export
has been performed.

The authorized supervisor failed to launch all three requests on 12 November
2026. Each local result contains
`{"error": "NotFoundError: No such file or directory", "usd": 0.0, ...}`;
there is no command output or collected shard log:

| Request | Command | Result |
| --- | --- | --- |
| 1 | Python inventory of LAB_DATA, help-train labels, available modules and model files | Launch failure |
| 2 | `pwd` | Same launch failure |
| 3 | `bash tools/autolab_tests.sh`, six shards | Same launch failure |

The minimal second request distinguishes this from an error in the inventory
script. The exact missing supervisor resource is unknown. No alternative Modal
route, credentials, exam data or files outside the authorized worktree/data route
were sought. Node's syntax check of the ignored TypeScript probe passed; it is not
an SDK build or runtime test. Core/SDK tests did not execute.

No practice score was requested (all three permitted calls remain). Eligible
negative counts, positive-pronunciation coverage, training losses, target wins,
clean-session behavior and studio regressions are all unknown. An unavailable
runtime is not evidence of zero wins or a false premise, so no killed outcome is
written. No engine change or model selection is left in place: running the main
listener would not evaluate sequence unlikelihood.

Resume by restoring the authorized supervisor, executing the training-only
premise probe, then building the matched refits if the premise holds. Training
selection must finish before complete target replay; a surviving candidate needs
its complete exported bundle verified before declaring `.autolab/model.json`.
