# Assistant bridge distillation

Status: blocked before implementation; unmeasured. This is not an acoustic kill.

## Prediction and mechanism

The card predicts two recovered additions and one collateral wrong call: listener −1, overall −1. Its premise is that the capacity gap between the original Muaalem teacher and the shipped student loses weak consonant distinctions. An intermediate-width assistant would learn from that teacher, then be frozen to supervise the final shipped-size student.

Both student arms must use the same architecture, starting checkpoint, training recordings, transcript loss, shipped augmentation and ordinary-reading preservation. The direct-distillation control must receive the same total training compute, including the bridge arm's assistant training and target generation. Only authorized transcribed Quran recordings and help-train training speakers' none-labelled takes may provide transcript supervision. Release-held-out speakers must be excluded, and checkpoint selection must use speaker-grouped partitions within training speakers. The final compact exports must retain the existing inventory, streaming clock and insertion-verification assets.

## Source audit

Read AGENTS.md and the audio-review procedure before work. Inspected the existing student training, preparation and export code and SDK insertion-teacher frontend.

- The shipped `spliced-clean` primary is the `full` roughly 70M student, not the roughly 22M `lite` model. The original teacher is `obadx/muaalem-model-v3_2`, roughly 600M parameters.
- The existing trainer selects checkpoints on `clean-match test`. Reusing that selection path would violate this card. The experiment needs separate training-speaker selection without changing the existing defaults.
- S1 preparation retains the speaker `cluster`, but the current training loader discards it. The experiment must retain those groups, reject unknown speaker identities and exclude groups represented outside the authorized training split.
- The shipped augmentation uses RIR, noise, gain and SpecAugment. Its dataset creates an unseeded generator per item; paired experiment arms need reproducible draws while retaining that augmentation recipe.
- An SDK bundle must preserve the existing insertion teacher as well as the newly exported primary student. A primary-only bundle would change insertion verification and would not test the card as specified.

These are source findings and implementation requirements, not evidence that assistant distillation recovers the targeted consonants. A possible assistant initialization is to widen the feed-forward layers while preserving the existing projection through zero-initialized extra output columns; this has not been built or validated.

## Execution blocker

Used only the authorized supervisor interface. Three CPU requests failed before producing a shard directory or command log:

| Request | Environment | Purpose | Supervisor result |
| --- | --- | --- | --- |
| 1 | train, 10-minute timeout | Inspect available checkpoints, teacher assets, S1 schema and help-train labels | `NotFoundError: No such file or directory` |
| 2 | standard, 5-minute timeout | Minimal Python path/model-directory probe | `NotFoundError: No such file or directory` |
| 3 | train, 5-minute timeout | Minimal Python `/student` directory probe | `NotFoundError: No such file or directory` |

The first request was submitted at 2026-11-12 06:38:16 UTC. Both retries were submitted at 06:41:59 UTC; their failures were observed by 06:47:37 UTC. All three result records report $0 spent and the full $150 GPU allowance remaining. The errors do not identify the missing file or volume. There is no command output establishing that Python started.

## Validation and limits

No engine, test, card, scorer, test-data, supplied-manifest or evidence changes were made. No assistant or matched control was trained, no student bundle was exported, and no model override was declared. The local worktree has no PyTorch or node_modules; heavy work and the required core/SDK tests remain dependent on the unavailable supervisor environment.

Practice scores used: 0 of 3. Target recoveries and warning wins are unknown, not zero. No clean-session, studio or overall changes were measured. Neither the card's ordered-consonant criterion nor the user's three-target stop rule has been evaluated. There is no basis for a killed outcome or a fold decision.

Resume by restoring the supervisor's missing resource and rerunning the preflight. Then implement and validate the paired training path, freeze both training-selected checkpoints, verify complete compact exports, and compare full-session exported decodes on the six card targets. Apply the acoustic/target stop rules only to those measurements; if at least three targets are won, run the authorized practice score before deciding.
