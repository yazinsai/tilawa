# Within-word phonation marginal

Status: draft engine implementation; training and target validation blocked by the supervisor execution path. The experiment card is unchanged.

## Prediction

The card predicts two false warnings removed and one caught substitution lost: listener −1 and overall −1. Its premise is that voice quality can change inside a word, creating consonant evidence that a fixed-phonation acoustic model misinterprets. The task's continuation rule requires at least three actual target warnings removed before requesting practice scoring; the card's two-reversal criterion is not enough to continue under that rule.

## Mechanism

`modal/student_phonation_prepare.ts` extracts observations with the SDK frontend from explicitly none-labelled, non-excluded help-train training speakers. It accepts the importer's absent exclusion field on eligible rows, rejects exclusion reasons and unscored rows, and excludes every speaker appearing outside the training split. Other labels, repeats and annotated error regions are rejected. No mistake script is converted to a performed transcript. The utility takes a generic reference index to obtain complete single-ayah pronunciations; it accepts no practice sessions or target list.

`modal/student_phonation.py` fits a compact switching acoustic model. Twenty coordinates average the primary eighty log-mel bins in groups of four. Three additional observations measure un-preemphasized spectral tilt, harmonic-to-noise ratio from normalized waveform autocorrelation, and excitation regularity from regularized LPC-residual autocorrelation. All use the original 25 ms window and 10 ms analysis hop.

The hidden state is a product of phone position and three phonation states. Each phone has a diagonal Gaussian log-mel mean and variance. A phonation state's additive spectral offset is shared across every phone, and its three voice-quality observations have phone-independent emissions. A shared transition matrix allows phonation changes both inside and between phones. Leading and trailing background states explain recording-edge silence. Every pronunciation phone must be visited; no consonant becomes optional.

Forward/backward EM integrates both phone boundaries and phonation paths. Equal phone self/advance probabilities, conditioned on complete traversal, give a normalized uniform boundary prior. Those timing probabilities are fixed, so transition estimation cannot optimize an unnormalized completion likelihood. Sparse phone means shrink toward broad manner classes. Phonation offsets use additive weighted regression, with a centered gauge; no phone-by-phonation table can assign an unrestricted correction to a particular consonant. Three deterministic speaker folds choose emission shrinkage and phonation persistence from a preregistered training-only grid before the final all-training-speaker fit. The export includes its speaker folds, source hash, validation scores and asset checksum.

In `packages/sdk/src/phonation.ts`, the complete written and already-accused pronunciations are compared over identical existing warning windows. The switching arm marginalizes the phonation path. Its matched fixed arm uses exactly the same emissions and initial prior but holds one sampled phonation state for the whole window. Both marginalize phone boundaries with the same normalized timing construction. A certificate is supplied only on a strict reversal: fixed prefers the rival and switching prefers the written pronunciation. Ties, unsupported phones, impossible complete traversals and unavailable observations abstain.

The core exposes actual, already-supported confirming comparisons without changing their windows, candidate generation or confirmation bars. Certificates must match the location, both complete pronunciations and both frame endpoints exactly and carry finite reversal scores. Only the public non-graph wrong-word verdict becomes uncertain; the internal baseline confirmation and ownership ledger remains intact for subsequent insertion and merge decisions. Primary decoding and tracker input are untouched. Audio sessions can apply the scorer before publishing a confirmation; serialized evidence carries the same comparison on the first primary token after ordinary gap and teacher-insertion enrichment.

The Node bundle loader optionally reads `phonation-asset.json` and verifies the fixed side-model filename, byte count, SHA-256, inventory, clock, shapes and normalized probabilities. A declared missing or corrupt file raises an error and releases the backend. Primary-only bundles retain their existing behavior. Supplied model manifests and registry data are not edited.

## Validation

Three Python synthetic numerical tests passed: complete forward/backward marginals and transition counts agree with exhaustive product-path enumeration; training/exclusion provenance is enforced; and a small synthetic EM fit produces finite shared parameters. No recording corpus was trained locally.

Eight synthetic complete-sequence comparisons between the Python trainer and compiled SDK agreed within 5.684341886080802e−14, covering both switching and fixed arms, candidate reversal and repeated phones. Focused TypeScript tests check an actual fixed-to-switching preference reversal, retention of a genuinely spoken rival, exhaustive latent paths, symmetric candidate ordering, complete repeated-phone traversal, unsupported-phone abstention, finite silence observations, audio rechunking, exact certificate identity, ordinary JSON/native-frame transport, unchanged primary values and insertion metadata, graph preservation, and optional/verified/missing/corrupt bundle loading.

The TypeScript core/SDK build passed locally using isolated ignored test dependencies. The final SDK suite passed 35 tests with one existing help end-to-end test skipped. Two additional extraction-eligibility tests cover the importer's absent/false exclusion markers and exclusion of annotated mistakes, repeats and non-training speakers.

The controlled core suite passed 761 of 762 tests. Its remaining startup-persistence test exceeded its hard-coded 15-second timeout, and Vitest reported one worker RPC timeout. Retrying the startup file alone passed its other two tests but hit the same 15-second limit again. No assertion failure was reported. All six new core certificate tests passed, including insertion and graph preservation. This is incomplete local core validation, not a fully passing suite; the prescribed Modal build/test harness has not executed successfully.

## Execution block and result limits

Three authorized supervisor requests all failed before container execution with `NotFoundError: No such file or directory`:

1. Read-only help-train split/data/library/storage inspection.
2. The prescribed six-shard `bash tools/autolab_tests.sh` harness.
3. The full SDK feature-extraction, speaker-grouped EM training and export/bundle command.

Their ignored result files report zero compute cost and contain no container logs. No help-train recording or registered target waveform was accessed. No fitted side model, training report or verified production bundle exists. `.autolab/model.json` is deliberately absent: a primary-only decode does not evaluate this change. The ignored outcome records an infrastructure block, not a scientific kill.

Zero practice scores were requested; all three allowances remain. Target wins, preference reversals, caught-substitution losses, clean-session behavior, studio regression and overall change are unknown. Synthetic preference reversals and unchanged existing unit assertions are not real-audio precision measurements or independent listening reviews.

## Resume and model limitations

Restore the supervisor execution path, inspect help-train split/exclusion provenance, run the committed extraction/training utilities, and freeze the complete exported model before querying any target. Verify the bundle retains the original primary model, manifest and all baseline teacher files. Then measure the four actual engine-requested windows and confirm primary posteriors, tracker events, graph decisions and insertion decisions are preserved across full streams. If fewer than three actual target warnings are removed, revert the engine draft and commit a killed result. If at least three are removed, request practice scoring before narrowing on aggregate losses.

The three phonation states are latent voice-quality proxies, not physiologically annotated breathy, regular and creaky labels. Coarse mel coordinates and diagonal emissions discard acoustic detail; a small none-labelled corpus may poorly cover phones or phonation transitions. Uniform initialization can converge to poor phone boundaries, and a shared additive effect can still suppress a genuine consonant distinction. The fixed control holds phonation across the entire existing window, which may include neighboring words, rather than across a perfectly isolated word. Per-window comparisons are bounded at 4096 observations. The current implementation retains session observations and audio for enrichment and has not been measured against the offline device budget. Node installation is implemented; web/native asset installation is not. Positive written preference does not prove that a weak real substitution is safe.

No scorer, lab tooling, supplied test data, evidence, manifests or experiment-card edits were made. No fold was performed.
