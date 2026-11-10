# Lexical prominence evidence

Status: ready-for-human — supervisor execution is unavailable; implementation
cannot be completed and the acoustic premise remains unmeasured.

## Prediction and mechanism

The unchanged card predicts one false substitution warning removed, no losses
counted: listener −1 and overall −1. Consonant blur might leave relative
syllable prominence intact despite a misleading decoded pronunciation.

The proposed CPU scorer compares complete written and already-accused
pronunciations on identical existing string-path warning windows. It models
relative vowel duration, energy and pitch movement, conditioned on syllable
weight and stop-versus-connected position. Boundaries and recitation
lengthening are marginalized; the model must learn prominence rather than
impose a stress rule. Fit and model selection use only none-labelled help-train
training speakers, with speaker-grouped training-only validation and
release-held-out speakers excluded.

A certificate requires different predicted prominence patterns and a frozen
acoustic preference for the written pattern. Primary decoding, tracker events,
parallel-graph confirmations and insertion decisions remain unchanged. The
task requires at least three actual target wins before practice scoring,
despite the card's smaller prediction.

## Why the engine change cannot be completed here

The authorized supervisor fails before container execution. The current
attempt retried data inspection, a current target trace and the required
six-shard tests. All three returned `NotFoundError: No such file or directory`,
zero compute cost and no container logs.

These failures prevent access to the authorized training recordings, their
speaker split assignments, the shipped listener bundle and the actual target
waveforms and warning windows. Local unit fixtures do not supply the required
none-labelled training corpus. Thus no acoustic parameters can be fitted or
selected by the stipulated procedure, and no complete side-scorer bundle can
be exported and verified. A source-only hook with no trained asset would run
the main listener unchanged and would not evaluate this experiment. This is
the reason no engine change is shipped after the previous attempt's feedback;
the mechanism has not been implemented or measured successfully.

The failure does not establish that the corpus is absent, that prominence
patterns agree on the targets, or that the kill criterion fired. No killed
outcome is asserted. Target wins and overall change remain unknown. No
practice score has been requested in either attempt; all three allowances
remain available.

## Draft and validation limits

The current attempt restored and inspected the previous draft, then removed
it from the engine diff. The draft remains under ignored
`.autolab/prominence-draft/` for resumption. It contains a checker discovery
hook and an SDK syllable/design compiler, not a trained acoustic scorer.

Discovery exposes the complete otherwise-supported written and accused
sequences and the exact existing acoustic interval, marking the word within
its shared context on ordinary, word-local and paired string paths. Graph
comparisons bypass the hook. Stop conditioning uses the existing labelled
pause test. The compiler groups expanded madd into a nucleus, retains
gemination as coda/onset weight, centers predictor inputs within the word and
abstains for identical designs. Identical designs guarantee identical
predictions for every fitted parameter, so consonant changes alone cannot
trigger this mechanism.

Seven local structural assertions passed using Node's TypeScript
transformation, covering madd expansion invariance, gemination weight,
identical-design abstention, differing-weight eligibility, vowel-free
abstention and stop conditioning. The earlier attempt also checked a
synthetic discovery/veto replay. Neither check is a TypeScript build, a
Vitest suite, training, independent listening, target evaluation or practice
scoring.

## Supervisor request record

Request and result files are ignored under `.autolab/`.

| Run | Request | Result |
| --- | --- | --- |
| 1 | Previous training-data/model inspection | Pre-execution `NotFoundError` |
| 2 | Previous six-shard required tests | Pre-execution `NotFoundError` |
| 3 | Previous minimal working-directory/data inspection | Pre-execution `NotFoundError` |
| 4 | Current help-train layout, label split/kind counts and model layout inspection | Pre-execution `NotFoundError` |
| 5 | Current trace for the registered 55:48:1 target at 282.64 s | Pre-execution `NotFoundError` |
| 6 | Current six-shard `bash tools/autolab_tests.sh` | Pre-execution `NotFoundError` |

Restore the authorized supervisor before continuing. Obtain exact target
comparisons and test the differing-design prerequisite, fit and freeze the
scorer on training speakers, then evaluate acoustic preferences on the target
windows. Fewer than three wins requires reverting and recording the premise
kill. At least three requires the complete verified bundle, passing required
tests and practice scoring; aggregate losses belong to the fold gate.

Variable recitation style, uncertain vowel boundaries, sparse weight coverage
and unreliable pitch on phone audio remain untested risks. A positive
prominence preference can still erase a genuine substitution. The scorer,
lab tooling, supplied manifests, evidence, test data and experiment card were
not changed. No model override, fold or discard was performed.
