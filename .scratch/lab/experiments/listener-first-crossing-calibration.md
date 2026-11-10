# Listener first-crossing calibration

Status: blocked by supervisor Modal infrastructure; unmeasured.

## Prediction and proposed mechanism

The immutable card predicts one fewer false warning, no counted losses, listener −1 and overall −1. The proposed explanation is repeated opportunities to cross a substitution bar on distinct, overlapping acoustic windows, rather than accumulation of their likelihoods.

The proposed implementation would retain each pending word occurrence's existing sequence-score trajectory, distinguish identical rechecks from changed windows, and calibrate its running maximum conditional on word length and distinct-window count. Calibration must use complete trajectories from none-labelled help-train training speakers, resample whole trajectories, and freeze selection through speaker-grouped training-only validation. Mistake-labelled recordings serve as output-preservation controls; their scripts are not verified performed transcripts. Release-held-out speakers must be excluded. A failed first-crossing acceptance test would travel as a listener veto certificate, with primary decoding, tracker events, graph confirmations and insertion decisions preserved.

## Source inspection

`packages/core/src/checker.ts` owns the ordinary sequence verification used by SDK sessions. `supported()` computes the current written-minus-decoded CTC negative-log-likelihood margin on one posterior stream. Its branches include word-local, paired-word, parallel-phrase and ordinary neighbour-context comparisons. No previous margin is accumulated or maximized.

Ordinary string-path suspicions wait for the confirmation delay and a following onset. `decide()` computes support again on the current evidence; either verdict seals the word against reopening. Lexical candidate discovery also performs support checks before adding an issue, and frozen parallel slots retain their frozen decision evidence. Consequently, counting calls alone cannot establish the proposed failure. The necessary runtime question remains whether a target confirms after genuinely changed overlapping windows, with an earlier maximum over the existing bar but a later complete-window score below it.

## Diagnostic and validation attempts

A temporary read-only diagnostic callback captured the existing score, required bar, complete written and decoded hypotheses, acoustic frame endpoints, evaluation time and verification branch. Actual calls were distinguished from additional probes of candidates in later alignment windows, including after confirmation. Identical endpoints and hypotheses identify an identical recheck on the immutable posterior stream. A generic SDK trace utility replayed both the ordinary engine and true-position arm; target identifiers appeared only in command arguments, never in engine code.

The requested baseline was exactly the card's listener, `exp-listener-teacher-insert-marginal-gate-ad6eb8c0`, on `lab-v1.5u` phone evidence. The four requested sessions were p11, p36, p55 and realuser sob-049. The card requires at least two targets with the temporal premise; the user's continuation rule additionally requires three wins from a built intervention before practice scoring.

Supervisor requests:

| Request | Purpose | Result |
| --- | --- | --- |
| 1 | Four-shard target trajectory replay | `NotFoundError: No such file or directory`; no shard logs or trajectories returned |
| 2 | Six-shard `bash tools/autolab_tests.sh` | Same wrapper error; no test results returned |
| 3 | Minimal filesystem/workspace preflight | Same wrapper error; no command output returned |
| 4 | Corrected SDK-source target trajectory replay | Same wrapper error; no shard logs or trajectories returned |

The initial temporary utility was placed under `.scratch`, which the supervisor's source archive omits. It was moved to `packages/sdk/src/bench` and its reference-asset paths were made relative to its module before request 4. The independent tests and filesystem preflight also failed, so the wrapper failures cannot be explained solely by that initial utility placement. The available results do not identify the missing filesystem object or prove a storage-capacity cause.

## Limits and remaining work

No target trajectories, help-train calibration data, calibration parameters, target wins, losses or practice score have been obtained. Practice-score requests used: **0 of 3**. The core/SDK tests were requested but could not be verified. No hypothesis kill criterion has been established. Infrastructure failures must not be recorded as zero wins or as a negative experimental result.

All temporary engine and SDK diagnostic changes were removed. The final tracked change is this write-up; the card, scorer, lab tooling, test data, evidence and manifests are unchanged. No model or side-ear asset is declared. The diagnostic callback patch and corrected generic replay utility are archived locally under ignored `.autolab/first-crossing-diagnostic.patch` and `.autolab/first-crossing-trace-cli.ts`; the four wrapper-error responses are `.autolab/modal-result-{1,2,3,4}.json`. The untracked outcome marks infrastructure blockage, not a killed hypothesis, and contains no invented measured numbers.

Once the authorized supervisor runner works, first repeat the target trajectory probe and verify observational equivalence with the ordinary engine. Only if the registered temporal premise survives should the training utility, frozen calibration and listener certificate consumption be built and unit-tested. A built rule winning at least three targets requires the practice score before a gate-based decision.
