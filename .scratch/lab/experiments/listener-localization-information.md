# Listener boundary-localization information

Status: needs-info — implementation and empirical validation are blocked on the supervisor's Modal runner.

## Prediction

The unchanged card predicts listener −1 and overall −1, with no counted losses. Its three registered omission false warnings are `lab-v1.5-p50/phone/warning:omitted_word/37:113:10/279.32`, `lab-v1.5-p65/phone/warning:omitted_word/4:12:40/114.12`, and `lab-v1.5-p65/phone/warning:omitted_word/4:15:6/210.60`. The implementation brief additionally requires three target wins before a practice score. Neither condition has been measured.

## Mechanism implemented so far

`packages/sdk/src/localization-information.ts` computes the joint start/end distribution of a word in a mandatory whole-ayah CTC read. The graph retains the ordinary contiguous word and ayah repeat arcs used by `PosteriorFrames.nll`; it has no skip or filler arcs. The first phone's first frame is the start and the last phone's last frame plus one is the end. Internal blanks belong to the word; facing-edge blanks do not.

The statistic uses the final contiguous complete visit to the flagged word. A full forward prefix sums earlier visits and ayah repeats; a suffix constrained not to re-enter the word selects its final visit. Internal word repeat arcs remain available. Every legal complete read contributes once, so repeat multiplicity does not turn the boundary distribution into an expected visit count. Partial-word repeat arcs are rejected because they lack complete-word boundaries.

The control keeps the same rows and graph. Only non-blank emissions in the flagged word's states are replaced by the mean non-blank probability of each row. Blank emissions and every surrounding word's emissions stay fixed. Localization information is KL(actual joint boundaries || identity-neutral joint boundaries), in nats. It does not compare read-versus-omit likelihoods or best durations. Both joint distributions are retained, including their start/end dependence.

`packages/sdk/src/localization-calibration.ts` implements leave-one-speaker-out training partitions. Each partition's initial rejection point is the other speakers' scripted-omission maximum. It records held-out omission false rescues and none-labelled reach, then freezes the envelope at the maximum omission information across training speakers. A veto must strictly exceed this point. None-labelled words cannot lower it, and examples marked `eval` are excluded. The real dataset's split and speaker metadata still need inspection before invoking this function; scripted omissions remain weak labels.

## Validation

Added SDK unit tests for exhaustive joint path mass, agreement with the existing CTC likelihood partition, duplicate repeat arcs, blank-preserving neutralization, first/final word boundaries, repeated-label minimum durations, absolute clocks, and speaker-grouped calibration with held-out exclusion.

Small direct Node checks of the pure TypeScript implementation passed locally. On an independently enumerated seven-frame, three-phone graph with word and ayah repeat arcs, the joint distribution summed to `0.9999999999999996`, its log partition was `−3.4204535982678497` (identical to exhaustive enumeration), and its largest joint-probability error was `3.3306690738754696e−16`. A two-identical-phone word required its intervening CTC blank and its joint distribution summed to 1. A synthetic speaker-partition check excluded an eval-speaker omission score of 100 and retained the training rejection point of 3. These are mathematical checks, not recitation results.

Supervisor requests 1 and 3 attempted read-only data inspection; request 2 attempted `bash tools/autolab_tests.sh` with six shards. All returned `NotFoundError: No such file or directory` before execution, with no command logs or data. A fourth request containing only `pwd` also returned the same error. The user was notified through the asynchronous input tool.

## Limits and remaining work

No training recordings or target evidence have been inspected. There is no frozen empirical rejection point, no target-win count and no practice score (0 of the allowed 3 scores used). Core and SDK suites have not executed remotely. Runtime omission-veto integration is deliberately unfinished until the training-only rejection point and premise can be evaluated; the committed primitives do not change decoding or warnings by themselves. No model, side-ear asset, deployment override or benchmark-specific engine rule was added. The experiment card, scorer, test data, manifests and evidence are unchanged.

The algorithm costs O(T² × word-state-count + T × ayah-state-count) time and O(T² + T × ayah-state-count) memory for T ayah frames. Its real-ayah runtime and memory use need measurement on Modal. Information can still concentrate on misplaced neighbour evidence; conservative weak-label calibration does not establish safety on unseen speakers.

After the runner is restored: inspect training split/speaker metadata, decode only authorized training recordings, freeze and record the speaker-grouped calibration before examining target statistics, then evaluate all three targets on the existing whole-ayah interval. If the premise fails or fewer than three targets are won, remove the SDK primitives and tests and record the measured kill. If three targets pass, finish ordinary listener-evidence veto transport, validate the scored path and tests, and take one practice score before deciding anything further. Infrastructure failure alone is not a premise kill; no killed outcome or invented measurements are declared.
