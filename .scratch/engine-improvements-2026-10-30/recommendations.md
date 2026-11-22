# Recommended improvements to Tilawa

30 October 2026. Research and diagnosis only; no engine, model, label or scoreboard changes. Three parallel reviewers investigated the tracker, checker and listener with ultra reasoning; the coordinating review checked measurement validity and prioritized the results.

## Recommendation

Start with the current full int8 model. Align the benchmark with the SDK, improve tracking through ambiguous/repeated passages, and improve the checker's evidence and word boundaries. In parallel, prepare reviewed learner/error data. Use that data for a controlled listener adaptation programme after the cheapest logic experiments establish a baseline. A bigger model has not earned priority: teacher and student have similar false-warning rates on the long stitched regression set.

## The order to run experiments

| Order | Experiment | Why this comes first | Evidence of success |
|---|---|---|---|
| 0 | Share the SDK/replay acquisition policy and record configuration with results | Session replay omits the SDK's one-second acquisition cadence. Different prefix decisions materially change results. | Matching policies produce matching tracker events on identical evidence; remaining differences are explained. This is measurement repair, not an accuracy claim. |
| 1 | Guard local tracking confidence; retain competing passage/occurrence paths | A recorded trace stays `locked` with zero confidence and has a premature jump within a similar passage. Other traces already contain the correct candidate but commit to a competing one. | Fewer wrong-position episodes and faster recovery, without replacing the problem with long `lost`/ambiguous periods or losing real jumps. Test repeats, restarts, identical passages and real errors separately. |
| 2 | Require independent local anchoring for declared-passage accusations | On the same six recordings, assignment produces 99 clean warnings versus 41 at no context; all 58 extra warnings occur in two recordings. | At least halve the excess in the diagnostic development cases, then confirm on other speakers. Preserve intended-passage substitutions and errors at the start. Do not make global tracker agreement mandatory. |
| 3 | Compare plausible word boundaries and acoustic hypotheses before confirming | The current CTC check already exists, but its audio window and competing text come from one alignment. High token confidence has accompanied human-confirmed false warnings. | Material clean-warning reduction at matched true-error recall, with no new losses hidden by gains elsewhere; bounded warning delay. Compare against a simple threshold-only baseline. |
| 4 | Give repeated visits distinct decision state; localize the changed word and error class | Correct and wrong visits to one Quran word can share cached decisions. Human review also found real substitutions reported on the preceding word and an omission called a substitution. | Correct lifecycle on repeat/self-correction cases; better exact word/class accuracy. Report this separately from reducing false accusations. |
| 5 | Resolve training-target choices, then train on known errors and real learners | The chosen student mixes reference targets and teacher distillation with a confounded weight comparison. It can restore the correct textbook sound when a wrong sound was spoken. | Preserve audible errors while improving clean speech: deployed int8 PER, word-error recall at fixed precision, and continuous-session false warnings must improve together. |

Orders 1–4 are small isolated prototypes, not a proposed simultaneous rewrite. Combine only independently successful changes and rerun the combination. Candidate numeric thresholds in the detailed notes are proposed experimental decision rules, not evidence of achieved gains or replacement release gates.

## How to approach each area

### Tracker: uncertainty should change the decision

Keep several plausible locations alive across time, with explicit transitions for continuing, repeating, restarting and jumping. A shared phrase need not choose its ayah until the next distinguishing words arrive. Use a consistent, bounded context prior for the chosen passage, page or resume point; preserve a separate observed position so the prior does not erase genuine mistakes or intentional jumps.

First, address the smaller observed problem: a poor local path should not automatically continue to display `locked`. Use evidence quality and competing path separation for that status, with a recovery mechanism. Measure both wrong-location time and time without a usable location to prevent an always-uncertain tracker from appearing improved.

Posterior evidence is a separate bounded ablation. The SDK intentionally supplies the tracker only decoded phonemes today. Score a small candidate shortlist using real listener uncertainty instead of treating every decoded phone as certain, and measure the runtime cost. Do not conflate this with multi-path state in one change.

### Checker: verify the accusation, not just the transcript

The checker already asks whether the decoded alternative fits better than the written words. Improve the question: does it still fit better under a few plausible boundaries, lawful duration variants, and a stable local alignment? Word identity must remain protected. Confidence in a decoded token is not the probability a person made a mistake.

For an assignment, the intended passage is known but the audio still needs to be anchored locally. Correct neighboring words can support that anchor even when the target word is wrong and the global tracker prefers another ayah. Preserve this distinction when reducing assignment-only warnings.

Use occurrence-aware state so a prior correct repetition cannot make the next repetition immune to error checking. Preserve an explicit relation between a wrong attempt and its self-correction. Prefer evidence from a new following word to repeated checks of the same frames. A longer timer alone is not evidence.

### Listener: teach what was said, including mistakes

First run the missing controlled 2×2 comparison: teacher-path versus spliced-reference target, crossed with distillation weight 1 versus 2. The existing winner changed both factors. Use a development split for choices and a fixed final evaluation; periodically viewing a split named `test` does not keep it untouched.

Then add a modest, reviewed set of error examples, including real learner clips and controlled audio splices with matched correct controls. A wrong spoken word must have a wrong-word **as-said** target. Disable or replace teacher supervision around edits if it contradicts the audible error; retain clean context and clean replay to limit forgetting.

Treat phone/channel adaptation and tempo adaptation as distinct experiments. Review fast-speech reference conventions before treating their disagreements as acoustic failures. Delay new encoders and expensive broad training until these tests show the existing architecture's limit. Improve the full model before distilling improvements into lite.

## Data and evaluation are part of the work

- Use the saved human reviews now. Keep genuine false accusations, word/class localization failures, incomplete labels and unresolved audio cases separate. Retain strict scoring and add diagnostic incident scoring; a more forgiving metric is not an improved engine.
- Build a small reviewed development panel of continuous learner sessions containing clean speech, repeats, self-corrections and mistakes with genuine same-speaker context. Existing clips cannot supply this: the coverage audit found no minute-long same-speaker run and no error clip with both same-speaker clean neighbours.
- Reserve distinct speakers/recordings for final validation. Reuse the current clips, QUD and E1b as regression/challenge sets with their limitations explicit. The current QUD expansion has no error positives and is reported-only.
- Compare paired outcomes: exact errors gained/lost, genuine false accusations removed/added, location coverage, wrong-position episodes and warning delay. Show speaker-level variation and uncertainty. Keep abstentions in the missed-error denominator.
- Retain the existing release bars. Any later metric-moving implementation needs the normal scoreboard row before merge. No report here claims those bars have been met.

## Detailed reviews

- [Tracker](tracker.md): code paths, trace examples, acquisition-cadence replay, temporal candidate and posterior experiments.
- [Checker](checker.md): five experiments, local hypothesis verification, assignment anchoring, occurrence state, reference variants and diagnosis.
- [Listener](listener.md): controlled target/loss ablations, error/learner adaptation, tempo/channel hypotheses, compute estimates and primary sources.
- [Evaluation](evaluation.md): benchmark-policy mismatch, attribution limits, reviewed label issues, paired session counts and experiment contract.

The first cheap deliverable should be a trustworthy paired baseline plus one tracker and one checker prototype. The training deliverable should be a reviewed pilot dataset and an isolated adaptation result, not a new architecture chosen on clean PER alone.
