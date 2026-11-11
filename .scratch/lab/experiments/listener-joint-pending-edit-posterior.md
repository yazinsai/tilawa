# Joint pending edit posterior

Status: blocked by Modal execution infrastructure; premise unmeasured. No engine
change is retained and the experiment card is unchanged.

## Prediction and mechanism

The card predicts listener −1 and overall −1, with no predicted losses. Its two
registered cases are the false wrong-word accusations in p11 at 104.32 s and p55
at 107.08 s. The proposed cause is competing unresolved lexical edits assigning
shared acoustic evidence to the wrong word.

The prototype discovers existing lexical substitutions and additions before
their individual posterior support tests and the wrong-word/addition tail merge.
It includes lexical hypotheses that have not yet independently confirmed. It
constructs compatible edit subsets over the available current and following
ayah, including unchanged text, and scores complete CTC chains on identical
original posterior frames. Each explanation consumes every frame once; each
event pays the existing PHRASE_SWITCH prior once outside the frame recurrence.
Word and ayah repeat transitions remain available. It generates no free phones.

The comparison sums all replacement explanations against explanations reading
the accused word correctly while retaining at least one neighboring edit. A
prospective veto requires the latter mass to be greater, and requires the
independent comparison to have preferred replacement. Duplicate hypotheses are
merged and alternatives for the same event are exclusive. Graph overflow returns
unavailable evidence rather than certifying a pruned posterior.

This was a premise-test prototype, not a completed streaming veto integration.
In particular, retention through the decision horizon, certificate consumption,
and preservation of live warning behavior still require implementation and
verification if the acoustic premise survives.

## Validation and limits

Read AGENTS.md and inspected the current checker, CTC scorer, lexical candidate
construction, SDK auxiliary evidence transport and audio-review instructions.
The earlier horizon-conditioned and ayah-marginal experiment branches named in
the brief are unavailable in this git snapshot, so no earlier implementation was
copied or represented as reproduced.

Four authorized supervisor requests failed:

1. Run 1: trace both target sessions using the card's selected listener.
2. Run 2: minimal container/data-directory check.
3. Run 3: the supplied six-shard build and core/SDK test harness.
4. Run 4: build and run the joint-posterior target probe.

Each returned exactly `NotFoundError: No such file or directory`, with zero
reported compute cost and no container log or downloaded output. These failures
occurred before target evidence or test results were available; they do not show
that the hypotheses are absent or that joint marginalization fails.

The prototype includes unexecuted unit tests for exhaustive small CTC path sums,
event priors, duplicate hypotheses, incompatible alternatives, overlapping greedy
spans, missing evidence and graph overflow. No test pass is claimed. No target
posterior, target recovery, independent listening judgment or aggregate wrong-call
measurement was obtained. Zero of the three practice-score allowances were used.

The prototype and its diagnostic source are saved in ignored
`.autolab/prototype.patch` for resumption. All engine and test edits were rolled
back. No scorer, lab tooling, test data, manifest, evidence or experiment-card
edits were made. No model bundle was declared and no fold was performed.

Resume after supervisor Modal execution is restored: apply the saved patch,
run the unit tests and target probe, and evaluate the task's kill instruction
using measured results. Infrastructure failure is not recorded as a premise kill.
