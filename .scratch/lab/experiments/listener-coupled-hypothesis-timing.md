# Coupled pronunciation timing

Status: implemented, target premise and full validation blocked by the supplied
Modal supervisor route. No target-win count or practice score is available. The
experiment card is unchanged; the implementation remains on its experiment branch.

## Prediction and mechanism

Predict listener −1 and overall −1: independently movable shared phones may
supply a rival's advantage without stronger evidence for the changed sounds.
The recheck compares the complete written and already-accused pronunciations on
the exact posterior rows used by the original confirmation. Context phones stay
complete; no phone becomes optional and no window boundary moves.

The SDK builds a paired ordinary CTC graph. Every maximum monotone exact-phone
correspondence participates, including alternative correspondences for repeated
phones. Matched phones have identical entry, occupancy and exit frames on both
tracks. Differing stretches retain separate CTC paths. Ordinary repeated labels
still require intervening blank; primary decoding and pronunciation expansion
are unchanged.

Each hypothesis has a uniform normalized prior over its complete legal timings.
Each correspondence likewise has a uniform normalized prior over legal paired
timings, and correspondences have equal prior weight. The joint acoustic weight
is the geometric mean of the two complete-path likelihoods, so equal shared
emissions contribute once and differing emissions receive symmetric weights.
Forward expectation propagation marginalizes shared boundaries jointly. The
coupled statistic is the posterior expectation of the complete-path log
likelihood ratio, with the same timing-prior normalization for both hypotheses.
Equivalently it is the derivative of the paired log partition at equal
hypothesis weight. It is an auxiliary statistic, not a new interpretation of
the original SUPPORT confirmation bar.

A matched uncoupled control uses exactly the same statistic with the shared
boundary constraints removed. This distinguishes coupling from merely changing
the aggregation statistic. A veto requires all of the following:

- The original full-sequence forward sum favours the rival.
- The normalized independent forward sum also favours the rival.
- The matched uncoupled control favours the rival.
- Independent Viterbi paths assign different boundaries to shared phones.
- The coupled complete-path preference strictly favours the written sequence.

Impossible complete paths, missing evidence, absent shared phones and graphs
with more than 1,024 correspondences abstain. The correspondence resource guard
never truncates a marginal or drops target phones. No quantity is fitted on
practice or help-train.

The core exposes complete acoustic comparisons only after the unchanged
confirmation bar passes, covering word-local and neighbouring-word string
paths. Parallel-graph confirmations bypass the recheck. The SDK runs the
scorer synchronously on rows already available to the streaming checker. It
also carries span- and pronunciation-specific veto certificates on recorded
primary tokens for ordinary JSON serialization and scored evidence replay.
Tokens, confidence values, posterior rows, tracker input and insertion
certificates are preserved. There are no new model assets or model selection
files.

## Validation and limits

Ten new unit cases passed under lightweight local Node execution of the real
TypeScript modules with Node's type transform, node:test and assert. The local
runner substitutes assertion bindings for Vitest; this establishes runtime
behaviour of those cases, not TypeScript type checking or a pass of the full
Vitest suite. It ran only these new small CPU cases, not the existing heavy
suite.

The scorer checks agree with exhaustive enumeration of complete raw CTC paths,
including normalized timing priors and alternative correspondences. They also
cover antisymmetry, identical hypotheses, repeated articulations, missing rows,
and preservation of strong changed-segment evidence. An unrelated synthetic
four-frame example has written-minus-rival independent preference −0.115449,
matched uncoupled preference −0.106134 and coupled preference +0.283070, with
different independently optimal shared-phone boundaries. This demonstrates a
real coupling reversal in the implemented statistic; it says nothing about the
registered Quran incidents.

Streaming tests cover incomplete-row abstention followed by a complete-window
veto, repeat evaluation, candidate-order symmetry and unchanged primary rows.
Core tests verify discovery of actual supported comparisons and certificate
transport through JSON, exact binding to word, both pronunciations and frame
interval, and interaction with an installed live recheck. `git diff --check`
passed.

The supplied supervisor requests were:

1. Four-shard build and target diagnostic on the card's listener evidence,
   using both the free tracker and true-position replay.
2. Six-shard `bash tools/autolab_tests.sh` for the required build and existing
   core/SDK suite.
3. A minimal one-shard `pwd` environment probe.

Every request returned `NotFoundError: No such file or directory` before
execution, with no shard output and no compute expenditure reported. The
failure therefore supplies no acoustic, test or scoring result. Neither
direct Modal credentials nor another remote route was used. A request to
restore the supplied environment remains outstanding.

The target diagnostic discovers comparisons from the unchanged checker;
incident locations only select which requests to report. It records complete
sequences, exact windows, independent Viterbi boundaries, correspondence count,
all three control preferences and the coupled preference, then replays the
veto through the checker. Its temporary source and the lightweight local runner
are retained under ignored `.autolab/diagnostics/`, outside the committed engine
diff, for retry after infrastructure recovery.

The card requires at least two premise-positive targets; the task's explicit
continuation rule additionally requires at least three target warnings won.
Neither prerequisite can currently be assessed. There is no justified killed
outcome, and no population gain, preservation claim or gate verdict. Zero of
the three allowed practice scores have been used. Once the route works, run
the target diagnostic and full suite; roll back and record a killed outcome if
the target criterion fails, otherwise obtain the required practice score.

The implementation retains posterior rows and caches comparisons for the
session, and a large paired graph can add synchronous confirmation latency.
Memory and latency have not been measured on device. A veto can lose genuine
substitutions or indirectly affect later checker decisions; small unit controls
do not establish preservation on practice, clean sessions or studio audio.
No scorer, lab tooling, supplied evidence, manifests, test data or experiment
card was changed. No fold was performed.
