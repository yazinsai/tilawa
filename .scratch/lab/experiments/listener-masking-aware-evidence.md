# Simultaneous masking acoustic evidence

Status: needs-human — engine implementation draft; supervisor execution is blocked.
The experiment card is unchanged. The scorer has not been fitted, no side-asset
bundle is declared, and this branch is not ready for practice validation.

## Prediction and premise

The prediction is listener −1 and overall −1: two false warnings removed and
one caught substitution lost. The mechanism tests whether temporary spectral
uncertainty from stronger neighbouring frequencies reverses a matched acoustic
scorer toward the complete written pronunciation.

The card names fewer than two target reversals as its kill criterion. The task's
execution instructions require at least three actual target wins to continue to
practice scoring. Neither threshold has been measured. Infrastructure failure
is not evidence for a premise kill, and no killed outcome is asserted.

## Implementation

The SDK observes the existing listener FFT, retaining its DC removal,
pre-emphasis, Povey window, frame geometry and spectral powers. The observer is
optional and does not alter the primary log-mel calculation. It aggregates power
into 24 equal-width Bark bands below 7.6 kHz and pools four 10 ms observations on
the native 40 ms evidence clock. Causal buffering gives identical observations
regardless of audio push boundaries.

For each weaker band, stronger neighbours within three band positions contribute
an asymmetric spreading threshold: 12 dB per Bark upward and 27 downward.
Observation variance is masking strength times the squared log of one plus the
threshold-to-observed-power ratio. This uncertainty changes with the speech in
each frame. It does not identify persistent missing bands, infer replacement
phones or reconstruct audio.

`modal/student_masking.py` accepts only none-labelled, training-split recordings
with explicit non-exclusion provenance, and additionally rejects speakers in a
required exclusion list. SDK observations and full-vocabulary shipped posteriors
are input alongside canonical phones. Canonical CTC forced alignment supplies
phone supervision; internal blanks are assigned to the nearest articulation.
Mistake scripts are never used as performed phone transcripts.

The utility fits phone-conditioned diagonal Gaussian emissions with EM for
latent unmasked log spectra. Observation uncertainty is analytically integrated
by adding its variance to the intrinsic phone variance. Global standardization
and pseudo-observations regularize each phone fit. Hash-ordered, round-robin
speaker partitions give three disjoint validation folds. Strength candidates
0, 0.25, 1 and 4 and shrinkage candidates 8, 32 and 128 are selected by mean
held-speaker predictive likelihood. Ties favour less masking and more shrinkage.
Zero strength is an explicit null option. The final fit uses all eligible
training speakers, and the report records folds, exclusions and the input hash.
These choices were specified without accessing target audio or practice scores.

The sequence scorer sums every monotonic placement of every phone in each
complete pronunciation. Each phone must occupy at least one native frame;
repeated labels retain distinct required states. A normalized uniform boundary
prior is shared by both arms. Windows longer than 512 frames, unknown phones,
phones without training observations and invalid observations abstain. Both
arms use the same fitted means, intrinsic variances, normalization and topology;
the control sets observation uncertainty to zero. A certificate requires a
strict control preference for the accusation and a strict masked preference for
the complete written pronunciation on identical original warning frames.

Core discovery reads comparisons only when an ordinary non-graph wrong word
would actually confirm. Certificates require finite scores and an exact match
of word, complete pronunciations and frame endpoints. The veto changes the
public final verdict to uncertain; the internal confirmation and ownership
ledger retains its original decision. Candidate generation, insertion-tail
merges, primary tokens, tracker events and graph confirmations therefore keep
their existing inputs and internal decisions.

Audio sessions can perform this comparison before publishing the warning.
Recorded evidence is also enriched through the ordinary listener-stage metadata
route, including comparisons on the existing gap-spliced evidence. Evidence-only
sessions consume certificates without requiring waveform access. The optional
Node bundle loader checks descriptor format, byte count, SHA-256, inventory,
clock and parameter dimensions. A declared missing or corrupt scorer fails
instead of silently using the main model. The training utility exports
`masking-emissions.json`, `masking-asset.json` and a training report; no supplied
manifest or registry data was edited.

## Validation and execution blocker

Three authorized supervisor requests failed before container execution with
`NotFoundError: No such file or directory`:

1. Run 1: read-only LAB_DATA, help-train, primary-model and scratch layout check.
2. Run 2: the supplied six-shard `bash tools/autolab_tests.sh` harness.
3. Run 3: a six-shard test retry after adding the acoustic and certificate tests.

All three result files report zero compute cost and contain no container logs.
Subsequent preservation and loader-test improvements were checked locally; no
Modal test snapshot executed. This does not establish absent training data or
absent masking evidence in the targets.

Nine acoustic and core certificate cases passed locally using Node's TypeScript
transformation on temporary copies with import suffixes adapted and a minimal
assertion harness. These cover signal-dependent asymmetric masking, unchanged
primary log-mel observations, audio rechunking, a synthetic preference reversal,
complete repeated phones, exact boundary marginalization, unsupported-input
abstention, exact certificate matching, and unchanged issue creation with only
the public final verdict altered. Python syntax compilation and `git diff
--check` passed; the session modules loaded under the same temporary transform.

These checks are not Vitest, a TypeScript build or the required complete core
and SDK suites. New Node-loader tests cover valid, missing and corrupt scorer
bundles, and the existing teacher-loader mock was adjusted to model an absent
optional descriptor; the Vitest loader tests still need Modal execution. A separate local check with
a mocked ONNX backend and real temporary bundle files passed optional, valid,
missing and corrupt scorer loading, checked backend release on errors, and
verified the primary manifest was unchanged. No actual
training fit or export, numerical Python/SDK parity check, release-speaker
exclusion audit, target comparison or full-stream preservation check ran.

No target recording was accessed and zero practice scores were requested.
Target wins and overall change are unknown. There is no `.autolab/model.json`
and no fitted asset: scoring the primary model alone would not evaluate this
experiment. The engine changes remain an inactive draft pending a complete,
verified side-scorer bundle.

## Resume and limitations

Restore the supervisor snapshot/run infrastructure, inspect the authorized
help-train split provenance, extract observations with the SDK implementation,
then fit and freeze the model before inspecting actual engine-requested target
comparisons. Verify the complete bundle with the original primary model,
manifest and any installed teacher files intact. Declare that verified bundle
only through untracked `.autolab/model.json`. Run the required build/tests and
check primary evidence, tracker events, graph warnings and insertion decisions.
If fewer than three targets are won, revert the engine draft and record a killed
outcome with the actual target count; otherwise request practice scoring before
allowing the fold gate to trade wins and losses.

The masking rule is a coarse signal-dependent uncertainty model, not a calibrated
human auditory judgment. Forced alignment can inherit listener errors, nearest
phone assignment of blanks is approximate, diagonal emissions ignore spectral
correlations, and sparse phone coverage can cause abstention. Gaussian latent
spectra and one-frame minimum durations can misrepresent degraded consonants
and expanded pronunciation labels. Positive masked support for the written
pronunciation limits intervention but cannot prove a genuine substitution will
survive. None of these risks has been measured on practice.
