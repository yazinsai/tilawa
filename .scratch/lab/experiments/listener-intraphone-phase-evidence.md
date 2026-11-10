# Ordered intraphone acoustic evidence

Status: needs-human — implementation draft; the supervisor could not execute
Modal requests. Training, the four target comparisons and population validation
remain blocked. This branch is not ready for scoring. The experiment card is
unchanged.

## Prediction and premise

Predict listener −1 and overall −1: one false substitution removed and no
predicted losses. The hypothesis is that a rival can explain a sound's average
spectrum while failing to explain its ordered attack, sustain and release.
The test compares complete pronunciations, including the interior-noon case;
it does not require an onset defect or a substitution diagonal.

The card's kill criterion is zero ordered-versus-pooled reversals on the four
registered complete comparisons. The task additionally requires at least three
actual target warnings removed to continue to practice scoring. Neither
criterion has been evaluated. Infrastructure failure does not establish a
premise kill or a measured zero gain.

## Implementation

The SDK computes the existing 80-bin log-mel frontend independently, preserving
its Povey window, DC removal, pre-emphasis and frame geometry. Four neighbouring
mel coordinates are averaged into each of twenty log-spectral coordinates.
Observations retain the 10 ms clock: an existing native 40 ms warning interval
indexes four observations per frame, without moving its endpoints or averaging
its temporal progression. Observation buffering is independent of microphone
push size.

Each pronunciation phone has three mandatory ordered states, attack, sustain
and release. Self loops marginalize phase durations; release-to-attack edges
marginalize phone boundaries. Repeated phones retain separate complete chains.
A forward sum integrates all monotonic placements of the entire pronunciation;
no target phone is made optional. Scores are conditioned on completing that
same chain within the observed window by subtracting its emission-free timing
partition. Unsupported phones, incomplete windows, non-finite observations and
windows over 2,048 observations abstain.

The matched control uses exactly the same fitted Gaussian emissions, transition
probabilities, complete-phone topology and timing normalization. At every phase
of a phone it replaces the phase-specific emission with the equally weighted
mixture of that phone's three emissions. It therefore retains the acoustic
inventory while removing the assignment of spectral observations to ordered
phases. A certificate requires a strict pooled preference for the rival and a
strict ordered preference for the written pronunciation. Ties abstain.

`modal/student_intraphone.py` fits diagonal Gaussian emissions and shared phase
transition probabilities by forward/backward EM. Initialization distributes
observations uniformly along the complete transcript; subsequent updates
integrate phone and phase boundaries. Sparse phone-phase statistics are shrunk
toward distributions shared by Arabic manner, place and voicing attributes.
The feature groups and variance floor are fixed before any target comparison.

Training inputs must explicitly identify none-labelled help-train training
speakers, carry non-exclusion provenance, and supply SDK observations and their
canonical pronunciations. A separate required exclusion list is checked as
well. Scripted mistakes are never treated as performed transcripts. Model
selection uses three hash-ordered, speaker-disjoint training folds; shrinkage
candidates 8, 32 and 128 are selected by held-speaker marginal predictive
likelihood, with ties favouring stronger shrinkage. Final parameters are fitted
on all eligible training speakers. The report records exclusions, input hash,
fold membership and every selection score. These safeguards still require an
audit against the actual supplied split metadata; that audit could not run.

The checker exposes its actual confirming non-graph comparison: the complete
written and already-accused pronunciations and exact original frame endpoints.
This covers its ordinary, word-local and paired support paths. The callback is
read-only unless the SDK's phase comparison vetoes. Serialized certificates must
match the word, both complete pronunciations and both endpoints exactly, and
contain finite scores with the required reversal. Graph confirmations are
exempt. The internal confirmation and ownership ledger keeps its original
verdict; only the public wrong-word verdict and returned mistake list change.
This preserves inputs to insertion-tail merges and later candidate generation.

Audio sessions can apply the comparison before publishing confirmation.
Recorded evidence receives certificates through the ordinary token metadata
route after existing gap decoding and teacher insertion enrichment. Evidence-only
sessions consume the certificates without waveform access. There is no broad
end-of-session retraction based on a different replay's warning list.

The optional Node bundle loader verifies `phase-asset.json`, its fixed
`phase-emissions.json` filename, byte count, SHA-256, inventory, clocks and
parameter shapes. A declared missing or corrupt model raises an error and
releases the backend. Primary-only bundles load normally. The trainer exports
the side scorer, descriptor and training report; supplied manifests and registry
data are untouched.

## Validation and execution block

All three authorized supervisor requests failed before container execution
with `NotFoundError: No such file or directory`:

1. Run 1 requested a read-only help-train, data, scratch and numerical-library
   layout check.
2. Run 2 requested the six-shard `bash tools/autolab_tests.sh` harness while the
   draft was being assembled.
3. Run 3 retried that six-shard harness after the main implementation and initial
   tests were present.

The result files report zero compute cost and provide no container logs.
Consequently neither required TypeScript build nor complete core/SDK Vitest
suite has been verified. The later graph-preservation test and session
simplification were checked locally; no Modal test snapshot executed.

Ten SDK/core cases passed locally using Node's TypeScript transformation on
workspace-local copies with import suffixes adapted and a minimal assertion
harness. They cover an ordered-versus-pooled reversal, retention of a genuinely
reversed pronunciation, exhaustive enumeration of phase and phone boundaries,
complete repeated-phone requirements, unsupported-input abstention, native
frontend observations and audio rechunking, model validation, exact certificate
matching, unchanged issue creation and preservation of a graph confirmation
when the callback rejects every string comparison.

Three small Python numerical unit tests passed with an isolated, ignored NumPy
environment: forward/backward values and state marginals match exhaustive
boundary enumeration; training/exclusion provenance is enforced; and a tiny
synthetic EM fit produces finite, positive parameters. No recording was fitted
on the Mac. Eight Python/SDK complete-sequence scores from synthetic parameters
matched to a maximum absolute difference of 5.68e−14, in both ordered and pooled
arms. A separate mocked-backend check using real temporary files passed optional,
verified, missing and corrupt side-scorer loading, preserved the primary
manifest and verified backend release on failures. The session module imported
under the same temporary TypeScript transformation. Python syntax compilation
and `git diff --check` passed. Local diagnostic copies and toy assets are ignored
under `.autolab`; they are not deployable model artifacts.

These checks establish small numerical and transport properties. They are not
full-stream preservation checks, real audio evaluations, the required complete
suites, or independent listening judgments.

Zero target recordings were accessed and zero practice scores were requested;
all three practice-score allowances remain. Target wins, losses, overall change,
clean-session behavior and studio behavior are unknown. There is no fitted
help-train model and no complete side-asset bundle. No `.autolab/model.json` is
declared: scoring the primary listener alone would not evaluate this experiment.
An ignored outcome records the infrastructure block, without claiming a kill.

## Resume and limits

Restore the supervisor execution path, inspect the authorized help-train split
and exclusion provenance, and extract SDK observations for its none-labelled
training speakers. Fit and freeze the scorer before querying the actual
engine-requested windows of the four targets. Verify its complete exported
bundle with the original primary ONNX, manifest and any installed teacher files
intact, then declare only that verified bundle through untracked
`.autolab/model.json` under the experiment's persistent Modal scratch directory.

Run the required build/tests and full-stream comparisons for primary evidence,
tracker events, graph confirmations and insertion decisions. If fewer than
three actual target warnings are removed, revert the draft engine changes and
record a killed outcome with the measured count. If at least three are removed,
request practice scoring before narrowing on any aggregate gate failure.

Uniform initialization can converge to poor latent boundaries; three mandatory
phases can misrepresent short phones and duration-expanded labels. The model
has no explicit phone-independent background state for utterance-edge silence.
Diagonal emissions discard correlations, coarse feature tying may erase real
phonetic differences, and tiny training coverage may cause abstention. Phase
names denote ordered latent states, not independently annotated physiological
boundaries. Transition updates fit the joint completion likelihood while
inference and selection condition on completion; this does not establish an
optimal conditional duration fit. The current implementation retains session
observations and audio for enrichment, and duplicate frontend work plus replay
cost still need device-budget measurement. Node bundle loading is implemented;
web/native asset installation is not. Positive written-word preference does
not prove that a weak or unfamiliar genuine substitution will survive.
