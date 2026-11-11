# Listener alignment-label noise marginalization

Status: infrastructure blocked; implementation retained, real training and target validation outstanding.

Experiment: `listener-alignment-noise-marginal`, listener/explore, branch
`exp/listener-alignment-noise-marginal`. Implementation commit: `d7cab038`.
The existing `lab-experiment.json` is unchanged.

## Prediction and decision

The preregistered prediction is listener −1 and overall −1: one fewer false
wrong-word warning, with substantial zero-gain risk. The proposed cause is incorrect
phone supervision at uncertain internal boundaries in intact training recordings.
It concerns acoustic placement of a fixed transcript, without changing the words
or assuming that a scripted edit was performed.

No target audio or help-train audio could be accessed. Four authorized supervisor
Modal requests failed before returning a container command log with
`NotFoundError: No such file or directory`. Real training, training-speaker
validation, target preference reversals and warning removal are unknown. This is
an infrastructure block, not a premise kill; untested targets are not counted as
zero wins. No practice score was requested, leaving all three allowances unused.
No claim about practice, studio or release performance is made.

The card's zero-reversal criterion remains untested. The task's stricter
continuation rule also applies: revert the engine implementation and record a kill
if the built, training-frozen change removes fewer than three registered warnings.
If it removes three or more, practice scoring must precede any narrowing based on
losses elsewhere. The fold decision belongs to the gate.

## Implemented mechanism

`modal/student_alignment_noise.py` fits two identical-capacity phone-conditioned
Gaussian emission heads on eight fixed mean log-mel coordinates, at the SDK
frontend's native 10 ms observation rate. Each head has a mean and diagonal
variance per phone, with the same shrinkage formula, features, recordings,
transcript and inference timing model.

1. The extraction CLI admits only none-labelled help-train training recordings.
   It excludes every speaker with any non-training recording, plus an explicit
   excluded-speaker list, before opening waveform files. The trainer independently
   enforces source, kind, split and exclusion metadata and requires three speakers.
2. Two transcript-constrained acoustic aligners are initialized independently
   using fixed random duration seeds 181 and 733. Eight EM iterations sum complete
   phone-boundary placements on intact recordings. Their most probable phone
   assignments supply a phone-level disagreement frequency, shrunk toward the
   pooled disagreement frequency with eight observations of prior weight.
3. Both heads receive exactly the first aligner's fixed assignments. The control
   treats those assignments as certain. The noise head models a latent choice
   between the assigned phone and its adjacent transcript phones. Twelve EM
   iterations integrate the assignment alternatives instead of deleting uncertain
   frames or teaching every boundary frame as the assigned phone. Each observation
   retains total assignment mass one. Zero disagreement yields identical heads.
4. Three speaker-grouped partitions, determined by speaker hashes, select
   regularization from 2, 8 and 32 using complete-transcript likelihood of the noise
   head. Feature normalization, both independent aligners, contamination rates and
   both heads are refitted exclusively on each partition's training speakers.
   Validation speakers influence selection only. The selected strength is then
   used to refit both heads on all eligible training speakers.
5. SDK inference compares complete written and existing rival pronunciations over
   the same actual engine-requested window. It sums all ordered, complete phone
   placements, with the same completion-conditioned geometric duration prior for
   both heads and both hypotheses. A certificate requires a finite, strict reversal:
   the control prefers the rival and the noise-trained head prefers the written
   pronunciation. Ties, unsupported phones and unavailable windows abstain.

Contamination concerns training labels. Inference uses each head's learned phone
emissions and does not grant an optional phone, change a pronunciation, reconstruct
speech or apply a contamination exemption to an accused word.

## Runtime and asset transport

The checker exposes the existing supported sequence comparison at confirmation,
including its word, full written/rival strings and exact frame coordinates. Graph
confirmations bypass this hook. Certificates must match all those fields. The
internal confirmed event and issue-ownership ledger remain the baseline's; the
public vetoed wrong-word event becomes uncertain and is excluded from mistakes.
This preserves downstream insertion ownership and candidate generation.

The SDK maintains native frontend observations for live audio confirmations and
also enriches recorded evidence after the existing gap and teacher insertion
processing. The enrichment discovers windows by running the ordinary engine,
without incident-specific windows, vocabulary or candidate construction. It
attaches certificates to copied evidence tokens for deterministic later replay.
The diagnostic CLI verifies unchanged tracker events and insertion decisions.

The Node directory/tier loader accepts an optional `alignment-noise-asset.json`
descriptor and checks its fixed `alignment-noise-emissions.json` file, byte count,
SHA-256, inventory, clock, dimensions and finite, positive emission variances.
A declared missing or corrupt file fails loading and releases the primary backend.
No supplied manifest, registry, scorer, lab tool or test data was edited.

No real fitted asset or complete model bundle exists. No `.autolab/model.json`
was written: decoding the shipped model alone would not validate this experiment.
The tiny synthetic utility smoke-test bundle is ignored under `.autolab/validation`
and is neither a deployable Quran model nor declared for scoring.

## Validation

Supervisor requests:

- Run 1: read-only data/label layout and scratch-capacity inspection.
- Run 2: the required six-shard `bash tools/autolab_tests.sh` harness.
- Run 3: a minimal container/data-path launch probe.
- Run 4: the six-shard harness after implementation and loader-test adjustments.

All failed with the same pre-execution `NotFoundError`. No alternate Modal route
was attempted. Required complete Modal tests and real ONNX bundle validation are
outstanding.

Local fallback checks were limited to small synthetic inputs and ordinary
repository fixtures, with temporary dependencies/configuration under ignored
`.autolab/validation`:

- **16 focused TypeScript tests passed**. They cover complete boundary sums,
  candidate-order symmetry, a strict finite reversal, missing phone coverage,
  frontend rechunking, exact certificate matching, unchanged issue creation,
  graph preservation, legacy teacher loading and verified/missing/corrupt side
  assets. ONNX loading is mocked in asset tests.
- **Five Python tests passed**. They cover normalized complete boundary marginals,
  assignment marginalization, exact equality with zero disagreement, correction
  of a synthetic misassigned boundary observation, and exclusion enforcement.
- Complete boundary sums matched exhaustive enumeration for up to six observations
  and every permissible phone count, with maximum error
  `1.3322676295501878e-15` nats.
- A complete three-speaker synthetic training utility run exercised grouped
  selection and export. The clear synthetic boundaries produced zero disagreement
  and identical heads in every fold. This establishes plumbing, not the premise
  on real speech.
- Core and SDK source type checking passed using temporary ambient declarations
  for unavailable ONNX peer packages. Python syntax compilation and
  `git diff --check` passed. This does not replace the full Modal suites.

## Resume and limitations

Restore the supervisor's execution path, inspect actual help-train metadata and
verify its speaker exclusions, then run the extraction CLI and CPU trainer. Save
the frozen training report before probing targets. The CLI's comparison mode
accepts waveform plus existing evidence and discovers all confirmation windows;
use the registered incidents only to select results for reporting. Compare
actual before/after engine warnings and verify primary evidence, tracker events,
graph confirmations and insertion decisions.

Before declaring a model selection, assemble a complete bundle under
`/scratch/listener-alignment-noise-marginal/` with the unchanged primary ONNX,
manifest, all existing teacher assets, and the fitted emissions/descriptor. Verify
every declared file and the runtime loader, then write the ignored model selection
and rerun the required suites. If fewer than three targets are won, revert the
engine changes and record the measured kill. Otherwise request practice scoring.

Two aligners can agree on the same wrong boundary; disagreement is a proxy for
label noise, not ground-truth contamination. Hard assignments retain only their
most probable placements before noise marginalization. The eight pooled mel
coordinates and diagonal emissions may lose relevant consonant distinctions.
The transcript chain has no independent silence state and only one duration
persistence, so edge silence and pronunciation expansion can bias fitting.
Shrinking to pooled moments may obscure rare phones. Training uses none-labelled
recordings only; preservation on real substitutions remains unproven. Real audio
coverage, memory use and recheck latency are unmeasured. Node asset installation
is implemented; web/native side-asset installation is not. Positive preference
reversal does not guarantee preservation of weak genuine substitutions.
