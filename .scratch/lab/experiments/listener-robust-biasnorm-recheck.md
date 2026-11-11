# listener-robust-biasnorm-recheck

Status: ready-for-human

## Prediction and premise

Prediction: remove two phone wrong-word false warnings and lose one caught
substitution, for listener -1 and overall -1. This is below the fold threshold
and within the lab's noise band. The experiment tests whether sparse hidden
channels dominate instantaneous BiasNorm energy and suppress distinguishing
consonants. The earlier normalization audit established instantaneous biased
RMS, with the original activation as numerator; it did not establish outliers
on these windows.

The registered targets are p11/26:106:1/104.32 s, p36/55:48:1/282.64 s,
p55/26:174:8/107.08 s, and realuser-v1-sob-049/92:13:1/36.44 s, all phone
wrong-word false warnings. No target identifiers appear in engine code.

## Implementation

`modal/student_robust_biasnorm.py` finds BiasNorm structurally in the exported
ONNX graph. It replaces the squared-channel mean with a Huber M-scale squared:

    mean(min((x - learned_bias)^2 / scale^2, cutoff^2)) = normal_consistency(cutoff)

The estimator starts from median squared energy with its Gaussian consistency
correction and performs twelve fixed clipped-energy updates. A small floor
protects degenerate vectors. The numerator, learned bias, learned gain, original
initializers, input/output declarations and streaming caches are retained.

The exporter admits only help-train training speakers' none-labelled original
recordings. Events, repeats, unsupported/excluded recordings, unscored records
and held-out speakers are rejected. A speaker appearing in any non-training
record cannot enter through another record. SHA256 of the speaker assigns all
of that speaker's recordings to fit or validation. No practice audio enters
calibration or model selection. Original Kaldi fbank features and canonical
phones come from the original recordings and generic reference inventory.

For each site, restoration gain is the geometric mean robust/ordinary RMS ratio
on fit recordings, weighting recordings equally. A matched gain-only graph
retains ordinary normalization and applies those same gains. Cutoff selection
from 1.5, 2 and 3 minimizes canonical CTC NLL on the disjoint training-speaker
validation partition; validation does not estimate gains. The declaration
records the candidates' robust and control losses, speaker partitions, gains,
and the maximum fit-recording single-channel energy share at each site. The
export copies the primary bundle unchanged, requires both existing insertion
teacher files, and writes both side graphs with SHA256 declarations tied to
the primary hash.

The SDK loader requires both declared graphs and checks their hashes and
parent identity before use. The recheck discovers existing string-path
wrong-word comparisons through the ordinary tracker/checker, then decodes the
same audio through independent full-session robust and gain-only streaming
states. It uses the original comparison and accused-slot coordinates. A
certificate requires complete written pronunciation under existing direct
soft-substitution, held-sound and pause/waqf stripping rules, failure to spell
the rival, a margin below the original confirmation bar, and strictly better
written-versus-rival preference than the gain-only control.

Certificates travel on primary phonemes and survive JSON evidence replay.
The checker requires the same word, hypotheses and exact frame intervals and
exempts parallel-graph confirmations. Primary phonemes, posteriors and live
tracker events are preserved. The live SDK withdraws only a certified
string-path warning at session completion after ordinary checker replay.
Existing insertion replay uses its original evidence before robust certificates
are attached.

## Validation

The eight Python unit tests pass locally. They cover bounded sparse-outlier
influence, scale/channel equivariance, admission and partition isolation,
retained original weights/bias/numerator/gain, gain-only behavior, refusal of
incomplete calibration and missing insertion assets, and actual tiny-streaming
cache/output geometry. The end-to-end tiny-bundle test fits and selects on
synthetic features, exports both graphs, checks their hashes and ONNX validity,
and verifies unchanged primary and teacher bytes. Its teacher files are inert
copy fixtures; this does not test production teacher inference.

Core and SDK TypeScript builds pass. The full SDK suite passes 36 tests with
the existing help end-to-end test skipped. The six focused core pronunciation
and certificate tests pass, including an actual-checker regression that vetoes
a wrong word while preserving an independently planted insertion unchanged.
The full core run passed 760 tests and failed one existing startup test on its
explicit 15-second deadline; that case took 16.38 seconds. Vitest also reported
two worker RPC deadline errors. The additional insertion regression was added
after that file had run and passed separately. All three startup assertions
passed in a focused diagnostic with a temporary 60-second deadline; the slow
case took 17.44 seconds. The original 15-second deadline was restored and the
existing test file has no diff. These checks cover all assertions, but the
ordinary local full suite still has the documented deadline/RPC limitation.
That startup fixture has no posterior rows or auxiliary certificates, so it
does not exercise the new acoustic recheck.

Local TypeScript suites use one worker and a 60-second default test deadline.
The insertion isolation test
caught and verified a fix for outer-array aliasing when recorded evidence was
replaced; insertion replay now retains its pre-certificate view.

All temporary installations and synthetic bundles stay in ignored
worktree-local directories. No scorer, lab tooling, supplied manifest,
test-data, evidence or experiment-card edits are included.

## Infrastructure block and unmeasured result

Three requests through the supplied supervisor `autolab.py modal-run` route
failed before executing any command with `NotFoundError: No such file or
directory`, each reporting zero compute expenditure. They included production
bundle/help-train inventory, the prescribed six-shard test harness, and even
`python3 -c "print(123, flush=True)"`. The returned results are retained in
ignored `.autolab/modal-result-{1,2,3}.json`. No alternate Modal route,
credentials, outside-worktree search or release-exam access was attempted.

Consequently production graph inspection, help-train calibration, the
training-only concentration envelope, and all four target robust/control
comparisons remain **unmeasured**. No practice score was requested; all three
attempts remain available. No target win count, score improvement or premise
kill is claimed. Runner failure supplies no evidence against the mechanism.

The ignored outcome records `blocked` without invented measurements.
`.autolab/model.json` selects the intended persistent `bundle` directory, which
has not been built. This deliberately blocks validation on the missing bundle
instead of allowing main-model-only decoding to appear to evaluate this change.
The declaration is pending and is not a claim of verified production assets.

## Limits and resumption

Tiny-model checks establish implementation mechanics, not production acoustic
behavior. Fit gains are derived from unchanged-stream activations; upstream
robust normalization can change downstream activation distributions. Production
scale restoration, M-estimator convergence, the concentration envelope and
runtime cost require real training-recording validation. A Gaussian consistency
correction is not an assumption that learned activations are Gaussian.

The current side path retains session audio and runs two additional complete
streaming decodes at completion. It can withdraw a warning after it was emitted;
immediate suppression, bounded session storage and acceptable SDK latency are
not established. Clean-session behavior, studio regressions, caught-mistake
losses and aggregate attribution are unknown.

Once the authorized runner works, inventory the actual complete baseline
bundle, then build `/scratch/listener-robust-biasnorm-recheck/bundle` using the
exporter's CLI and help-train training-only records. Verify both graphs and the
existing teacher assets, inspect target energy concentration against the frozen
training envelope, and measure positive-reading wins through ordinary free
tracker/checker discovery. The card's premise requires at least two target
windows beyond the envelope with improvement over gain alone. Under the user's
final rule, a disproved premise or fewer than three built target wins requires
reverting engine changes and recording a measured killed outcome. Three or
more target wins requires practice scoring before any narrowing; losses outside
the targets belong to the fold gate. Infrastructure blockage alone is not a
premise kill.
