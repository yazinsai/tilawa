# Acoustic decision radius

Status: implementation draft; blocked before calibration and Modal validation.
The experiment card is unchanged. No acoustic target result or practice score exists.

## Prediction and mechanism

The card predicts two false wrong-word warnings removed and one caught mistake
lost: listener −1, overall −1. Its initial premise asks for two target radii inside
the training-frozen rejection region. The task's continuation rule is stricter:
fewer than three built target vetoes requires rollback and a killed result; three
or more requires practice scoring before deciding. Neither condition has been
measured.

The SDK discovers existing non-graph wrong-word confirmations through a read-only
checker callback. It captures the actual complete written and accused hypotheses,
the original acoustic frame interval, the original margin and the existing
confirmation bar. Primary tokens, their confidence and timing, and tracker
positions never depend on a perturbed view. The callback runs at final settlement
or when certainty is frozen, rather than changing candidate generation.

The additive noise basis uses actual waveform patches from completed preceding
token-free pauses, with an analysis-frame collar away from speech. It requires
enough nonzero patches and a pause power below preceding speech's median power.
DC removal and Gram–Schmidt produce four waveform columns. They have unit
mean-square and are mutually orthogonal, so squared coefficient norm is added
energy relative to the observed pause energy over the disputed duration.
Perturbations add those columns to the original samples over the original support
interval; they replace no speech or context and change no sample count.

The side inference first replays the original PCM on the native streaming grid
and saves the model state before the earliest input window intersecting the
perturbation. Every optimization evaluation resumes a fresh copy of that state.
Only overlapping feature frames are recomputed. The original replay must reproduce
the checker margin within 1e−6 nats; otherwise it abstains. The recheck uses the
same top-k, probability rounding, posterior expansion and CTC scorer as the
original comparison, including its complete neighbouring text where applicable.

Deterministic central finite differences seek a direction reducing the existing
confirmation margin. Projected descent searches up to four times measured pause
energy; each observed crossing is bisected along its witnessed direction. The
fixed initial protocol is four dimensions, eight iterations, 0.05 coefficient
differences and twelve bisection steps. These values have not been tuned on
practice. An observed crossing is an **upper estimate** of the minimum energy;
the search does not prove global optimality. Failure to find a crossing is
censored and supplies no veto or robustness guarantee. This searches for a
boundary rather than counting agreement under random perturbations.

## Training-only calibration and asset path

`modal/student_radius_extract.ts` mines engine-raised accusations only from
explicit `train` rows in `help-train`. It excludes supplied release-held-out
speakers and the union of all speakers with any nontraining label metadata.
Script-overlapping confirmed wrong words are weak positive controls. Only
none-labelled recordings without events or repeats supply negative accusations;
other scripts are not treated as verified clean transcriptions. No script is
used as a performed frame transcript. The miner accepts only a `help-train`
directory and can run in independent supervisor shards.

`modal/student_radius_calibrate.py` requires every extraction shard, consistent
primary weights and search protocol, measured true substitutions from at least
two speakers, and measured false accusations. Each leave-one-speaker-out fit
chooses the largest cutoff preserving its training positives. The final region
uses the minimum of the fold cuts and held-out positive radii. A calibration
that rejects no training false accusation fails rather than creating an
uncalibrated cutoff. Fold losses, false vetoes, censored searches and abstentions
are recorded separately. These are training diagnostics, never lab scores.

The normal Node directory/tier loader enables the feature only with an installed
`acoustic-radius.json` matching the primary SHA-256 and inventory. Invalid rules
fail loading and release the backend. Successful measurements inside its frozen
rejection region travel as ordinary token metadata. The checker validates the
witness and matches the exact word occurrence, complete hypotheses, frame
interval and existing bar. It can only turn that string-path confirmation into
uncertain. Graph comparisons and insertion decisions are outside this interface.
Audio sessions publish an end-of-session uncertainty update only when the entire
original event matches the replayed baseline occurrence; evidence-only replay
consumes the same certificate directly.

No calibration file, model bundle or `.autolab/model.json` has been declared.
The implementation is consequently disabled on ordinary model bundles. Running
the main model now would not evaluate this experiment. A complete future bundle
must retain the primary ONNX, its unchanged manifest and every existing teacher
asset, and add the verified training-frozen calibration before model selection.
`modal/student_radius_probe.ts` measures all discovered confirmations in a
supplied recording without taking labels or target locations; report selection
can then check the registered incidents after calibration is frozen.

## Validation and limits

Supervisor requests 1 and 2 attempted read-only help-train/model inspection.
Request 3 attempted the supplied six-shard `tools/autolab_tests.sh` command.
All three returned **`NotFoundError: No such file or directory` before executing
their commands**. There are no command logs, training observations or remote
test results. This is an infrastructure block, not evidence against the acoustic
premise. It does not justify a killed outcome or a fabricated zero target gain.

Local Python unittest validation passed three checks: speaker-grouped cutoff
freezing, excluded speakers/missing shards, and refusal to calibrate failed
searches. A dependency-free Node harness executed the actual TypeScript modules
and the new focused tests using Node's TypeScript transform. Ten checks passed:
the analytic nearest-energy direction and determinism; failed-search abstention;
preceding-pause-only basis and its energy identity; equality of local and full
feature recomputation; JSON/native-cell metadata transport; native stateful replay
and baseline mismatch detection; event-preserving discovery/final veto; exact
certificate identity and validity; graph preservation; insertion preservation.
This harness is an ignored diagnostic, **not Vitest or a TypeScript typecheck**.
The committed Node loader test also covers matching/mismatched primary weights;
it still needs the normal test runner. `git diff --check` passes.

Remaining required work is help-train inspection/extraction and calibration,
full build/core/SDK tests on Modal, registered target measurements, and a complete
asset bundle if the premise survives. All three practice scores remain unused.
If fewer than three targets are actually vetoed, revert the engine draft and
record the measured killed result. Otherwise run the prescribed practice score
and let its gate price losses; do not tune against target radii or practice.

Even with successful calibration, sparse scripted positives and a low-dimensional
noise basis do not establish safety on unseen learners. A genuine weak swap can
be fragile. Pause bounds derive from primary emissions and waveform power rather
than an independent human speech annotation, so a quiet breath or unrecognized
speech could contaminate a patch. Exact replay rejects gap-redecoded windows that
raw PCM cannot reproduce. The draft buffers session audio, performs repeated CPU
inference and settles vetoes at EOF; it is not an immediate streaming warning
solution or a demonstrated device-budget implementation. No scorer, lab tooling,
supplied manifest, test data, evidence or experiment-card edits were made, and no
fold was performed.
