# Gesture path signatures

Status: implementation preserved; training and validation blocked by the
supervisor's Modal infrastructure. The acoustic premise has not been tested.
No practice score was requested and no fold was performed. The experiment card
is unchanged.

## Prediction

Variable within-word pacing may distort acoustic comparisons indexed by time
while preserving articulation order. Predict two false warnings removed and one
caught substitution lost: listener −1, overall −1. The card lists four target
windows. The task's continuation rule requires at least three actual target wins;
the card's two-reversal criterion alone does not authorize scoring.

## Implemented mechanism

`modal/student_gesture.py` fits compact pronunciation-conditioned Gaussian
distributions from none-labelled help-train training-speaker recordings only.
It rejects scripted-mistake transcripts, non-training splits, explicitly
excluded speakers, and invalid SDK observations. The utility requires an
explicit exclusion list. Three deterministic speaker folds choose two or three
spectral coordinates, signature order two or three, and shared regularization.
Selection uses speaker-balanced predictive pronunciation-contrast loss rather
than comparing continuous densities across different dimensions. Neither
selection nor training accepts practice incidents.

The SDK uses its existing fbank frontend and native 40 ms observation clock.
Fixed pooling of the 80 mel bands produces spectral coordinates, normalized by
training-only means and scales. Chen's product of segment exponentials computes
truncated iterated integrals of their piecewise-linear trajectory, retaining
ordered cross-band changes without a time coordinate. An initial spectral
coordinate retains articulation level. The control resamples the same trajectory
at uniformly spaced traversal times, with exactly the same feature dimension and
number of Gaussian parameters.

Both arms marginalize all positive-duration phone segments and optional silence
segments in the same bounded semi-Markov lattice, requiring every pronunciation
phone. They share a normalized completion-conditioned timing prior. Segment
summaries are invariant to monotone speed changes along the same geometric path;
the sampled frontend and boundary lattice are finite approximations, so full
waveform or sequence-score invariance is not established.

The checker exposes the complete written and already-accused pronunciations and
the exact acoustic window used by an existing string-path confirmation. The
side scorer compares both pronunciations on identical observations. Only a
strict, finite control-rival/signature-written preference reversal can produce a
veto. Certificates match the exact pronunciation pair, location and frame bounds
and survive token-preserving serialization. Primary tokens, posterior rows,
tracker events, graph confirmations and internal insertion ownership remain
unchanged; the public wrong-word confirmation becomes uncertain. The SDK also
supports the same comparison during audio-backed streaming sessions.

The Node loader supports an optional checksummed `gesture-asset.json` descriptor
and `gesture-emissions.json` beside the primary model. An installed descriptor
with absent or corrupt data fails loading. No asset was generated, no descriptor
was installed and `.autolab/model.json` was not created: the required complete
bundle does not yet exist. Running the primary model would not evaluate this
experiment.

## Validation and blocking evidence

Three requests used only the supervisor interface authorized by the task:

1. Modal run 1: read-only audit of LAB_DATA and help-train eligibility.
2. Modal run 2: minimal working-directory/environment probe.
3. Modal run 3: the prescribed six-shard `bash tools/autolab_tests.sh`.

Each returned `NotFoundError: No such file or directory`, with zero reported
compute cost and no command log or shard output. The error does not identify
which infrastructure path is absent. No dataset audit, exclusion audit, model
training, target comparison, build or runtime test result was returned.

Python source compilation and `git diff --check` passed locally. A tiny local
Python unit-test attempt could not import NumPy: its installed x86_64 extension
is incompatible with the arm64 interpreter. This is an environment failure,
not a passing or failing acoustic result. No heavy computation ran on the Mac.

Added unit tests cover ordered cross-band integrals, collinear speed changes,
closed-loop signed area, stationary trajectories, equal control dimensionality,
exhaustive tiny boundary lattices, native frontend chunking, invalid model data,
training exclusions, exact certificate scope, public verdict-only vetoes and
graph preservation. They remain unexecuted because of the infrastructure block.

Target windows tested: **0**. Target wins, missed-mistake losses, clean-session
behavior, studio behavior and overall score change are **unknown**. All three
practice-score opportunities remain available. This is not a premise kill or a
measured zero gain.

To resume, restore the supervisor's Modal runner, execute the added tests, audit
the available training labels and exclusion list, export SDK observations for
eligible none-labelled takes, then run speaker-grouped training and save its
report and checksummed side assets under the experiment's persistent scratch
directory. Copy the unchanged primary bundle and required existing auxiliaries
alongside those assets. Verify loading, certificate transport and primary output
preservation before testing the actual engine-requested target windows. Apply
the task's target-win stopping rule; only a surviving implementation proceeds
to practice scoring and the fold gate.
