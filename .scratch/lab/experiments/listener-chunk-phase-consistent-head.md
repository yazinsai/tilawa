# Chunk-phase-consistent phone head

Status: implemented training recipe; execution blocked by the supervisor before
workers started. No trained bundle, target measurement, or practice score exists.
The experiment card is unchanged. This is not a premise kill.

## Prediction and mechanism

Predict one recovered addition with no counted losses: listener −1, overall −1.
The proposed failure is that the same articulation has different frozen encoder
representations depending on its position within a streaming chunk. Selecting
between unchanged predictions does not train the head to reconcile those views.

`modal/student_phase_head.py` exposes the activation immediately before the
shipped exported linear phone head. The int8 encoder and its carried states remain
unchanged. It caches each full authorized recording on two grids: the ordinary
origin and an origin half a chunk earlier, using leading silence padding. For the
existing 32-feature-frame shift these differ by 16 fbank rows, or four output
frames. Those synthetic leading outputs are discarded. Original fbank rows,
samples, transcript order, and output timestamps are retained.

`modal/student_phase_consistency.py` computes exact transcript-conditioned CTC
forward/backward occupancies. Each phone occurrence pools the full phone identity
distribution over its possible emission times, independently in each view.
Symmetric Jensen–Shannon divergence compares these pooled distributions. This is
not framewise agreement: a phone may occur at different times. Repeated identities
retain distinct occurrence slots. The alignment weights are detached; transcript
CTC learns timing while consistency updates the readout rather than shifting the
latent weights to evade the penalty.

Both arms start from the same dequantized shipped head and see exactly the same
two cached views, recording order, optimizer settings, and update count. Both use
transcript CTC and KL preservation of the shipped output over the whole recording,
including real pauses. Only the consistency term differs. A common checkpoint
epoch minimizes mean validation CTC plus preservation across both arms. The 80/20
speaker split is deterministic and frozen before training; selection never reads
practice incidents. Cache and refit share a maximum one-day wall-clock budget.

The loader accepts help-train training speakers' none-labelled recordings and an
optional authorized catalog of performed Quran transcripts. Mistake/repeat takes,
excluded recordings, unsupported labels, and any speaker with an eval-tagged take
are rejected. Canonical within-ayah character runs are collapsed using the shipped
training convention for held sounds; separate ayahs retain their boundaries.
Full recordings preserve ordinary speech and pauses. No new augmentation is
introduced. The actual supplied corpus and its speaker coverage remain uninspected
because the worker never started.

Export appends the selected float32 linear readout to the frozen streaming graph,
retains its inventory and clock, copies the required unchanged teacher assets, and
updates the new bundle's checksum. It checks PyTorch/ONNX head parity and exact
recurrent-state parity across multiple windows before declaring the bundle ready.
No SDK, tracker, checker, insertion-verification, scorer, or input-data rule changes.

## Validation and limits

Unit tests cover legal origins and timestamp trimming, speaker grouping and
training eligibility, exhaustive CTC paths with equal and distinct consecutive
phones, timing-invariant identity pooling, nonzero consistency gradients, identical
control/experimental CTC and preservation, common checkpoint selection, and
extraction/export of a generated dynamic-int8 head while preserving every frozen
initializer and interface field.

The supervisor rejected all three authorized requests with
`NotFoundError: No such file or directory`, before returning worker logs:

- Request 1: inspect the supplied training data and primary model.
- Request 2: retry inspection and run the numerical unit tests.
- Request 3: the required six-shard `bash tools/autolab_tests.sh` run.

The result files report zero compute spend. No alternative Modal access was
attempted. Local syntax checks passed. All 12 lightweight numerical/geometry unit
tests passed using PyTorch 2.8.0, ONNX 1.19.0, and ONNX Runtime 1.22.1 in a worktree
virtual environment. The export test uses a generated tiny graph, not the supplied
production model; it does not establish that the production graph can be patched
or that its teacher assets are available. The required core/SDK suites remain
blocked by the supervisor.

Neither target consonant recovery under both origins relative to the matched
control nor the three-target prerequisite for practice scoring can be assessed.
Zero of the three allowed practice scores have been used. There is no
`.autolab/model.json`: scoring the ordinary model would not evaluate this refit.
No fold or discard was performed, and no killed outcome was fabricated from an
infrastructure error. Resume with worker input inspection, authorized-data caching,
matched refitting, complete bundle verification, then target diagnostics. If those
diagnostics establish the specified premise/target kill, revert the engine code,
retain the measured write-up, and record the killed outcome as instructed.
