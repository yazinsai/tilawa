# Glottal-source invariant phone readout

Status: needs-info. Implementation prepared; training, complete model export and full test-suite validation are blocked by the supplied supervisor. This is not a measured kill or a deployable experiment result. The card is unchanged.

## Prediction and decision rule

The card predicts two false warnings removed and one caught substitution lost: listener −1, overall −1. The premise is that excitation differences are mistaken for articulation differences in the frozen student's phone readout. Paired source changes should teach phone identity to survive those differences.

The task's final continuation rule takes precedence over the card's two-window kill threshold: fewer than three actual target warnings removed requires reverting the engine changes and recording a measured kill. At least three requires practice-score before deciding. None of the four target windows has been inspected, and recoveries are unmeasured. Infrastructure failure supplies no evidence against the acoustic premise. No killed outcome file has been written.

## Implementation

`modal/student_glottal_source.py` estimates conservative voicing and a text-independent pitch contour. It fits regularized autocorrelation LPC filters, inverse-filters the original waveform, and synthesizes paired views through the same filters. The changed excitation uses Rosenberg glottal-flow derivatives at two opening fractions, with two aspiration levels. Both views retain the estimated pitch phase and sample timeline. Unvoiced and uncertain boundary samples are copied directly. An actual inverse-filter/synthesis identity view must reconstruct the input within a fixed numerical tolerance.

The training cache contains four views per take: original, identity resynthesis, short-opening/low-aspiration, and long-opening/higher-aspiration. Both readouts receive exactly these same cached features and transcripts. The filters, pitch estimates, source settings, seeds and example order are shared; additional examples and synthesis artefacts are therefore present in the control as well.

`modal/student_glottal_export.py` exposes the deployed final linear head's acoustic input in a separate ONNX graph. For a dynamically quantized head this is the dequantized activation code, with the effective weight scales and zero points retained. Export verifies every original output and recurrent state over eight consecutive windows, and checks reconstruction of the primary logits. The frozen encoder is never fine-tuned.

The cache accepts only usable help-train `none` records explicitly marked `train`, without mistakes, repeats or exclusion flags. Explicitly excluded speakers and any speaker appearing in a nontraining record are excluded. The supplied help-train copy is documented to omit release-held-out speakers. Canonical reference transcripts supply supervision; no practice audio, labels, rivals or windows enter fitting or selection. Actual label schema and eligible coverage remain unverified because the data audit did not execute.

Both linear CTC readouts start from the deployed effective weights and biases and receive eight epochs with identical order, optimizer, learning rate and weight anchoring. The invariant arm adds Jensen-Shannon consistency between nonblank identity distributions pooled by soft transcript CTC occurrence occupancies. These occupancies are computed once per view from the frozen parent. Repeated phones remain distinct occurrences; this is neither raw frame matching nor hard word-boundary supervision. The control receives the same CTC and anchoring losses with consistency weight zero. A fixed two-value consistency grid is selected by speaker-balanced transcript CTC loss in three training-speaker folds. No practice-based selection is implemented. Cache and fitting share a persistent 23-hour deadline.

`packages/sdk/src/glottal-source.ts` discovers only otherwise confirmed string-path wrong-word comparisons. Both complete pronunciations, including any context requested by the checker, are scored on the exact existing frame window by complete-transcript CTC. A certificate requires a finite strict preference for the written pronunciation from the invariant readout and a finite strict preference for the rival from the matched control. Unknown phones, impossible completions, ties, malformed geometry and nonfinite scores abstain.

The core matches word identity, both complete strings and exact frame bounds. Graph confirmations are exempt. Internal issue confirmation and ownership stay intact; only the public wrong-word verdict is downgraded. Primary phonemes, primary posteriors and tracker input are retained. This avoids altering insertion merges or subsequent issue generation. Metadata survives token-preserving serialization. Audio sessions perform the auxiliary review at session end and publish occurrence-specific downgrades. This buffered implementation does not establish timely streaming vetoes, bounded session memory or mobile support.

The Node loader requires every declared side file to exist and match its hash, validates readout geometry and its primary-model identity, and checks the primary checksum. The bundle stage copies the original model directory and manifest verbatim, retaining existing teacher external data, adds only the declared side files, verifies their copied hashes, and repeats eight-window primary/extractor parity. Training caches are kept outside the deployment bundle.

## Validation and limits

Seven numerical Python tests pass using the local Python under Rosetta: identity reconstruction, duration and unvoiced preservation, deterministic distinct source views, retained pitch on known periodic input, conservative noise protection, exact enumerated repeated-phone occupancies, and speaker/label exclusion. The Torch consistency/gradient test is present but skipped locally because Torch is unavailable. These synthetic tests do not demonstrate that the estimated filter and voicing mask preserve all real consonants, especially on degraded audio.

Direct execution of the actual TypeScript checker passes supported string-candidate discovery, an exact-window strict reversal veto, rejection of changed bounds and tied control preferences, and graph-confirmation exemption. Direct SDK checks pass four exhaustive complete CTC path comparisons, strict paired reversal, independent recurrent state and the native 40 ms clock. These checks use Node's TypeScript transform and local import resolution; they are not substitutes for the repository test suites or TypeScript builds.

Direct execution of the actual audio-session review with mocked acoustics also passes final downgrades, stored certificates, and unchanged positions and insertion decisions. Unit tests additionally cover metadata transport, occurrence-specific retractions and declared missing/corrupt assets. The existing insertion-loader mock now returns ENOENT for absent optional files. These Vitest tests remain unexecuted. Python compilation, TypeScript source/test syntax checks and `git diff --check` pass.

Supervisor request 1 (read-only data/model/dependency audit), request 2 (the prescribed six-shard test harness) and request 3 (minimal `pwd`) all returned `NotFoundError: No such file or directory`, zero compute cost, and no shard output. Neither model features nor real training recordings were accessed. No readout was fitted and no complete bundle exists. No `.autolab/model.json` is declared: decoding with the main model would not evaluate this change. Practice scores used: zero of three. Overall, component, target, clean-session and studio results remain unmeasured.

## Resume after the supervisor is restored

First inspect the actual available model, training label schema and eligible speaker coverage, and run `bash tools/autolab_tests.sh` through the prescribed six-shard supervisor command. Run the Python numerical/Torch tests in a training image with NumPy, SciPy, Torch, torchaudio, ONNX and ONNX Runtime.

Use a training directory and a separate bundle directory under the experiment's authorized persistent scratch root. Prepare an explicit exclusions JSON from training metadata; release-held-out speakers must remain absent. Caching resumes completed takes after a supervisor timeout, checking waveform, transcript and primary-model hashes without resetting the one-day clock. The implemented entry points are:

```sh
python3 modal/student_glottal_export.py --model "$primary" --out "$training"
python3 modal/student_glottal_source.py cache --model "$primary" --labels "$training_labels" --reference packages/core/assets/ref-v2.json --excluded-speakers "$exclusions" --out "$training"
python3 modal/student_glottal_source.py fit --excluded-speakers "$exclusions" --out "$training"
python3 modal/student_glottal_source.py bundle --model "$primary" --excluded-speakers "$exclusions" --out "$training" --bundle-dir "$bundle"
```

Inspect warnings with the generic SDK CLI, which asserts unchanged tracker events and other error decisions:

```sh
pnpm --filter @tilawa/sdk exec tsx src/bench/glottal-source-cli.ts --model "$bundle" --audio "$audio" --evidence "$evidence" --out "$comparisons"
```

Freeze selection before inspecting the four target sessions. Count actual warnings removed after engine replay, rather than positive scores alone. If fewer than three targets are won, revert the engine/test changes and record the measured kill. Otherwise verify the complete bundle, declare its relative scratch path in untracked `.autolab/model.json`, and run the authorized practice-score before any narrowing or final decision.
