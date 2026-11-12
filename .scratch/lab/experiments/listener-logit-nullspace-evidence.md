# Final phone-projection null-space evidence

Status: needs-human. Implementation drafted; acoustic validation blocked by the supplied supervisor. This is not a measured kill or fold result. The experiment card is unchanged.

## Prediction

Two false warnings removed and one caught substitution lost: listener −1, overall −1. The hypothesis is that the final phone projection discards useful acoustic distinctions present in its input features. A certificate requires the null-space scorer to prefer the complete written pronunciation while a matched primary-logit scorer prefers the already accused pronunciation.

## Implemented mechanism

`modal/student_logit_nullspace_export.py` identifies the frozen deployed final MatMul, Gemm or dynamically quantized integer phone head in a separate ONNX copy. For the integer head it exposes dequantized activation codes, not pre-quantizer activations, and includes weight scales and zero points in the effective matrix. Thus the proposed coordinates are invisible to the effective linear projection, including its blank logit. An SVD records the complete row and null bases and verifies their reconstruction. A fixed-seed orthogonal projection retains as many null coordinates as there are primary logits, matching the capacity of the full-logit control. This samples the unused space; it does not exhaust every possible null-space representation.

Export must verify eight consecutive windows against the unchanged primary graph, using the SDK's enabled CPU graph optimizations: every primary output and recurrent state must remain exactly equal. It also checks effective affine-head reconstruction. These checks are implemented but have not run on the supplied graph.

The generic preparation CLI accepts only usable `help-train` training-speaker `none` takes without labelled mistakes or repeats. It rejects excluded recordings and explicitly excluded speakers, and excludes any speaker appearing in a nontraining label. The trainer independently checks eligibility. The supplied isolated training copy excludes release-held-out speakers by construction; neither practice recordings nor practice labels enter fitting or selection.

Both arms fit full-Gaussian phone emissions with identical dimensions, four EM updates, and a shared regularization value selected from a fixed grid by three speaker-grouped training folds. Selection averages conditional phone-identity validation loss across both arms, weighting validation speakers equally. Parent posteriors initialize soft transcript occupancies; subsequent updates marginalize boundaries under each arm's fitted likelihood. Legal CTC timing arcs are normalized, and scoring conditions their prior on pronunciation completion to remove a path-count reward. No phone may be skipped; equal successive labels require blank separation.

Training-only preservation controls use actual `none` audio as the spoken rival against counterfactual complete written references. Up to three lexical single-phone neighbours per recording are chosen deterministically from the generic reference lexicon. Validation reports how often a null/control reversal would veto that spoken pronunciation. These controls do not select regularization, do not synthesize audio, and do not establish preservation on verified scripted substitutions.

The SDK discovers the checker's otherwise supported string-path wrong-word comparisons and uses their exact existing frame windows and complete written/rival strings. Both scorers use identical observations and timing topology. Only finite strict opposing preferences issue a certificate; unobserved training phones abstain. Core matches both strings, the word identity and exact frame bounds. Graph confirmations are exempt. Internal confirmation and ownership remain unchanged, so the public wrong-word verdict can become uncertain without changing primary tokens, posteriors, tracker input or insertion merges.

The Node loader requires the two declared side files to exist and match their hashes, verifies the unchanged primary checksum, validates orthonormal null coordinates and their annihilation of the effective matrix, and checks Gaussian geometry. Audio sessions retain audio for an end-of-session auxiliary review, retain certificates in recorded evidence, and publish explicit downgrades of the exact live occurrences removed by identical-evidence replay. This buffered route does not establish timely streaming warnings, mobile support, bounded memory or device latency.

## Validation so far

Python source compilation, Node type-stripping syntax checks of all twelve changed/new TypeScript source/test files, and `git diff --check` pass. These are syntax checks, not TypeScript builds or test execution. Local numerical tests could not import the installed NumPy binary because its architecture differs from the local Python architecture. No numerical assertion ran locally.

Tests added cover SVD reconstruction/null annihilation, quantized scales and zero points, exhaustive complete-path likelihood and occupancy marginalization, training eligibility, strict opposing preferences, exact certificate matching, graph exemption, native acoustic clock and independent cache state, unchanged primary evidence, occurrence-specific retractions, session-end downgrades, and missing/corrupt assets. The existing insertion-loader mock now returns ENOENT for absent optional files rather than a manifest for every read. These TypeScript and ONNX tests remain unverified here.

Supervisor requests 2 and 3 for the prescribed six-shard test harness, request 4 for a minimal explicit-node-image `pwd` command, and request 5 for a minimal CPU training-image data-availability check all returned `NotFoundError: No such file or directory`, zero compute cost, and no shard output. The initial read-only model/data audit request 1 is still pending, without output. The user has been asked to restore the supplied execution environment. No deployed head, eligible training recording, or target acoustic window has been inspected. No scorer has been fitted and no bundle has been verified.

Target recoveries and overall/component changes are unmeasured. Practice scores used: zero of three. No `.autolab/model.json` is declared; a main-model score would not evaluate this experiment. No scorer, lab tooling, supplied manifest, evidence, test data or experiment card was edited.

## Continuation

Restore the supplied supervisor, inspect the actual graph and training labels, run the numerical and TypeScript tests, export/verify the extractor, prepare eligible recordings, fit/freeze the two arms, and verify a complete bundle retaining the unmodified primary model, manifest and existing insertion-teacher files. The added files are `logit-nullspace.onnx`, `logit-nullspace-emissions.json`, and `logit-nullspace-asset.json`; the metadata and training report document the fit.

Export entry point: `python3 modal/student_logit_nullspace_export.py --model <primary-dir> --out <bundle-dir>`. Prepare: `pnpm --filter @tilawa/sdk exec tsx src/bench/logit-nullspace-cli.ts --model <primary-dir> --extractor <bundle-dir> --labels <help-train-labels> --excluded-speakers <exclusions.json> --out <observations.jsonl>`. Fit: `python3 modal/student_logit_nullspace.py --input <observations.jsonl> --metadata <bundle-dir>/logit-nullspace-metadata.json --excluded-speakers <exclusions.json> --out <bundle-dir>`.

Inspect actual engine-requested windows with the same CLI using `--model <complete-bundle> --audio <wav> --evidence <evidence.json> --out <comparisons.json>`. It asserts tracker and insertion equality. It contains no target/session list. Count actual target warnings removed after replay, not just favourable numerical comparisons. The task's continuation rule requires at least three target recoveries; fewer than three requires reverting engine changes and recording a measured kill. At least three requires practice-score before deciding. The infrastructure failure supplies no evidence for an acoustic kill.
