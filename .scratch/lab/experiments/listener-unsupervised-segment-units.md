# Listener: unsupervised segment units

Prediction: listener −1, overall −1. The hypothesis is that transcript-forced acoustic segmentation absorbs weak consonants into their neighbors, whereas acoustic segment units learned without text can retain those consonants. The card is unchanged. The four specified targets are:

- `lab-v1.5-p11/phone/warning:wrong_word/26:106:1/104.32`
- `lab-v1.5-p36/phone/warning:wrong_word/55:48:1/282.64`
- `lab-v1.5-p55/phone/warning:wrong_word/26:174:8/107.08`
- `realuser-v1-sob-049/phone/warning:wrong_word/92:13:1/36.44`

## Result

**Blocked by the supervisor execution environment; not measured.** Supervised Modal requests 1 (data/dependency discovery), 2 (a minimal `/lab/data` listing), and 3 (the required six-shard `bash tools/autolab_tests.sh`) all returned `NotFoundError: No such file or directory`, with no command output and zero compute charges. The error occurs before either training code or tests execute. The repair request is pending.

No help-train recording, target audio or target evidence was available locally. Eligible recording counts, speaker coverage, feature extraction parity on the shipped graph, model selection, and all four target preferences remain unmeasured. There have been **zero practice-score calls**. No target-win count or premise kill is asserted. No `.autolab/outcome.json` or `.autolab/model.json` has been declared: there is no trained and verified bundle to score. Scoring the main listener would not evaluate this implementation.

## Implementation

`modal/student_segment_units_export.py` creates a diagnostic copy of the deployed graph exposing the actual final encoder trajectory. For the int8 head it exposes dequantized activation codes, retaining the deployed activation quantization. The exporter checks the primary checksum and is written to require exact equality of every original output and recurrent state over eight consecutive streaming windows before it writes the extractor metadata. This check has not yet run against the shipped model.

`packages/sdk/src/bench/segment-units-cli.ts` caches trajectories using the SDK fbank, streaming clock, padding and fresh recurrent state. The training reader includes all usable train-speaker takes, including mistakes and repeats; it excludes explicit recording exclusions and every speaker with any non-train split. It separately identifies none-labelled takes without planted events or repeats for pronunciation supervision. The independent fitting function receives only acoustic arrays, never transcripts, scripts or engine visits.

`modal/student_segment_units.py` uses eight whitened acoustic projection coordinates and three equally spaced interpolated samples in normalized segment time. Both codebooks have 32 units, the same 24-dimensional descriptors, one-to-twelve-frame segments and the same fixed segment penalty. The independent arm alternates dynamic-programming segmentation and weighted clustering. The control clusters segments from canonical-transcript CTC-forced phone/blank placement, splitting long intervals at the same duration bound. Both use all eligible acoustic recordings. Only none takes fit the phone-to-unit transducers: initialization uses soft transcript occupancies, followed by forward/backward EM over phone correspondence and soft unit assignments. A deterministic speaker-grouped training partition chooses between two Dirichlet regularizers using canonical sequence likelihood; complexity and selected regularization are frozen before refitting on all eligible training speakers.

The SDK compares complete written and already-accused pronunciations over the exact existing checker windows. It marginalizes acoustic segment lengths, unit identity and monotone phone correspondence, including blank separation of repeated phones. An acoustic-background completion prior removes path-count rewards while allowing the codebook to influence boundary probabilities. A certificate requires strict independent preference for the written sequence and strict forced-control preference for the rival; ties, nonfinite results and phones without canonical training coverage abstain.

The checker retains baseline internal confirmation and ownership state and changes only the public verdict for an exact certified string-path comparison. Graph confirmations remain exempt. Session completion carries certificates in phoneme metadata, replays the same evidence, and downgrades only matching live occurrences. Primary tokens, posterior rows, tracker events and insertion decisions are preserved. Certificates are currently computed at session completion; this route can downgrade an earlier public confirmation rather than preventing its initial display.

The Node loader requires a complete hashed side asset tied to the unchanged primary graph. The CLI's bundle mode copies the primary model and manifest unchanged, verifies and includes the already shipped insertion teacher and external data, declares both scorer files only after copying them, and loads the result before reporting success. No real bundle has been produced.

## Validation and limits

- Both package TypeScript builds passed.
- Five Python unit tests passed, covering duration normalization, text-free boundary learning, marginalized correspondence, forced-control boundaries, and retaining multi-phone supervision within a single acoustic unit. Unit durations are expanded for transducer correspondence without changing acoustic boundaries. The local Python installation required its x86 mode to load its installed NumPy extension; no real-data training ran locally.
- Focused core certificate tests passed: discovery preserves baseline decisions, only finite reversals veto, changed windows/spellings cannot reuse certificates, internal issue creation is retained, and graph confirmations remain unchanged.
- Thirteen focused SDK tests passed for the scorer, certificate session route, missing/corrupt assets, existing insertion-loader behavior, matched capacity, normalized distributions, unchanged serialized primary evidence, and Python/SDK score parity. The final coverage guard is separately checked by the scorer tests.
- Initial complete local suites encountered test and worker-RPC timeouts under default parallel load. The side-asset loader also exposed a broad existing filesystem mock; that mock now reports the undeclared side asset as absent and its focused tests pass. A one-worker core rerun passed 28 files without a reported failure before it was stopped: complete local verification was too costly for the Mac. Full-suite verification remains pending on the authorized Modal service. No full-suite pass is claimed.

Synthetic tests establish arithmetic and transport behavior, not acoustic discrimination or release safety. The unavailable data must still establish usable coverage of every target pronunciation and fidelity of forced alignment on mistake-labelled recordings. The implementation must not silently drop such recordings to make the control fit. Speaker exclusions and actual cache completeness also need auditing against the supplied labels. No listening judgment or independent audio opinion was obtained or inferred from model output.

## Resume after execution is repaired

1. Repeat data/dependency discovery through the authorized supervisor service. Export the diagnostic graph, audit its parity check, then cache all eligible help-train training recordings into persistent experiment scratch storage. Verify the shard counts against the complete eligible label list.
2. Fit both codebooks and transducers using only the frozen training partition procedure. Retain the model and audit, including every eligible recording and the selected regularizer.
3. Run the generic SDK diagnostic CLI on the supplied WAV and evidence for each target session. Record the complete checker-requested strings, exact shared frame window and four sequence scores; verify the corresponding warning is removed and tracker/insertion results stay identical.
4. Apply the user's target rule: fewer than three recovered targets kills the experiment; three or more requires a verified complete bundle, `.autolab/model.json`, and an official practice score. Other losses are for the fold gate to trade off. No constants may be selected from target or practice scores.

The committed branch is a resumable implementation, **not ready for supervisor scoring or a fold decision** until its real assets and measurements exist.
