# Consonant-event CTC readout

Status: blocked by the supervisor execution environment; implementation prepared,
training and target measurements not run. The experiment card is unchanged.
This is not an experimental kill and is not ready for validation with a new ear.

## Prediction and mechanism

The card predicts listener −1 and overall −1: weak consonant identity may be
obscured by supervision for vowels and recitation length. The task's stopping
rule requires at least three target wins before a practice score, superseding
the card's less restrictive criterion. No target win count is established.

`modal/student_consonant_event_export.py` exposes the shipped encoder's actual
final linear input in a separate ONNX copy. For an int8 graph this includes
dequantized activation codes, rather than substituting a float parent. Export
checks equality of all primary outputs and caches over eight sequential windows,
as well as feature/logit geometry. These checks are implemented but have not run.

The SDK training utility caches complete recordings through the existing fbank
and streaming geometry, with separate encoder states. Only usable help-train
none-labelled training-speaker takes without edits or repeats are eligible;
non-training and explicitly excluded speakers are rejected. Canonical text
supplies transcripts, never engine word timings. No practice audio, warning
windows, planted labels, teacher outputs or release speakers enter fitting.

The consonant transcript removes vowels and duration marks. Generated shadda
and ghunna runs become one articulation, with nasal variants mapped to their
consonant identity. Vowel-separated and word-separated repeated articulations
remain separate events. The full-inventory control retains the original
expanded transcript on the identical recordings and frozen features.

Both CTC heads have the same 64-unit tanh architecture, full-sized output
matrix, initialization, parameter count, optimizer and bounded update budget
(maximum 120 epochs). In the consonant arm, non-consonant output slots receive
no positive transcript labels; they remain negative softmax classes to match
capacity exactly. This avoids comparing heads with different parameter counts.
The ordinary CTC objective marginalizes emission times. A deterministic
speaker-grouped training-only partition selects each checkpoint by validation
CTC loss. Feature normalization uses fitting speakers only. Reports retain
partition membership, selected epochs and the observation-file checksum.

At inference, the checker exposes only already-supported string-path wrong-word
comparisons. The SDK scores their complete written and already-accused
hypotheses over exactly the same existing frames with both readouts. It
abstains when consonant transcripts are equal, incomplete or impossible. A
certificate requires a finite strict preference for written consonants while
the full-inventory control strictly prefers the rival. Certificates carry
complete hypotheses, pronunciation pieces, word identity and exact frames;
the checker revalidates these before retracting the public verdict.

Primary token identities, confidences, posteriors and tracker events are not
modified. Graph confirmations remain exempt. The checker's internal confirmed
ownership remains intact for later merges. Audio sessions replay the original
live evidence to retract only the exact occurrence removed by a certificate;
auxiliary insertion decisions retain their existing route. This experimental
path buffers audio and completes rechecking at session end, so immediate
streaming warning latency and device resource use remain unvalidated.

The normal Node model loader requires a complete, hashed side descriptor tied
to the primary model checksum. Export preserves the original primary/auxiliary
directory and adds the extractor and paired heads. A training-only feature
parity fixture checks the exported JavaScript readout against PyTorch outputs.
No model bundle exists yet and `.autolab/model.json` was not created. Running
the main listener cannot evaluate this experiment.

## Validation and limits

Four supervisor requests failed before command execution with
`NotFoundError: No such file or directory`. Each result reports $0 compute:

| Request | Purpose | Result |
| --- | --- | --- |
| 1 | Inspect allowed data, model manifest and scratch storage | Pre-execution failure |
| 2 | Minimal `pwd` to isolate command/data dependencies | Same failure |
| 3 | Six-shard `bash tools/autolab_tests.sh` | Same failure; tests did not run |
| 4 | Data/dependency inspection using the supervisor's CPU training image | Same failure |

The required environment repair was requested. No alternate Modal credentials,
external datasets or release benchmark access were sought.

Local lightweight source checks passed: Python syntax; articulation compilation;
CTC forward scores against exhaustive paths including repeated identities;
positive paired reversal and vowel-only abstention. A synthetic checker source
check verifies complete hypothesis pieces, exact-window retraction and stale
window rejection. These checks use no acoustic model or practice cases and are
not a substitute for TypeScript builds, the required core/SDK test suites,
training, model parity, target replay or practice scoring.

Unit tests were added for those behaviors, graph preservation, public-verdict
ownership, audio-session certificate transport and missing/corrupt side assets.
Their Vitest execution is pending. No practice score was requested; all three
permitted scores remain available. The three registered target windows have
not been read or scored by the built readouts. The premise and kill criterion
therefore remain unresolved, and no fold or discard was performed.

## Resume after execution is restored

1. Inspect `LAB_DATA`, dependencies and available training labels through the
   supervisor. Run the supplied six-shard test harness on the current files.
2. Export the frozen extractor into the experiment's persistent scratch
   directory, using `student_consonant_event_export.py --model ... --out ...`.
3. Cache eligible recordings with the SDK's `consonant-event-cli.ts --model ...
   --extractor ... --labels ... --excluded-speakers ... --out ...`. Fit with
   `student_consonant_event.py --observations ... --metadata ... --out ...`.
4. Package the selected heads with the exporter `--bundle ... --fitted ...`;
   verify readout parity through the CLI's `--verify ... --fitted ...` route.
5. Replay the registered sessions using the CLI's generic `--audio ...
   --evidence ... --fitted ...` route. It reports comparisons and before/after
   warnings and checks tracker and other error-class preservation. Diagnostic
   case selection stays outside engine code.
6. If fewer than three targets are won or the premise is wrong, revert engine
   changes and record the measured kill. Otherwise declare the verified bundle
   with untracked `.autolab/model.json` and run the required practice score
   before deciding whether any narrowing is warranted.

No scorer, lab tooling, supplied manifests, evidence, test data or experiment
card edits were made. The prepared implementation is retained for resumption;
there is no trained result to fold or experimental evidence supporting a kill.
