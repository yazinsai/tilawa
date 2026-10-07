# 09: SDK runtimes and session API

**What to build:** The shippable SDK. The listener runs on onnxruntime-web, onnxruntime-node and onnxruntime-react-native; the session layer sits on the core event streams; there's a web sample for testing on a physical device.

**Blocked by:** 04, 08.

**Status:** ready-for-human

- [x] Microphone capture, resampling and VAD into streaming listener inference on each runtime. WebGPU falls back to WASM SIMD on web.
- [x] Model tiers downloaded on first use, versioned separately, cached, within the 200 MB budget.
- [x] `startMemorization({ surah, fromAyah, toAyah })` and `startFreeRecitation()` with mode and policy.
- [x] `unsupported` reported per class from the shipped model's metadata.
- [x] Web sample app for the user's own device testing.
- [x] Docs credit tlog and QUD (CC-BY).

## Comments

**Agent, 2026-10-07.** Branch `ticket-09-sdk`.

- **Export** (`modal/student_export.py`): spliced-clean as icefall's streaming CTC Zipformer2 graph, fp32, 257.7 MB, chunk 16 / left 128 (45 fbank frames in, advance 32, 8 CTC frames out). Python ORT streaming vs the PyTorch chunk-masked forward on 6 help clips: PER 0.0, frame top-1 agreement ≥ 99.7% (`docs/benchmarks/onnx-parity-spliced-clean-v1.json`).
- **SDK** (`packages/sdk`): TS fbank (matches torchaudio kaldi fbank), streaming listener over one `InferenceBackend` interface; `@tilawa/sdk/node` (onnxruntime-node), `/web` (onnxruntime-web, WebGPU then WASM SIMD, Cache API, AudioWorklet mic, resampler, energy VAD), `/react-native` (onnxruntime-react-native; app supplies storage and mic PCM). Model registry `src/models-v1.json`: versioned tiers, download on first use, length + SHA-256 check, 200 MB budget enforced. Session API `createEngine` → `startMemorization({ surah, fromAyah, toAyah })` (declared passage into `CheckerOptions.passage`) / `startFreeRecitation()`, policy `countSelfCorrections` (Decision 5, default off), `confirm`, `minConfidence`, `evidenceMs`; first event is `capabilities` with the 11 pronunciation classes `unsupported`.
- **End to end** (`pnpm bench:help`, `test/help-e2e.test.ts` gated on `TILAWA_MODEL_DIR`/`HELP_DIR`): 388 help eval WAVs → session API → onnxruntime-node reproduce the scoreboard exactly: declared 83.6% / 87.5% (56/11/8), free 81.8% / 42.2% (27/6/37). 381/388 decodes are identical to the PyTorch evidence. Node RTF 0.046 (1 thread, M-series). Scoreboard row `spliced-clean-onnx`.
- **Finding (checker):** the free-recitation checker depends on evidence cadence. Pushing every 320 ms listener step instead of 1 s chunks gives 27/10/37 (73.0% precision); the declared arm is unchanged. The session batches to 1 s by default (`evidenceMs`), which costs word-highlight latency against the 600 ms bar. Worth a checker ticket: results should not depend on chunking.
- **Web:** Chrome desktop, WASM, 1 thread: RTF 0.092, push p95 35 ms; the sample confirms the labelled wrong word on a help clip. WebGPU fails in ORT's JSEP `Expand` kernel during warmup and falls back to WASM; the reason is shown in the sample.
- **Not done / for ticket 10:** fp32 is over the 200 MB budget (loads only with `allowOverBudget`); there is no `lite` tier; React Native is typechecked only, not run on a device; phone runs of the sample are still to do.
