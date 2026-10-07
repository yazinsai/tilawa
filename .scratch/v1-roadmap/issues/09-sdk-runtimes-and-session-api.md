# 09: SDK runtimes and session API

**What to build:** The shippable SDK. The listener runs on onnxruntime-web, onnxruntime-node and onnxruntime-react-native; the session layer sits on the core event streams; there's a web sample for testing on a physical device.

**Blocked by:** 04, 08.

**Status:** claimed

- [ ] Microphone capture, resampling and VAD into streaming listener inference on each runtime. WebGPU falls back to WASM SIMD on web.
- [ ] Model tiers downloaded on first use, versioned separately, cached, within the 200 MB budget.
- [ ] `startMemorization({ surah, fromAyah, toAyah })` and `startFreeRecitation()` with mode and policy.
- [ ] `unsupported` reported per class from the shipped model's metadata.
- [ ] Web sample app for the user's own device testing.
- [ ] Docs credit tlog and QUD (CC-BY).
