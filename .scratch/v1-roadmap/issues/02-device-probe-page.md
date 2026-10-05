# 02: Device probe page

**What to build:** A throwaway web page that answers "can a student-sized streaming model run in real time on WASM SIMD on the Android floor?" before distillation starts.

**Status:** ready-for-agent

- [ ] Static page loading an off-the-shelf sherpa-onnx streaming Zipformer, at around 25 MB int8 and around 100 MB, through onnxruntime-web WASM SIMD with no WebGPU.
- [ ] Streams a bundled 60 s clip in chunks; reports real-time factor, p95 chunk latency, peak memory, and load time.
- [ ] Optional WebGPU toggle for comparison.
- [ ] Hosted somewhere the user can open on a phone, with results copyable as JSON.
- [ ] Results recorded on this ticket. If `lite` misses 1× real time on the Android floor, reopen the tier sizes in PLAN.md.
