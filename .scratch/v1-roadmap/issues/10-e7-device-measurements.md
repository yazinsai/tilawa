# 10: E7 quantize, export, device measurements

**What to build:** Quantized exports of the chosen students, measured on the device floor through the SDK.

**Blocked by:** 08, 09.

**Status:** needs-triage

- [ ] int8 (and fp16 where useful) exports; accuracy compared with fp32 on PER and the scripted-error test.
- [ ] On a physical Snapdragon 7xx with 4 GB, an iPhone XR or 11, and the last two versions of Chrome and Safari: real-time factor, memory, and battery over 10 minutes.
- [ ] Latency: word highlight p95 ≤ 600 ms; `suspected` within about 1 s.
- [ ] `lite` runs in real time on the Android floor with WASM SIMD and no WebGPU.
