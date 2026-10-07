# 10: E7 quantize, export, device measurements

**What to build:** Quantized exports of the chosen students, measured on the device floor through the SDK.

**Blocked by:** 08, 09.

**Status:** claimed

- [ ] int8 (and fp16 where useful) exports; accuracy compared with fp32 on PER and the scripted-error test.
- [ ] On a physical Snapdragon 7xx with 4 GB, an iPhone XR or 11, and the last two versions of Chrome and Safari: real-time factor, memory, and battery over 10 minutes.
- [ ] Latency: word highlight p95 ≤ 600 ms; `suspected` within about 1 s.
- [ ] `lite` runs in real time on the Android floor with WASM SIMD and no WebGPU.

## Comments

**Agent, 2026-10-07.** Branch `ticket-10-device`.

- **lite training estimate:** `spliced-clean-lite` = spliced-clean's recipe at size `lite` (22.3M), from scratch (there is no LibriSpeech lite init, so base LR stays at the scratch default 0.045), `--target spliced --kd-weight 2 --epochs 12`, gold exclusion on. spliced-clean took 3.84 H100 hours for 12 epochs at 64.6M; lite is the same data pipeline, so ≤ 3.8 h. Modal H100 $3.95/h + 16 CPU ($0.76/h) + 128 GiB ($1.02/h) ≈ $5.7/h → **≈ $22 worst case**, under the approved $30.
