# 10: E7 quantize, export, device measurements

**What to build:** Quantized exports of the chosen students, measured on the device floor through the SDK.

**Blocked by:** 08, 09.

**Status:** ready-for-human

- [x] int8 (and fp16 where useful) exports; accuracy compared with fp32 on PER and the scripted-error test.
- [ ] On a physical Snapdragon 7xx with 4 GB, an iPhone XR or 11, and the last two versions of Chrome and Safari: real-time factor, memory, and battery over 10 minutes.
- [ ] Latency: word highlight p95 ≤ 600 ms; `suspected` within about 1 s.
- [x] `lite` runs in real time on the Android floor with WASM SIMD and no WebGPU (RTF 0.21, 4 GB Android, 2026-10-09).

## Comments

**Agent, 2026-10-07.** Branch `ticket-10-device`.

- **lite training estimate:** `spliced-clean-lite` = spliced-clean's recipe at size `lite` (22.3M), from scratch (there is no LibriSpeech lite init, so base LR stays at the scratch default 0.045), `--target spliced --kd-weight 2 --epochs 12`, gold exclusion on. spliced-clean took 3.84 H100 hours for 12 epochs at 64.6M; lite is the same data pipeline, so ≤ 3.8 h. Modal H100 $3.95/h + 16 CPU ($0.76/h) + 128 GiB ($1.02/h) ≈ $5.7/h → **≈ $22 worst case**, under the approved $30.

**Agent, 2026-10-07: results.** Everything except the physical-device runs is done. Shipped: `full` = spliced-clean int8, `lite` = spliced-clean-lite int8 (`packages/sdk/src/models-v1.json`).

Accuracy (onnxruntime-node via the SDK, 20 ms pushes; scoreboard rows `spliced-clean-{onnx,int8,fp16}`, `spliced-clean-lite-{onnx,int8}`):

| Variant | Size | Gold PER (bar 3.65%) | Declared P / R | Free P / R | Node RTF |
|---|---|---|---|---|---|
| spliced-clean PyTorch | — | 4.71% | | | |
| spliced-clean fp32 | 259 MB | 4.80% | 83.6 / 87.5 | 80.0 / 43.8 | 0.069 |
| **spliced-clean int8 (full)** | 67.8 MB | 4.82% | 82.4 / 87.5 | 77.8 / 43.8 | 0.049 |
| spliced-clean fp16 | 129.6 MB | 4.79% | 83.6 / 87.5 | 80.0 / 43.8 | 0.071 |
| spliced-clean-lite fp32 | 89.1 MB | 5.75% | 70.0 / 87.5 | 63.4 / 40.6 | 0.046 |
| **spliced-clean-lite int8 (lite)** | 24.7 MB | 5.74% | 70.0 / 87.5 | 63.4 / 40.6 | 0.040 |

Weight-only 8-bit (`MatMulNBits`, 75 MB) was measured and dropped: RTF 0.256 in Chrome WASM, 2× int8. fp16 is faster in Chrome (0.096) but peaks at 659 MB page memory vs 252 MB, so int8 is `full`. Neither tier reaches the 3.65% gold bar (open since ticket 08).

Desktop browsers (M-series Mac, WASM SIMD, 1 thread, 12 help clips / 106.8 s, 20 ms pushes, `scripts/device-headless.mjs`):

| Browser | Tier | Load | RTF | Push p95 | Highlight p50 / p95 | Suspected p95 | Peak memory |
|---|---|---|---|---|---|---|---|
| Chrome 154 | full | 3.0 s | 0.123 | 31 ms | 364 / 513 ms | 1206 ms | 252 MB |
| Chrome 154 | lite | 1.2 s | 0.073 | 15 ms | 350 / 481 ms | 902 ms | 128 MB |
| WebKit 18.2 | full | 3.5 s | 0.122 | 27 ms | 370 / 530 ms | 1212 ms | n/a |
| WebKit 18.2 | lite | 2.1 s | 0.088 | 16 ms | 357 / 497 ms | 907 ms | n/a |

Latency over all help clips (node, n ≈ 4,700 words / 153 errors): highlight p95 490–495 ms (bar 600, pass), suspected p95 1010–1020 ms (bar "about 1 s", borderline). Highlight is timed from the decoded phoneme, so it excludes the ~200 ms the model needs before it emits a phoneme.

Cadence: the tracker acquires on a fixed 1 s evidence-time grid (`acquireMs`), the checker anchors on locks reached only at `end()`, and the SDK feeds evidence in 40 ms cells instead of ≥1 s batches. Events are identical for 40/80/160/320 ms/1 s chunking (`packages/core/test/cadence.test.ts`, `packages/sdk/test/session.test.ts`).

Spend: lite training 2.62 H100-hours ≈ $15 (estimate $22); export and gold decode < $1.

**Device checklist (human).** Phone and laptop on the same network.
1. `cd packages/sample && pnpm install && pnpm assets && pnpm build && pnpm preview --host --port 4318` (HTTPS via the sample's dev cert; or `cloudflared tunnel --url https://localhost:4318 --no-tls-verify` and use that URL).
2. Open `https://<laptop-ip>:4318/device.html`. Unplug the charger, close other apps, screen brightness about 50%.
3. Tier `lite`, provider `wasm`, threads `1`, Load. Wait for "loaded".
4. Run clips. Note RTF; it must be < 1 (lite real-time box).
5. Run battery (10 min). Keep the screen on (the page holds a wake lock). On iOS/Safari type the battery % at start and end when asked.
6. Copy results JSON and paste it into this ticket under a heading naming the device.
7. Reload, repeat steps 3–6 with tier `full`.
8. Devices: Snapdragon 7xx with 4 GB (Chrome), iPhone XR or 11 (Safari), desktop Chrome and Safari, each in its two latest versions.
9. Tick the boxes: RTF < 1 for lite on Android; highlight p95 ≤ 600 ms and suspected p95 ≈ ≤ 1 s everywhere. If `full` is over 1 RTF on the iPhone, say so; the registry would then default slow devices to `lite`.

**Human, 2026-10-09: Android phone (Android 10, 4 GB, 8 cores, Chrome 154; SoC not confirmed as Snapdragon 7xx).** Phone test page, WASM, 1 thread.

| Tier | Load | RTF (clips) | RTF (10 min real time) | Push p95 | Highlight p95 | Suspected p95 (n) | Peak memory |
|---|---|---|---|---|---|---|---|
| lite | 98.2 s (cache) | 0.208 | 0.436 | 43 ms | 568 ms ✓ | 943 ms ✓ (9) | 121 MB |
| full | 146.3 s (download) | 0.294 | — | 73 ms | 615 ms ✗ | 1267 ms ✗ (8) | 251 MB |

- lite runs in real time on this phone with margin; full also keeps up but misses both latency bars, so this device class defaults to `lite`.
- Battery: 100% → 100%, unusable (phones hold 100% for a while off the charger). Rerun starting below 90%.
- Load: 98 s for lite **from cache** is a product blocker on its own; needs profiling (WASM compile vs session creation vs cache read).
- Suspected latency rests on 8–9 errors; treat as indicative.

Raw: `{"schema":2,"date":"2026-10-09T13:59:08.366Z","device":{"userAgent":"Mozilla/5.0 (Linux; Android 10; K) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/154.0.0.0 Mobile Safari/537.36","hardwareConcurrency":8,"deviceMemoryGB":4},"tiers":{"lite":{"model":{"id":"spliced-clean-lite","version":"1-int8","bytes":24650054,"backend":"onnxruntime-web/wasm","fallbacks":[],"loadMs":98235,"fromCache":true},"pushMs":20,"audioS":106.8,"rtf":0.208,"pushComputeMs":{"n":5340,"p50":0,"p95":43,"max":171},"highlightMs":{"n":55,"p50":401,"p95":568,"max":585},"suspectedMs":{"n":9,"p50":49,"p95":943,"max":943},"peakMemoryMB":121},"full":{"model":{"id":"spliced-clean","version":"1-int8","bytes":67759744,"backend":"onnxruntime-web/wasm","fallbacks":[],"loadMs":146256,"fromCache":false},"pushMs":20,"audioS":106.8,"rtf":0.294,"pushComputeMs":{"n":5340,"p50":0,"p95":73,"max":199},"highlightMs":{"n":59,"p50":432,"p95":615,"max":828},"suspectedMs":{"n":8,"p50":164,"p95":1267,"max":1267},"peakMemoryMB":251}},"battery":{"minutes":10.1,"rtf":0.436,"batteryPct":{"start":100,"end":100,"per10min":0}}}`
