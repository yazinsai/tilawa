# 02: Device probe page

**What to build:** A throwaway web page that answers "can a student-sized streaming model run in real time on WASM SIMD on the Android floor?" before distillation starts.

**Status:** ready-for-human

- [x] Static page loading an off-the-shelf sherpa-onnx streaming Zipformer, at around 25 MB int8 and around 100 MB, through onnxruntime-web WASM SIMD with no WebGPU.
- [x] Streams a bundled 60 s clip in chunks; reports real-time factor, p95 chunk latency, peak memory, and load time.
- [x] Optional WebGPU toggle for comparison.
- [ ] Hosted somewhere the user can open on a phone, with results copyable as JSON.
- [ ] Results recorded on this ticket. If `lite` misses 1× real time on the Android floor, reopen the tier sizes in PLAN.md.

## Comments

**Agent, 2026-10-06.** Page is in `packages/probe` (Vite + onnxruntime-web 1.30, own fbank and greedy transducer search, no sherpa-onnx wasm). Models and clips are fetched by a script and gitignored.

Model sizes differ from the ticket's ~25 MB / ~100 MB because those are the off-the-shelf streaming Zipformers that exist. Totals (encoder + decoder + joiner): `20m-int8` ≈ 42 MB, `70m-int8` (2023-06-26, chunk-16) ≈ 68 MB, `20m-fp32` ≈ 88 MB. The 20M int8 is the closest stand-in for `lite`. Its parameter count is right, but sherpa's int8 export leaves a lot in fp32. A distilled `lite` should be smaller and faster.

Sanity check: the English clip decodes to "AFTER EARLY NIGHTFALL THE YELLOW LAMPS WOULD LIGHT UP HERE AND THERE THE SQUALID QUARTER OF THE BRAFFLS", so features and decoding are correct. On the Quran clip the transcript is English garbage, which is expected. Only the timings matter there.

**Desktop numbers.** MacBook (Apple Silicon, 10 cores), headless Chrome 153, cross-origin isolated, 60 s Quran clip, 320 ms audio chunks:

| model | backend | threads | RTF | p95 chunk ms | max chunk ms | load ms | peak JS heap MB | UA memory MB |
|---|---|---|---|---|---|---|---|---|
| 20m-int8 | wasm | 1 | 0.044 | 14.4 | 25 | 506 | 63 | 140 |
| 20m-int8 | wasm | 4 | 0.032 | 13.7 | 58 | 1600 | 64 | 142 |
| 70m-int8 | wasm | 1 | 0.097 | 35.5 | 55 | 2664 | 99 | 201 |
| 70m-int8 | wasm | 4 | 0.061 | 24.9 | 53 | 2603 | 98 | 207 |
| 20m-fp32 | wasm | 1 | 0.041 | 14.4 | 38 | 3016 | 100 | 245 |
| 20m-fp32 | wasm | 4 | 0.035 | 15.3 | 47 | 3205 | 109 | 250 |
| 70m-int8 | webgpu | 1 | 0.493 | 168 | 930 | 1688 | 75 | 204 |
| 20m-int8 | webgpu | – | fails: `[WebGPU] Concat /Concat_13 non concat dimensions must match` | | | | | |

How to read these:
- The desktop has about 20× headroom for the 20M model. If the Android floor is about 10–15× slower single-threaded than an M-series core, which is a guess, `lite` lands around RTF 0.5–0.7. That is plausible but not safe. The phone run decides it.
- WebGPU is a net loss here. The joiner and decoder run once per output frame, so per-dispatch overhead dominates. The 20M export also hits an ORT WebGPU Concat bug. Neither affects the WASM-only floor requirement.
- "Peak JS heap" comes from `performance.memory` and excludes WASM memory. "UA memory" comes from `measureUserAgentSpecificMemory` and is the better number. It is Chrome-only and needs cross-origin isolation.
- Threads above 1 need cross-origin isolation (COOP/COEP headers on a secure origin). Without it the page falls back to 1 thread. The floor verdict should use the 1-thread row anyway, since thread counts vary by phone.

**Deploy steps (human).** The phone needs HTTPS for cross-origin isolation. Pick one option:

Option A, quickest (tunnel from this Mac, no account):
```sh
cd packages/probe
pnpm install
pnpm fetch-assets        # needs curl, tar, ffmpeg; ~250 MB into public/models, public/clips
pnpm build
pnpm preview --port 4317 &
npx cloudflared tunnel --url http://localhost:4317   # open the printed https://*.trycloudflare.com URL on the phone
```
`vite preview` already sends COOP/COEP (see `vite.config.ts`), and the tunnel passes them through.

Option B, persistent (Netlify honors `public/_headers`):
```sh
cd packages/probe && pnpm install && pnpm fetch-assets && pnpm build
npx netlify-cli deploy --dir dist --prod
```
Do not use Cloudflare Pages: its 25 MiB per-file limit rejects the 41–85 MB encoders. GitHub Pages can't set COOP/COEP, so it is 1 thread only.

**On the phone:** select `20m-int8`, WASM SIMD, threads 1, and the Quran 60 s clip. Run it twice, because the first load is cold. Copy the JSON and paste it here. Then repeat with the highest thread count offered, and with `70m-int8`. Check that `crossOriginIsolated` in the JSON is `true`.

**Still open for the human:**
- Hosting box: unchecked.
- Android floor (Snapdragon 7xx, 4 GB) and iPhone XR/11 results: unchecked.
- Verdict on `lite`: if `20m-int8` with 1 thread shows RTF ≥ 1, or p95 above about 320 ms, on the Android floor, reopen the tier sizes in PLAN.md.
