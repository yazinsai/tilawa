# Tilawa SDK harness

A minimal page for testing the current Tilawa web SDK on desktop or a phone. It imports `@tilawa/sdk/web` and the core directly from this checkout, so edits to the tracker, checker or SDK are picked up by Vite. The default path matches `/live`: full `spliced-clean-int8` listener, WebGPU with WASM fallback, one thread, free recitation, recorded evidence, and microphone VAD off. The browser uses the single offline student model; the auxiliary teacher is not loaded.

Click **Start recording**. The page loads the main model automatically and chooses the browser runtime. To test a file, expand **Test an existing recording**, choose a WAV, and click **Play and check**. WAV playback is paced in real time by default; uncheck that option for a faster tuning run. Replay again after changing settings to compare the same recording. **Stop** also stops file replay. Microphone takes stop automatically at three minutes.

The passage shows the current ayah and tracked word. Feedback shows each check's current state (`suspected`, `confirmed`, `resolved`, `uncertain`), while the counted mistake total comes from `session.mistakes()`. The collapsed **Downloads and technical details** section shows the best three position candidates, heard phonemes, RTF, and push p95. RTF excludes queue wait and includes the final session drain; these are harness timings, not the SDK's word-highlight latency benchmark.

The collapsed **Tuning settings** controls apply to the next take: free or declared-passage mode, confirmation delay/next word, minimum confidence, evidence cell size, self-correction counting, and microphone VAD. Defaults are 2500 ms, 0.6 confidence, and 40 ms evidence cells. Model tier, provider and thread choices are automatic on this page; device measurements remain available separately.

**Download test report** (or **Copy test report**) exports the build revision, model/backend, session policy, capabilities, stats, every event with arrival time, mistakes and recorded phoneme/posterior evidence. **Download recording** exports the exact mono PCM supplied to the SDK as a float WAV, using the same take ID. With VAD enabled this is the gated audio. Files stay local; there is no upload endpoint.

Harness settings do not alter SDK defaults. Any subsequent listener/tracker/checker tuning that can change a lab score must follow the experiment workflow in `AGENTS.md`.

## Run

Needs each registry tier's export under `<main checkout>/.scratch/models/` (`spliced-clean-int8`, `spliced-clean-lite-int8`; `modal run modal/student_export.py`, then `tools/quantize_onnx.py`), or `MODELS_DIR=<dir>`. `pnpm assets` links them, and a fixed set of 12 help eval clips (6 with a memorization error, 6 clean) for the device page; set `HELP_DIR` if the clips aren't in the main checkout.

```sh
pnpm install
cd packages/sample
pnpm dev            # https://<this Mac's LAN IP>:5173, self-signed
```

On the phone (same Wi-Fi): open the `Network:` URL Vite prints, accept the certificate warning (Safari: Show Details → visit this website; Chrome: Advanced → Proceed). HTTPS is what gives the page a secure context, which the microphone needs. Vite also sends COOP/COEP, so `crossOriginIsolated` is true and WASM threads above 1 work.

Then: click Start recording and recite. The first start downloads the 68 MB main model over the LAN; later starts use the cache. Free recitation needs no passage declaration.

Off the LAN: `npx cloudflared tunnel --url https://localhost:5173 --no-tls-verify` gives a public HTTPS URL, but the first download over a tunnel is slow (ticket 02 saw about 0.5 MB/s).

## Device measurements

`device.html` (linked from the sample page): pick a tier, provider and threads, Load model, then

1. **Clips**: the 12 clips as declared-passage sessions, pushed in 20 ms blocks as fast as the device goes. It reports RTF, push compute p95, word highlight and `suspected` latency (p50/p95, on the real-time clock model of `measureLatency`) and peak memory.
2. **Battery run**: the clips on a loop in real time for N minutes (default 10), with the screen kept awake. It reports RTF, latency and memory over the run, and battery % at start and end. Chrome on Android reads the level itself; elsewhere type it in from the system battery screen.

"Copy results JSON" copies everything, including the device, model, load time and whether the page was cross-origin isolated. Memory comes from `measureUserAgentSpecificMemory` (Chrome, cross-origin isolated; includes WASM). Otherwise it is the JS heap, which leaves out the model.

Headless desktop run: `pnpm build && pnpm preview --port 4318`, then `node scripts/device-headless.mjs --tier full` (needs `playwright`).
