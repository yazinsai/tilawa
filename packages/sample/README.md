# Tilawa SDK sample

A page for testing the SDK on your own phone: microphone or WAV input, memorization or free recitation, the policy settings, the passage with live position and mistakes, and per-device real-time factor. "Copy log JSON" copies the device, model, stats and every event.

## Run

Needs each registry tier's export under `<main checkout>/.scratch/models/` (`spliced-clean-int8`, `spliced-clean-lite-int8`; `modal run modal/student_export.py`, then `tools/quantize_onnx.py`), or `MODELS_DIR=<dir>`. `pnpm assets` links them, and a fixed set of 12 help eval clips (6 with a memorization error, 6 clean) for the device page; set `HELP_DIR` if the clips aren't in the main checkout.

```sh
pnpm install
cd packages/sample
pnpm dev            # https://<this Mac's LAN IP>:5173, self-signed
```

On the phone (same Wi-Fi): open the `Network:` URL Vite prints, accept the certificate warning (Safari: Show Details → visit this website; Chrome: Advanced → Proceed). HTTPS is what gives the page a secure context, which the microphone needs. Vite also sends COOP/COEP, so `crossOriginIsolated` is true and WASM threads above 1 work.

Then: Load model (the first load downloads 68 MB for `full` over the LAN; later loads come from the cache), pick the passage, Start microphone, recite.

Off the LAN: `npx cloudflared tunnel --url https://localhost:5173 --no-tls-verify` gives a public HTTPS URL, but the first download over a tunnel is slow (ticket 02 saw about 0.5 MB/s).

## Device measurements

`device.html` (linked from the sample page): pick a tier, provider and threads, Load model, then

1. **Clips**: the 12 clips as declared-passage sessions, pushed in 20 ms blocks as fast as the device goes. It reports RTF, push compute p95, word highlight and `suspected` latency (p50/p95, on the real-time clock model of `measureLatency`) and peak memory.
2. **Battery run**: the clips on a loop in real time for N minutes (default 10), with the screen kept awake. It reports RTF, latency and memory over the run, and battery % at start and end. Chrome on Android reads the level itself; elsewhere type it in from the system battery screen.

"Copy results JSON" copies everything, including the device, model, load time and whether the page was cross-origin isolated. Memory comes from `measureUserAgentSpecificMemory` (Chrome, cross-origin isolated; includes WASM). Otherwise it is the JS heap, which leaves out the model.

Headless desktop run: `pnpm build && pnpm preview --port 4318`, then `node scripts/device-headless.mjs --tier full` (needs `playwright`).
