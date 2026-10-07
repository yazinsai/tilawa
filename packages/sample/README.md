# Tilawa SDK sample

A page for testing the SDK on your own phone: microphone or WAV input, memorization or free recitation, the policy settings, the passage with live position and mistakes, and per-device real-time factor. "Copy log JSON" copies the device, model, stats and every event.

## Run

Needs the exported model at `<main checkout>/.scratch/models/spliced-clean/` (`modal run modal/student_export.py --run spliced-clean`), or `MODEL_DIR=<dir>`.

```sh
pnpm install
cd packages/sample
pnpm dev            # https://<this Mac's LAN IP>:5173, self-signed
```

On the phone (same Wi-Fi): open the `Network:` URL Vite prints, accept the certificate warning (Safari: Show Details → visit this website; Chrome: Advanced → Proceed). HTTPS is what gives the page a secure context, which the microphone needs. Vite also sends COOP/COEP, so `crossOriginIsolated` is true and WASM threads above 1 work.

Then: Load model (the first load downloads 258 MB over the LAN; later loads come from the cache), pick the passage, Start microphone, recite.

Off the LAN: `npx cloudflared tunnel --url https://localhost:5173 --no-tls-verify` gives a public HTTPS URL, but the 258 MB first download over a tunnel is slow (ticket 02 saw about 0.5 MB/s).
