# Tilawa web frontend (tilawa.dev)

Vanilla TypeScript + Vite 7, served by a small Hono server (`server/`).

- `/` is the demo (`index.html`, `src/`). It runs `@tilawa/core/web` with no options except a same-origin `assetBase`: one Start button, free recitation, the current ayah and word, and confirmed memorization mistakes as notes. "Practise a passage" starts a declared-passage session.
- `/help` is the mistake-recording site (`help/`).
- `/v1/*` serves the immutable SDK assets (models, reference, onnxruntime-web runtime) from `ASSETS_V1_DIR` (default `dist/v1`).

`@tilawa/core` is a `file:../../packages/core` dependency, so the demo uses the built package. Build it first.

```bash
(cd ../../packages/core && npm ci && npm run build)
npm ci
npm run build                 # tsc + vite build: / and /help (BUILD_ONLY=help for just /help)
npm run assets:v1 -- --models-dir <exports>   # dist/v1, after the build (vite empties dist/)
npm run build:server && npm start             # http://localhost:5000
npm run dev                   # vite dev server; serves dist/v1 at /v1 too
```

## /v1 assets

`scripts/prepare-v1-assets.mjs` writes `models/<id>/<version>/{model.onnx,manifest.json}` for every tier in `packages/core/src/runtime/models-v1.json`, `reference/ref-v2.json`, `runtime/ort/<version>/` (the onnxruntime-web files `@tilawa/core/web` loads, for the version in `package-lock.json`) and `manifest.json` (path, bytes, sha256 of each). Models come from `--models-dir` (`<id>-<suffix>/model.onnx`, e.g. `spliced-clean-int8/`) or `--models-url` (a `/v1/` mirror, or a template with `{id}` and `{version}`). Each is checked against the registry's sha256. Existing files are never overwritten with different bytes.

The demo bundles onnxruntime-web's non-inlined build, so its `.mjs`/`.wasm` load only from `/v1/runtime/ort/<version>/`. Keep one onnxruntime-web version (`package.json` pins it) so the bundle and that folder agree.

## Tests

```bash
npx vitest run                # deterministic: server, /help, demo helpers, asset script, old-client cleanup
TILAWA_MODELS_DIR=<exports> npx playwright test   # opt-in end to end (see playwright.config.ts)
```

The Playwright test builds everything, prepares `dist/v1`, starts the built server and recites `lab/benchmark/test_corpus/multi_036_001_005.wav` (Yā-Sīn 36:1–5) into Chromium's fake microphone. Screenshots go to `test/demo-screens/`.

## Old clients

Browsers that ran the 0.x demo have a ~66 MB model in the IndexedDB database `tarteel-models`, and possibly a service worker. `src/migrate.ts` deletes that database, unregisters service workers and drops every Cache API cache except the SDK's `tilawa-models`. `public/sw.js` is the old service worker's kill switch. The 0.x `/api/reports` and `/api/diagnostics` POST routes stay on the server for old clients.
