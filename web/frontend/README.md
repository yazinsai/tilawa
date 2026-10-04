# Tilawa web frontend

Vanilla TypeScript + Vite 7. The demo uses Zipformer2-CTC (`a0w-ep1-a0.5` int8) for both Tracking and Correction. The previous default, `interp-gentle-a0.5` int8, is still fetched for comparison.

## Zipformer (`a0w-ep1-a0.5`)

Streaming Zipformer2-CTC. The status pill shows the active engine. Model artefacts are NPL-1.2; the word-level tracker is the native MIT recitation engine, written from `lab/docs/specs/recitation-engine-spec.md` plus 23 vector oracles. It now lives in the SDK (`packages/core/src/recitation/`) and the worker here is a thin host over `ZipformerSession` from `@tilawa/core`.

Assets are gitignored (ONNX + NPL-derived lexicon). The `*.io.json` manifests are committed. Fetch the rest once:

```bash
bash web/frontend/scripts/fetch-zipformer-assets.sh
# copies from a local export / the main checkout public/ tree, else downloads
# zipformer_a0w_ep1_a05.int8.onnx from GitHub release zipformer-a0w-ep1-a0.5,
# zipformer_interp_gentle_a05.int8.onnx and zipformer_quran.json from v0.3.0
```

Then from `web/frontend` (symlink `node_modules` from the main checkout if you are in a worktree):

```bash
npm run dev
# open http://localhost:5173/
```

Node smoke (onnxruntime-node, no browser):

```bash
cd web/frontend
npx tsx test/zipformer-node-smoke.ts
```

Streaming stability:

```bash
npx tsx test/stability-report.ts --repeats=3 --json=test/track-c-v1-stability.json
npx tsx test/stability-report.ts --repeats=3 --corpus=test_corpus_v2 --json=test/track-c-v2-stability.json
npx tsx test/stability-report.ts --repeats=1 --corpus=test_corpus_v3 --json=test/track-c-v3-stability.json
```

int8 ONNX sha256 `eaf099af…` (66 MB). Threads stay off (`numThreads=1`, EP `wasm`). First load is ~66 MB into IndexedDB under `zipformer-interp-gentle-a05-int8`.
