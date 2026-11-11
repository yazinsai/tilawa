# Self-hosting and offline use

By default, `@tilawa/core/web` downloads the model, the Quran reference and the ONNX Runtime files from `https://tilawa.dev/v1/`, once, and keeps them in the browser's Cache API. You don't need anything on this page to get started.

Read on if you want to:

- serve the files from your own origin or CDN,
- make your app work without any request to tilawa.dev, or
- use an onnxruntime-web version that tilawa.dev doesn't host.

## The `/v1/` layout

```
v1/
  manifest.json                                   every file below: path, bytes, sha256
  models/spliced-clean/1-int8/model.onnx          full tier, 67,759,744 bytes
  models/spliced-clean/1-int8/manifest.json       its model manifest
  models/spliced-clean-lite/1-int8/model.onnx     lite tier, 24,650,054 bytes
  models/spliced-clean-lite/1-int8/manifest.json
  reference/ref-v2.json                           the Quran reference, 4,749,281 bytes
  runtime/ort/<version>/ort-wasm-simd-threaded*   onnxruntime-web's .wasm and .mjs files
```

- Paths are `models/<id>/<version>/<file>`. The SDK's bundled registry (`MODEL_REGISTRY`) holds each tier's relative path, size and sha256.
- Every file except the top-level `manifest.json` is immutable. A new model, reference or runtime gets a new path. Nothing is overwritten.
- The model `manifest.json` describes the model's inputs, states and phoneme inventory. The browser SDK reads it from its bundled registry, not from the server. Node's `modelDir` option reads it from disk (download both files of a tier into one directory).

`manifest.json` at the top looks like this:

```json
{
 "schema": 1,
 "ort": "1.30.0",
 "files": [
  { "path": "models/spliced-clean/1-int8/model.onnx", "bytes": 67759744, "sha256": "714e98ad…" },
  …
 ]
}
```

`ort` is the onnxruntime-web version the tree was built for. `files` lists every file, including every `runtime/ort/<version>/` folder present.

## Point the SDK at your mirror

```ts
import { createTilawa } from "@tilawa/core/web";

const tilawa = await createTilawa({
  // A mirror of https://tilawa.dev/v1/. Models, the reference and the onnxruntime-web files all load from here.
  assetBase: "https://cdn.example.com/tilawa/v1/",
});
console.log(tilawa.model.backend);
```

With `assetBase` set, the SDK fetches:

- the model from `<assetBase>models/<id>/<version>/model.onnx`,
- the reference from `<assetBase>reference/ref-v2.json`,
- the onnxruntime-web files from `<assetBase>runtime/ort/<version>/`, where `<version>` is the onnxruntime-web version in your bundle. Pass `wasmPaths` to load them from somewhere else.

On Node, `assetBase` only moves the model download: the reference ships inside the package.

### Mirror tilawa.dev/v1

Copy the files listed in `manifest.json` and check each one:

```ts
import { createHash } from "node:crypto";
import { mkdir, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";

interface AssetManifest {
  schema: 1;
  ort: string;
  files: { path: string; bytes: number; sha256: string }[];
}

const source = "https://tilawa.dev/v1/";
const out = "public/tilawa/v1"; // served by your app as /tilawa/v1/

const manifest = (await (await fetch(source + "manifest.json")).json()) as AssetManifest;
for (const file of manifest.files) {
  const res = await fetch(source + file.path);
  if (!res.ok) throw new Error(`${res.status} fetching ${file.path}`);
  const bytes = new Uint8Array(await res.arrayBuffer());
  const sha256 = createHash("sha256").update(bytes).digest("hex");
  if (bytes.length !== file.bytes || sha256 !== file.sha256) throw new Error(`Checksum mismatch for ${file.path}`);
  await mkdir(dirname(join(out, file.path)), { recursive: true });
  await writeFile(join(out, file.path), bytes);
}
await writeFile(join(out, "manifest.json"), JSON.stringify(manifest, null, 1));
console.log(`Mirrored ${manifest.files.length} files (onnxruntime-web ${manifest.ort})`);
```

You can skip the tier you don't use, but `tier: "auto"` may pick either one.

### A different onnxruntime-web version

If your app bundles an onnxruntime-web version that the mirror doesn't have, copy that version's runtime files from your own `node_modules/onnxruntime-web/dist/` into `runtime/ort/<your version>/`. The files are the `ort-wasm-simd-threaded*.wasm` and `.mjs` files that `onnxruntime-web/all` loads. Or put them anywhere and pass `wasmPaths`.

In this repository, `web/frontend/scripts/prepare-v1-assets.mjs` builds the whole tree, runtime included, from the locked onnxruntime-web version, and checks every model against the registry.

## Fully offline

The app can avoid any network request for Tilawa files by serving them itself and bundling the reference:

```ts
import { MODEL_REGISTRY, withBase } from "@tilawa/core";
import type { ReferenceAsset } from "@tilawa/core";
import reference from "@tilawa/core/assets/ref-v2.json" with { type: "json" };
import { cacheStorage, createTilawa, loadWebModel } from "@tilawa/core/web";

const base = "/tilawa/v1/"; // served by your app, same layout as https://tilawa.dev/v1/

const model = await loadWebModel({
  tier: "lite",
  registry: withBase(MODEL_REGISTRY, base),
  storage: cacheStorage("tilawa-models"),
  wasmPaths: `${base}runtime/ort/1.30.0/`, // the onnxruntime-web version your app installs
});
const tilawa = await createTilawa({ reference: reference as unknown as ReferenceAsset, model });

// ...sessions as usual...

await tilawa.dispose();
await model.backend.release(); // you loaded the model, so you release it
```

Notes:

- `reference` skips the reference download. `ref-v2.json` is 4.7 MB, so bundling it grows your bundle by that much. Serving it from `assetBase` and letting the SDK cache it is often better.
- When you pass `model`, `createTilawa` doesn't load one, and its `wasmPaths`, `providers` and `threads` options are unused. Give them to `loadWebModel`.
- `loadWebModel` defaults to `tier: "full"` and to the registry's relative paths, which resolve against the page. Use `withBase` to point them at your files.
- You own a model you loaded: `tilawa.dispose()` doesn't release it.
- After the first load, everything is in the Cache API, so the simple `assetBase` setup also works offline as long as the browser keeps the cache. Browsers can evict it under storage pressure. Call `navigator.storage.persist()` if your app must work offline.
- On Node, the reference is bundled already. To avoid the model download, pass `modelDir` (a directory with `model.onnx` and `manifest.json` of one tier) or pre-fill `cacheDir`.

## Headers

Serve the files with these headers. `https://tilawa.dev/v1/` does.

| Header | Value | Why |
|---|---|---|
| `Content-Type` | `.wasm`: `application/wasm`. `.mjs`/`.js`: `text/javascript`. `.json`: `application/json`. `.onnx`: `application/octet-stream`. | Browsers refuse to compile WASM streamed with the wrong type, and refuse module scripts with the wrong type. |
| `Cache-Control` | `public, max-age=31536000, immutable` for versioned files. `no-cache` for the top-level `manifest.json`. | The files never change. The top-level manifest does when versions are added. |
| `Access-Control-Allow-Origin` | `*` (or your app's origin) | When the files are on a different origin from the app. |
| `Cross-Origin-Resource-Policy` | `cross-origin` | When the app's page is cross-origin isolated (COEP `require-corp`) and the files are on another origin. Without it, the browser blocks them. |
| `X-Content-Type-Options` | `nosniff` | Recommended. |

Range requests (`Accept-Ranges: bytes`) are supported by tilawa.dev but not required by the SDK.

The SDK downloads the model and the reference with a plain `fetch(url)`, so the files must be readable without cookies or auth headers.

## Checksums and caching

- **Models.** After a download, the SDK checks the length, then the sha256, against the bundled registry. A mismatch rejects with `ModelIntegrityError` and nothing is cached. A cached model is checked by length only on later loads.
- **Reference.** The same: length and sha256 on download, length on a cache hit. On Node the bundled file is read as is.
- **Without WebCrypto** (for example a plain-http page, or React Native), the sha256 check is skipped and only the length is checked.
- **Upgrades.** When a new version of a model id is stored, older versions of that id are deleted from the cache.
- **Budget.** Before downloading, the SDK checks that this tier plus the other tiers already cached fit in 200 MiB (`MODEL_BUDGET_BYTES`). If not, it rejects with `ModelBudgetError`.
- **Where.** Web: the Cache API cache `tilawa-models`. Node: `~/.cache/tilawa/<id>@<version>/model.onnx` (or `cacheDir`, or `$TILAWA_CACHE_DIR`). React Native: the `storage` you pass, in memory by default.
- **Your own store.** Any object with async `get`, `put`, `delete` and `keys` is a `ModelStorage`. Pass it as `storage`.
- **The ONNX Runtime files** are fetched by onnxruntime-web itself. They aren't checksummed by the SDK; the browser's HTTP cache keeps them.

To clear the browser cache from your app: `await caches.delete("tilawa-models")`.
