// docs/self-hosting.md: "Fully offline: bundle the reference, load the model yourself"
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
