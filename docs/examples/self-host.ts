// docs/self-hosting.md: "Point the SDK at your mirror"
import { createTilawa } from "@tilawa/core/web";

const tilawa = await createTilawa({
  // A mirror of https://tilawa.dev/v1/. Models, the reference and the onnxruntime-web files all load from here.
  assetBase: "https://cdn.example.com/tilawa/v1/",
});
console.log(tilawa.model.backend);
